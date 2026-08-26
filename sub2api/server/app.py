"""HTTP gateway: the single consumer of the normalized event stream.

The server knows nothing about any specific agent backend — it only
speaks ChatRequest/Event. All channel-specific behavior lives behind
the registry.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__, channels  # noqa: F401  (channels = registration)
from ..core.aggregate import collect_answer
from ..core.channel import ChannelConfig
from ..core.config import AppSettings
from ..core.errors import ChannelMismatchError, SessionNotFoundError
from ..core.registry import available_channels, create_channel, get_channel_class
from ..core.sessions import Session, SessionStore
from ..core.types import TERMINAL_EVENTS, ChatRequest, Event, SessionStarted, TurnCompleted

PermissionMode = Literal["default", "acceptEdits", "yolo"]


class ChatBody(BaseModel):
    prompt: str = Field(min_length=1)
    session_id: str | None = None
    model: str | None = None
    permission_mode: PermissionMode = "default"
    enabled_tools: list[str] | None = None
    workspace: str | None = None
    images: list[str] | None = None
    context_files: list[str] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def error_response(status_code: int, code: str, message: str, **extra: Any) -> JSONResponse:
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    payload["error"].update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def sse_frame(event: Event) -> str:
    return f"data: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or AppSettings.from_env()
    channel = create_channel(settings.channel, settings.channel_config)
    store = SessionStore()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await channel.start()
        yield
        await channel.close()

    app = FastAPI(
        title="sub2api",
        description="Subscription-to-API gateway with pluggable agent channels.",
        version=__version__,
        lifespan=lifespan,
    )
    # Exposed for tests and introspection.
    app.state.settings = settings
    app.state.channel = channel
    app.state.store = store

    def resolve_session(body: ChatBody) -> tuple[Session, JSONResponse | None]:
        """Common session logic: reuse or create, validate channel match."""
        if body.session_id:
            try:
                session = store.get(body.session_id)
            except SessionNotFoundError as exc:
                return None, error_response(404, exc.code, str(exc))
            if session.channel != channel.name:
                return None, error_response(
                    409,
                    ChannelMismatchError.code,
                    f"session belongs to channel `{session.channel}`, not `{channel.name}`",
                )
            return session, None
        return store.create(channel.name), None

    def build_request(body: ChatBody, session: Session) -> ChatRequest:
        return ChatRequest(
            prompt=body.prompt,
            session_id=session.id,
            native_session_id=session.native_session_id,
            model=body.model,
            workspace=body.workspace,
            permission_mode=body.permission_mode,
            enabled_tools=tuple(body.enabled_tools) if body.enabled_tools else None,
            images=tuple(body.images) if body.images else None,
            context_files=tuple(body.context_files) if body.context_files else None,
            extra=body.extra,
        )

    @app.post("/v1/chat")
    async def chat(body: ChatBody):
        session, failure = resolve_session(body)
        if failure is not None:
            return failure
        request = build_request(body, session)
        answer, terminal = await collect_answer(channel.chat(request))
        store.touch(session)
        if isinstance(terminal, TurnCompleted):
            store.adopt_native(session, terminal.session_id, terminal.model)
        elif terminal is None or terminal.type == "turn.failed":
            detail = terminal.message if terminal else "stream ended without a terminal event"
            code = terminal.code if terminal else "internal_error"
            return error_response(502, code, detail, session_id=session.id)
        return {
            "answer": answer,
            "session_id": session.id,
            "channel": channel.name,
            "model": getattr(terminal, "model", None) or body.model,
        }

    @app.post("/v1/chat/stream")
    async def chat_stream(body: ChatBody):
        session, failure = resolve_session(body)
        if failure is not None:
            return failure
        request = build_request(body, session)

        async def stream() -> AsyncIterator[str]:
            yield sse_frame(SessionStarted(session_id=session.id, channel=channel.name))
            async for event in channel.chat(request):
                if isinstance(event, TurnCompleted):
                    store.adopt_native(session, event.session_id, event.model)
                yield sse_frame(event)
            store.touch(session)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/v1/channels")
    async def channels():
        configured = channel.name
        listing = []
        for name in available_channels():
            cls = get_channel_class(name)
            listing.append(
                {
                    "name": name,
                    "display_name": cls.display_name or name,
                    "models": list(cls.models),
                    "configured": name == configured,
                }
            )
        return {"channels": listing, "active": await channel.health()}

    @app.get("/v1/sessions/{session_id}")
    async def session_info(session_id: str):
        try:
            session = store.get(session_id)
        except SessionNotFoundError as exc:
            return error_response(404, exc.code, str(exc))
        return session.to_dict()

    @app.get("/healthz")
    async def healthz():
        return {
            "status": "ok",
            "version": __version__,
            "channel": await channel.health(),
        }

    return app
