"""Shared per-app state and helpers for the route modules.

``create_app`` used to define every route as a closure over ``settings``,
``channel`` and ``store``.  Splitting the routes into modules needs those
three reachable without closures, so they travel as one frozen
:class:`AppContext` built once in ``create_app`` and handed to each
``router(ctx)`` factory.

Nothing that changes after startup lives here.  ``app.state.usage_store`` in
particular is opened by the lifespan and is ``None`` while the routers are
being constructed, so :func:`make_meter` takes the live ``Request`` and reads
it at call time instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..compat.errors import anthropic_error
from ..core.config import AppSettings
from ..core.errors import ChannelMismatchError, SessionNotFoundError
from ..core.sessions import Session, SessionStore
from ..core.types import ChatRequest, Event
from . import openai as oai
from .proxy import UsageMeter

PermissionMode = Literal["default", "acceptEdits", "yolo"]


class ChatBody(BaseModel):
    prompt: str = Field(min_length=1)
    session_id: str | None = None
    model: str | None = None
    permission_mode: PermissionMode = "default"
    enabled_tools: list[str] | None = None
    workspace: str | None = None
    images: list[str] | None = None
    thinking_effort: str | None = None


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = None


class CompletionBody(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    stream: bool = False
    thinking_effort: str | None = None
    reasoning_effort: str | None = None  # OpenAI-compat alias for thinking_effort


def error_response(
    status_code: int, code: str, message: str, **extra: Any
) -> JSONResponse:
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    payload["error"].update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def passthrough_json(resp: httpx.Response) -> Any:
    """Safely extract JSON from an upstream response.

    The proxy may return non-JSON bodies on error (HTML, plain text, empty),
    which would crash ``resp.json()``.  Fall back to the raw text so the
    client still sees the status code and body.
    """
    if not resp.content:
        return {}
    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError):
        return {"raw": resp.text[:2000]}


def sse_frame(event: Event) -> str:
    return f"data: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"


def parse_body(raw: bytes) -> tuple[dict[str, Any], JSONResponse | None]:
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}, JSONResponse(
            status_code=400,
            content=oai.openai_error(
                "invalid JSON body",
                err_type="invalid_request_error",
                code="bad_request",
            ),
        )
    return (body if isinstance(body, dict) else {}), None


def model_not_found(model: str, dialect: str) -> JSONResponse:
    """404 for an unroutable model, in the caller's own error dialect.

    This is what ``api.anthropic.com`` answers for a model it does not
    host; the alternative — forwarding to whichever provider happens to
    be the default — produces an opaque upstream 500.
    """
    if dialect == "anthropic":
        return JSONResponse(
            status_code=404,
            content=anthropic_error(f"model `{model}` not found", "not_found_error"),
        )
    return JSONResponse(
        status_code=404,
        content=oai.openai_error(
            f"model `{model}` not found",
            err_type="invalid_request_error",
            code="model_not_found",
        ),
    )


def wants_usage_chunk(body: dict[str, Any]) -> bool:
    """True when the client itself asked for the trailing usage chunk.

    sub2api always asks upstream for it (that is the only way to meter an
    OpenAI stream), so when the client did not, the chunk is stripped
    from the relayed stream to keep the response byte-shape the client
    expects.
    """
    options = body.get("stream_options")
    return isinstance(options, dict) and bool(options.get("include_usage"))


def make_meter(
    request: Request, route: Any, sid: str, path: str, stream: bool
) -> UsageMeter:
    """One metering context per request; a no-op when no store is open."""
    return UsageMeter(
        store=request.app.state.usage_store,
        session_id=sid,
        model=route.model,
        provider=route.provider,
        protocol=route.protocol,
        path=path,
        stream=stream,
    )


@dataclass(frozen=True, slots=True)
class AppContext:
    """The three objects fixed for the life of the app, shared by all routes."""

    settings: AppSettings
    channel: Any
    store: SessionStore

    def effective_tools(
        self,
        request_tools: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        if request_tools:
            return tuple(request_tools)
        return self.settings.enabled_tools

    def resolve_session(self, body: ChatBody) -> tuple[Session, JSONResponse | None]:
        """Common session logic: reuse or create, validate channel match."""
        if body.session_id:
            try:
                session = self.store.get(body.session_id)
            except SessionNotFoundError as exc:
                return None, error_response(404, exc.code, str(exc))
            if session.channel != self.channel.name:
                return None, error_response(
                    409,
                    ChannelMismatchError.code,
                    f"session belongs to channel `{session.channel}`, not `{self.channel.name}`",
                )
            return session, None
        return self.store.create(self.channel.name), None

    def build_request(self, body: ChatBody, session: Session) -> ChatRequest:
        return ChatRequest(
            prompt=body.prompt,
            session_id=session.id,
            native_session_id=session.native_session_id,
            model=body.model,
            workspace=body.workspace,
            permission_mode=body.permission_mode,
            enabled_tools=self.effective_tools(body.enabled_tools),
            images=tuple(body.images) if body.images else None,
            thinking_effort=body.thinking_effort,
        )

    def default_model(self) -> str:
        return self.channel.models[0] if self.channel.models else "default"

    # -- native passthrough ------------------------------------------------
    # When the active channel is a transparent cloud proxy (e.g. adal-cloud),
    # forward Anthropic (/v1/messages) and OpenAI (/v1/chat/completions)
    # requests verbatim to the proxy. This skips the normalized-event layer
    # entirely — CLIProxyAPI (or any Anthropic/OpenAI client) speaks its own
    # protocol end-to-end with zero loss (tools, usage, multi-turn all pass).
    def channel_supports_passthrough(self) -> bool:
        return all(
            hasattr(self.channel, m)
            for m in (
                "acquire_slot",
                "release_slot",
                "proxy_headers",
                "resolve_route",
                "resolve_target",
                "resolve_provider",
                "rewrite_body",
            )
        )

    def not_supported(self) -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content=oai.openai_error(
                f"channel `{self.channel.name}` has no native passthrough",
                err_type="api_error",
                code="channel_not_supported",
            ),
        )
