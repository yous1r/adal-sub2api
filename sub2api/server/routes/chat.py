"""Native sub2api chat surface: the normalized ChatRequest/Event protocol.

This is the only endpoint pair that speaks sub2api's own shape rather than an
OpenAI/Anthropic dialect, so it is also the only one that owns sessions
(``resolve_session``) instead of forwarding an opaque body upstream.
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ...core.aggregate import collect_answer
from ...core.types import SessionStarted, TurnCompleted
from ..auth import unauthorized
from ..deps import AppContext, ChatBody, error_response, sse_frame


def router(ctx: AppContext) -> APIRouter:
    api = APIRouter()
    channel = ctx.channel
    store = ctx.store

    @api.post("/v1/chat")
    async def chat(body: ChatBody, request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        session, failure = ctx.resolve_session(body)
        if failure is not None:
            return failure
        request = ctx.build_request(body, session)
        answer, terminal = await collect_answer(channel.chat(request))
        store.touch(session)
        if isinstance(terminal, TurnCompleted):
            store.adopt_native(session, terminal.session_id, terminal.model)
        elif terminal is None or terminal.type == "turn.failed":
            detail = (
                terminal.message
                if terminal
                else "stream ended without a terminal event"
            )
            code = terminal.code if terminal else "internal_error"
            return error_response(502, code, detail, session_id=session.id)
        return {
            "answer": answer,
            "session_id": session.id,
            "channel": channel.name,
            "model": getattr(terminal, "model", None) or body.model,
        }

    @api.post("/v1/chat/stream")
    async def chat_stream(body: ChatBody, request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        session, failure = ctx.resolve_session(body)
        if failure is not None:
            return failure
        request = ctx.build_request(body, session)

        async def stream() -> AsyncIterator[str]:
            yield sse_frame(SessionStarted(session_id=session.id, channel=channel.name))
            async for event in channel.chat(request):
                if isinstance(event, TurnCompleted):
                    store.adopt_native(session, event.session_id, event.model)
                yield sse_frame(event)
            store.touch(session)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return api
