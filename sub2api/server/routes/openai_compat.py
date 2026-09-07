"""OpenAI Chat Completions surface, plus its two base-url compat aliases.

Two very different paths share one handler: a verbatim passthrough when the
channel is a transparent cloud proxy, and the normalized ChatRequest pipeline
otherwise.  ``completion_stream`` lives here because only the normalized
branch of this endpoint emits OpenAI chunk frames.
"""

from __future__ import annotations

import json
import httpx
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...compat.affinity import affinity_key
from ...core.aggregate import collect_answer
from ...core.types import ChatRequest, TurnCompleted, TurnFailed
from .. import openai as oai
from ..auth import unauthorized
from ..deps import (
    AppContext,
    CompletionBody,
    make_meter,
    model_not_found,
    parse_body,
    passthrough_json,
    wants_usage_chunk,
)
from ..proxy import forward_to_proxy, usage_from_payload


def router(ctx: AppContext) -> APIRouter:
    api = APIRouter()
    channel = ctx.channel
    settings = ctx.settings

    async def completion_stream(
        chat_request: ChatRequest, created: int
    ) -> AsyncIterator[str]:
        cid = oai.completion_id()
        model = chat_request.model or ctx.default_model()

        def frame(delta: dict[str, Any], finish_reason: str | None = None) -> str:
            payload = oai.build_chunk(
                id=cid,
                created=created,
                model=model,
                delta=delta,
                finish_reason=finish_reason,
            )
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield frame({"role": "assistant"})
        async for event in channel.chat(chat_request):
            if event.type == "text.delta":
                yield frame({"content": event.text})
            elif event.type == "message.completed":
                yield frame({"content": event.text})
            elif event.type == "thought.delta":
                # DeepSeek-style extension; ignored by clients that don't use it
                yield frame({"reasoning_content": event.text})
            elif isinstance(event, TurnCompleted):
                yield frame({}, finish_reason="stop")
                break
            elif isinstance(event, TurnFailed):
                _, err_type = oai.error_status_and_type(event.code)
                payload = oai.build_chunk(
                    id=cid, created=created, model=model, delta={}, finish_reason="stop"
                )
                payload["error"] = {
                    "message": event.message,
                    "type": err_type,
                    "code": event.code,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break
        yield "data: [DONE]\n\n"

    @api.post("/v1/chat/completions")
    async def chat_completions(body: CompletionBody, request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        # Fast path: when the channel is a transparent cloud proxy, forward the
        # client's OpenAI request verbatim — tools, usage, and the full
        # multi-turn messages array all pass through unchanged.
        if ctx.channel_supports_passthrough():
            raw = await request.body()
            body_dict, failure = parse_body(raw)
            if failure is not None:
                return failure
            route = channel.resolve_route("/v1/chat/completions", body_dict)
            if route is None:
                return model_not_found(str(body_dict.get("model") or ""), "openai")
            url = f"{channel.proxy_url}/proxy{route.proxy_path}"
            fwd_body = channel.rewrite_body(raw, route.proxy_path)
            slot, sid = await channel.acquire_slot(
                affinity_key(body_dict, route.protocol)
            )
            meter = make_meter(
                request, route, sid, "/v1/chat/completions", bool(body.stream)
            )
            if body.stream:
                fwd_result = await forward_to_proxy(
                    channel=channel,
                    url=url,
                    fwd_body=fwd_body,
                    slot=slot,
                    sid=sid,
                    target=route.target_url,
                    provider=route.provider,
                    stream=True,
                    protocol=route.protocol,
                    meter=meter,
                    strip_usage_chunk=not wants_usage_chunk(body_dict),
                )
                if isinstance(fwd_result, httpx.Response):
                    return JSONResponse(
                        status_code=fwd_result.status_code,
                        content=passthrough_json(fwd_result),
                    )
                return StreamingResponse(fwd_result, media_type="text/event-stream")
            # Non-streaming: use a dedicated request with a short read timeout
            # so upstream errors (403/502) return fast instead of hanging.
            resp = await forward_to_proxy(
                channel=channel,
                url=url,
                fwd_body=fwd_body,
                slot=slot,
                sid=sid,
                target=route.target_url,
                provider=route.provider,
                stream=False,
                protocol=route.protocol,
            )
            payload = passthrough_json(resp)
            meter.record(
                status=resp.status_code,
                usage=usage_from_payload(route.protocol, payload),
            )
            return JSONResponse(status_code=resp.status_code, content=payload)
        chat_request = ChatRequest(
            prompt=oai.messages_to_prompt([m.model_dump() for m in body.messages]),
            session_id=None,
            model=body.model,
            permission_mode=settings.openai_permission_mode,
            thinking_effort=body.thinking_effort or body.reasoning_effort,
        )
        created = oai.now_epoch()
        if body.stream:
            return StreamingResponse(
                completion_stream(chat_request, created), media_type="text/event-stream"
            )
        answer, terminal = await collect_answer(channel.chat(chat_request))
        if terminal is None or isinstance(terminal, TurnFailed):
            code = terminal.code if terminal else "internal_error"
            message = (
                terminal.message
                if terminal
                else "stream ended without a terminal event"
            )
            status, err_type = oai.error_status_and_type(code)
            return JSONResponse(
                status_code=status,
                content=oai.openai_error(message, err_type=err_type, code=code),
            )
        return oai.build_completion(
            id=oai.completion_id(),
            created=created,
            model=getattr(terminal, "model", None)
            or chat_request.model
            or ctx.default_model(),
            content=answer,
        )

    @api.post("/v1/v1/chat/completions")
    async def chat_completions_v1v1(body: CompletionBody, request: Request):
        """Compat alias for clients whose base-url includes /v1 twice."""
        return await chat_completions(body, request)

    @api.post("/v1/completions")
    async def completions_compat(body: CompletionBody, request: Request):
        """Compat alias for legacy /v1/completions (maps to chat/completions)."""
        return await chat_completions(body, request)

    return api
