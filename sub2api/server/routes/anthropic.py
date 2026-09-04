"""Anthropic Messages surface: passthrough, token counting, and the /v1/v1 alias.

Non-stream requests deliberately do not use a non-stream upstream POST — see
:func:`messages_passthrough` — which is why this module leans on
``forward_collect`` where the OpenAI modules use ``forward_to_proxy``.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...compat.affinity import affinity_key
from ...compat.errors import anthropic_error
from ..auth import unauthorized
from ..deps import AppContext, make_meter, model_not_found, parse_body
from ..proxy import forward_collect, forward_to_proxy


def router(ctx: AppContext) -> APIRouter:
    api = APIRouter()
    channel = ctx.channel

    @api.post("/v1/messages")
    async def messages_passthrough(request: Request):
        """Anthropic Messages API passthrough to the cloud proxy.

        Non-streaming requests are served by streaming upstream and
        aggregating server-side (:func:`forward_collect`): the proxy answers a
        non-stream body above ``max_tokens: 21333`` with a hard HTTP 500 and
        reports request-shape rejections as opaque 500s, while its streaming
        surface handles both.  The client still sees one JSON envelope.
        """
        denial = unauthorized(request)
        if denial is not None:
            return denial
        if not ctx.channel_supports_passthrough():
            return ctx.not_supported()
        raw = await request.body()
        body, failure = parse_body(raw)
        if failure is not None:
            return failure
        route = channel.resolve_route("/v1/messages", body)
        if route is None:
            return model_not_found(str(body.get("model") or ""), "anthropic")
        stream = bool(body.get("stream"))
        url = f"{channel.proxy_url}/proxy{route.proxy_path}"
        fwd_body = channel.rewrite_body(raw, route.proxy_path)
        slot, sid = await channel.acquire_slot(affinity_key(body, route.protocol))
        meter = make_meter(request, route, sid, "/v1/messages", stream)
        if stream:
            fwd_gen = await forward_to_proxy(
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
            )
            return StreamingResponse(fwd_gen, media_type="text/event-stream")
        status, payload = await forward_collect(
            channel=channel,
            url=url,
            fwd_body=fwd_body,
            slot=slot,
            sid=sid,
            target=route.target_url,
            provider=route.provider,
            protocol=route.protocol,
            meter=meter,
        )
        return JSONResponse(status_code=status, content=payload)

    @api.post("/v1/messages/count_tokens")
    async def messages_count_tokens(request: Request):
        """Anthropic token counting, implemented as a 1-token probe.

        The cloud proxy rejects this path outright (``Unsupported Bedrock
        proxy path``), so the count comes from a real ``max_tokens: 1``
        streaming call whose ``message_start.usage.input_tokens`` is exactly
        what a full call would bill.  ``tools``/``system`` are preserved so
        the number matches the real request, and the sampled output is one
        token (~2e-5 credits).
        """
        denial = unauthorized(request)
        if denial is not None:
            return denial
        if not ctx.channel_supports_passthrough():
            return ctx.not_supported()
        raw = await request.body()
        body, failure = parse_body(raw)
        if failure is not None:
            return failure
        route = channel.resolve_route("/v1/messages", body)
        if route is None:
            return model_not_found(str(body.get("model") or ""), "anthropic")
        probe = dict(body)
        probe["max_tokens"] = 1
        probe["stream"] = True
        fwd_body = channel.rewrite_body(json.dumps(probe).encode(), route.proxy_path)
        url = f"{channel.proxy_url}/proxy{route.proxy_path}"
        slot, sid = await channel.acquire_slot(affinity_key(body, route.protocol))
        meter = make_meter(request, route, sid, "/v1/messages/count_tokens", False)
        status, payload = await forward_collect(
            channel=channel,
            url=url,
            fwd_body=fwd_body,
            slot=slot,
            sid=sid,
            target=route.target_url,
            provider=route.provider,
            protocol=route.protocol,
            meter=meter,
        )
        if status != 200:
            return JSONResponse(status_code=status, content=payload)
        usage = payload.get("usage") if isinstance(payload, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        counted = usage.get("input_tokens")
        if counted is None:
            return JSONResponse(
                status_code=502,
                content=anthropic_error(
                    "upstream reported no input token count", "api_error"
                ),
            )
        return {"input_tokens": int(counted)}

    @api.post("/v1/v1/messages")
    async def messages_passthrough_v1v1(request: Request):
        """Compat alias for clients whose base-url includes /v1 twice."""
        return await messages_passthrough(request)

    return api
