"""OpenAI Responses surface: create, retrieve, delete, and the /v1/v1 alias.

The GET/DELETE handlers talk to ``channel._client`` directly rather than
through :mod:`sub2api.server.proxy`, whose forwarding helpers are POST-shaped;
their 501 bodies are spelled inline for the same historical reason.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...compat.affinity import affinity_key
from .. import openai as oai
from ..auth import unauthorized
from ..deps import (
    AppContext,
    make_meter,
    model_not_found,
    parse_body,
    passthrough_json,
)
from ..proxy import forward_to_proxy, usage_from_payload


def router(ctx: AppContext) -> APIRouter:
    api = APIRouter()
    channel = ctx.channel

    @api.post("/v1/responses")
    async def responses_passthrough(request: Request):
        """OpenAI Responses API passthrough to the cloud proxy.

        The Responses API is OpenAI's modern, stateful endpoint for reasoning
        models and agentic workloads.  When the active channel is a transparent
        cloud proxy (e.g. adal-cloud), forward the client's request verbatim to
        the proxy's ``/proxy/v1/responses`` path — ``input``, ``tools``,
        ``reasoning``, ``background`` mode, streaming events
        (``response.created`` … ``response.completed``) all pass through
        unchanged.  The proxy keeps the ``/v1/...`` suffix from the inbound
        path and sets ``X-Target-URL`` to the upstream OpenAI host.

        Non-stream stays a direct POST: unlike Anthropic's, this upstream
        surface answers non-stream bodies cleanly at every measured size.
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
        route = channel.resolve_route("/v1/responses", body)
        if route is None:
            return model_not_found(str(body.get("model") or ""), "openai")
        stream = bool(body.get("stream"))
        url = f"{channel.proxy_url}/proxy{route.proxy_path}"
        fwd_body = channel.rewrite_body(raw, route.proxy_path)
        slot, sid = await channel.acquire_slot(affinity_key(body, route.protocol))
        meter = make_meter(request, route, sid, "/v1/responses", stream)
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

    @api.get("/v1/responses/{response_id}")
    async def responses_get(response_id: str, request: Request):
        """Retrieve a previously created Responses API object."""
        denial = unauthorized(request)
        if denial is not None:
            return denial
        if not ctx.channel_supports_passthrough():
            return JSONResponse(
                status_code=501,
                content=oai.openai_error(
                    f"channel `{channel.name}` has no native passthrough",
                    err_type="api_error",
                    code="channel_not_supported",
                ),
            )
        target = channel.resolve_target("/v1/responses", {})
        slot, sid = await channel.acquire_slot()
        headers = channel.proxy_headers(sid, target, "", slot=slot)
        url = f"{channel.proxy_url}/proxy/v1/responses/{response_id}"
        client = channel._client
        assert client is not None
        try:
            resp = await client.get(
                url,
                headers=headers,
                timeout=httpx.Timeout(60.0, connect=15.0, read=30.0),
            )
        except Exception:
            await channel.release_slot(slot, success=False)
            raise
        await channel.release_slot(slot, success=resp.status_code < 500)
        return JSONResponse(
            status_code=resp.status_code, content=passthrough_json(resp)
        )

    @api.delete("/v1/responses/{response_id}")
    async def responses_delete(response_id: str, request: Request):
        """Delete a stored Responses API object."""
        denial = unauthorized(request)
        if denial is not None:
            return denial
        if not ctx.channel_supports_passthrough():
            return JSONResponse(
                status_code=501,
                content=oai.openai_error(
                    f"channel `{channel.name}` has no native passthrough",
                    err_type="api_error",
                    code="channel_not_supported",
                ),
            )
        target = channel.resolve_target("/v1/responses", {})
        slot, sid = await channel.acquire_slot()
        headers = channel.proxy_headers(sid, target, "", slot=slot)
        url = f"{channel.proxy_url}/proxy/v1/responses/{response_id}"
        client = channel._client
        assert client is not None
        try:
            resp = await client.delete(
                url,
                headers=headers,
                timeout=httpx.Timeout(60.0, connect=15.0, read=30.0),
            )
        except Exception:
            await channel.release_slot(slot, success=False)
            raise
        await channel.release_slot(slot, success=resp.status_code < 500)
        return JSONResponse(
            status_code=resp.status_code, content=passthrough_json(resp)
        )

    @api.post("/v1/v1/responses")
    async def responses_passthrough_v1v1(request: Request):
        """Compat alias for clients whose base-url includes /v1 twice."""
        return await responses_passthrough(request)

    return api
