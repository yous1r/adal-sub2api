"""Responses-only must not touch the Anthropic endpoints or the Responses path.

The switch re-homes OpenAI ingress (Chat Completions) onto the Responses
upstream.  Anthropic ingress keeps its native passthrough — a regression that
silently rerouted ``/v1/messages`` through ``/v1/responses`` would still answer
200, so the upstream path is asserted directly.
"""

from __future__ import annotations

import json

import httpx
import pytest

import sub2api.channels.adal_cloud as cloud
import sub2api.server.app as app_module
from sub2api.core.channel import ChannelConfig
from sub2api.core.config import AppSettings
from sub2api.core.pool import AccountConfig, AccountPool, PoolConfig


@pytest.fixture
async def responses_only_gateway(monkeypatch):
    clients = []
    pools = []

    def create(handler):
        config = PoolConfig(
            accounts=[AccountConfig(token="test-token", session_id="only-session")]
        )
        channel = cloud.AdalCloudChannel(ChannelConfig())
        channel._started = True
        channel._catalog = {
            "models": [
                {
                    "key": "anthropic-claude-x",
                    "model_id": "claude-x",
                    "provider": "anthropic",
                },
                {
                    "key": "openai-gpt-6-astra",
                    "model_id": "gpt-6-astra",
                    "provider": "openai",
                },
            ]
        }
        channel._pool = AccountPool(config)
        channel._pool_signature = channel._pool_signature_for(config)
        channel._registered = {"only-session"}
        channel._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        pools.append(channel._pool)
        clients.append(channel._client)
        monkeypatch.setattr(cloud, "load_pool_config", lambda: config)
        monkeypatch.setattr(app_module, "create_channel", lambda *args: channel)
        app = app_module.create_app(
            AppSettings(channel="adal-cloud", responses_only=True)
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
        )
        clients.append(client)
        return client

    yield create
    for client in clients:
        await client.aclose()
    for pool in pools:
        await pool.close()


def _anthropic_sse(text: str) -> bytes:
    return (
        "event: message_start\n"
        f"data: {json.dumps({'type': 'message_start', 'message': {}})}\n\n"
        "event: content_block_delta\n"
        f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
        "event: message_stop\n"
        f"data: {json.dumps({'type': 'message_stop'})}\n\n"
    ).encode()


@pytest.mark.anyio
async def test_messages_keeps_native_upstream_under_responses_only(
    responses_only_gateway,
):
    paths = []

    def upstream(request):
        paths.append(request.url.path)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=_anthropic_sse("native"),
        )

    client = responses_only_gateway(upstream)
    response = await client.post(
        "/v1/messages",
        json={
            "model": "claude-x",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert paths == ["/proxy/v1/messages"]
    assert '"text": "native"' in response.text


@pytest.mark.anyio
async def test_count_tokens_keeps_native_upstream_under_responses_only(
    responses_only_gateway,
):
    paths = []

    def upstream(request):
        paths.append(request.url.path)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=(
                "event: message_start\n"
                f"data: {json.dumps({'type': 'message_start', 'message': {'usage': {'input_tokens': 7}}})}\n\n"
                "event: message_stop\n"
                f"data: {json.dumps({'type': 'message_stop'})}\n\n"
            ).encode(),
        )

    client = responses_only_gateway(upstream)
    response = await client.post(
        "/v1/messages/count_tokens",
        json={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 7}
    assert paths == ["/proxy/v1/messages"]


@pytest.mark.anyio
async def test_responses_still_uses_the_responses_upstream(responses_only_gateway):
    paths = []

    def upstream(request):
        paths.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "model": "gpt-6-astra",
                "status": "completed",
                "output": [],
            },
        )

    client = responses_only_gateway(upstream)
    response = await client.post(
        "/v1/responses", json={"model": "gpt-6-astra", "input": "hi"}
    )

    assert response.status_code == 200
    assert paths == ["/proxy/v1/responses"]
