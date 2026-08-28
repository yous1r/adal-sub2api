from __future__ import annotations

import httpx
import pytest

from sub2api.core.channel import ChannelConfig
from sub2api.core.config import AppSettings
from sub2api.server.app import create_app
from sub2api.server.openai import (
    build_chunk,
    build_completion,
    content_to_text,
    error_status_and_type,
    messages_to_prompt,
)


# --- pure mappings ----------------------------------------------------------


def test_single_user_message_maps_to_raw_prompt():
    assert messages_to_prompt([{"role": "user", "content": "hello"}]) == "hello"


def test_multi_role_history_becomes_tagged_transcript():
    prompt = messages_to_prompt(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert prompt == "[SYSTEM]\nbe terse\n\n[USER]\nhi\n\n[ASSISTANT]\nhello"


def test_multimodal_content_parts_flatten_to_text():
    content = [
        {"type": "text", "text": "look at"},
        {"type": "image_url", "image_url": {"url": "x.png"}},
        {"type": "text", "text": "this"},
    ]
    assert content_to_text(content) == "look at\nthis"


def test_none_content_maps_to_empty():
    assert content_to_text(None) == ""


def test_error_status_mapping():
    assert error_status_and_type("session_not_found") == (404, "invalid_request_error")
    assert error_status_and_type("upstream_failed") == (502, "api_error")
    assert error_status_and_type("mystery") == (502, "api_error")


def test_completion_shapes():
    completion = build_completion(id="chatcmpl-x", created=1, model="m", content="ans")
    assert completion["object"] == "chat.completion"
    assert completion["choices"][0]["message"] == {
        "role": "assistant",
        "content": "ans",
    }
    chunk = build_chunk(id="chatcmpl-x", created=1, model="m", delta={"content": "a"})
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["choices"][0]["delta"] == {"content": "a"}


# --- HTTP surface -----------------------------------------------------------


@pytest.mark.anyio
async def test_chat_completions_sync(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "echo-mini",
            "messages": [{"role": "user", "content": "openai hello"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert data["model"] == "echo-mini"
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "openai hello"
    assert choice["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_chat_completions_uses_full_history(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "question"},
            ]
        },
    )
    data = resp.json()
    # echo channel returns the flattened transcript verbatim
    assert (
        data["choices"][0]["message"]["content"] == "[SYSTEM]\nsys\n\n[USER]\nquestion"
    )


@pytest.mark.anyio
async def test_chat_completions_stream(client):
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "a b"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    chunks: list[dict] = []
    done_sentinel = False
    for line in resp.text.splitlines():
        if line == "data: [DONE]":
            done_sentinel = True
        elif line.startswith("data: "):
            import json

            chunks.append(json.loads(line[6:]))

    assert done_sentinel
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    contents = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks[1:]
        if c["choices"][0]["finish_reason"] is None
    )
    assert contents == "a b"
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


@pytest.mark.anyio
async def test_models_listing(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    ids = {m["id"] for m in data["data"]}
    assert "echo-mini" in ids
    for entry in data["data"]:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "echo"


# --- auth guard -------------------------------------------------------------


@pytest.fixture
async def secured_client():
    settings = AppSettings(
        channel="echo",
        host="testserver",
        port=0,
        api_key="sk-secret",
        channel_config=ChannelConfig(workspace="."),
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as c:
        yield c


@pytest.mark.anyio
async def test_missing_api_key_rejected(secured_client):
    resp = await secured_client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"

    resp = await secured_client.get("/v1/models")
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_wrong_api_key_rejected(secured_client):
    resp = await secured_client.get(
        "/v1/models", headers={"Authorization": "Bearer sk-wrong"}
    )
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_correct_api_key_accepted(secured_client):
    headers = {"Authorization": "Bearer sk-secret"}
    resp = await secured_client.get("/v1/models", headers=headers)
    assert resp.status_code == 200
    resp = await secured_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_healthz_stays_open_without_auth(secured_client):
    resp = await secured_client.get("/healthz")
    assert resp.status_code == 200


# --- /v1/responses passthrough -------------------------------------------


@pytest.mark.anyio
async def test_responses_passthrough_501_without_cloud_channel(client):
    """Echo channel has no passthrough → /v1/responses returns 501."""
    resp = await client.post(
        "/v1/responses",
        json={"model": "echo-mini", "input": "hi"},
    )
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == "channel_not_supported"


@pytest.mark.anyio
async def test_responses_get_501_without_cloud_channel(client):
    resp = await client.get("/v1/responses/resp_123")
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == "channel_not_supported"


@pytest.mark.anyio
async def test_responses_delete_501_without_cloud_channel(client):
    resp = await client.delete("/v1/responses/resp_123")
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == "channel_not_supported"


@pytest.mark.anyio
async def test_responses_v1v1_alias_501(client):
    """Compat alias /v1/v1/responses also returns 501 for echo channel."""
    resp = await client.post(
        "/v1/v1/responses",
        json={"model": "echo-mini", "input": "hi"},
    )
    assert resp.status_code == 501


@pytest.mark.anyio
async def test_responses_requires_api_key(secured_client):
    resp = await secured_client.post(
        "/v1/responses", json={"model": "echo-mini", "input": "hi"}
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


@pytest.mark.anyio
async def test_responses_get_requires_api_key(secured_client):
    resp = await secured_client.get("/v1/responses/resp_123")
    assert resp.status_code == 401
