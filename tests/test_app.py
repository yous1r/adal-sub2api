from __future__ import annotations

import pytest

from tests.conftest import parse_sse


@pytest.mark.anyio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["channel"]["channel"] == "echo"
    assert data["channel"]["ready"] is True


@pytest.mark.anyio
async def test_channels_listing(client):
    resp = await client.get("/v1/channels")
    assert resp.status_code == 200
    data = resp.json()
    names = {c["name"] for c in data["channels"]}
    assert {"echo", "adal-cli", "adal-sdk"} <= names
    configured = [c for c in data["channels"] if c["configured"]]
    assert [c["name"] for c in configured] == ["echo"]
    assert data["active"]["channel"] == "echo"


@pytest.mark.anyio
async def test_chat_sync_echo(client):
    resp = await client.post("/v1/chat", json={"prompt": "hello world"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "hello world"
    assert data["channel"] == "echo"
    assert data["model"] == "echo-mini"
    assert data["session_id"]


@pytest.mark.anyio
async def test_chat_session_resume_roundtrip(client):
    first = (await client.post("/v1/chat", json={"prompt": "one"})).json()
    second = (
        await client.post("/v1/chat", json={"prompt": "two", "session_id": first["session_id"]})
    ).json()
    assert second["session_id"] == first["session_id"]

    info = (await client.get(f"/v1/sessions/{first['session_id']}")).json()
    assert info["turns"] == 2
    # echo adopts its own id as native; proves native-id plumbing works
    assert info["native_session_id"].startswith("echo-")


@pytest.mark.anyio
async def test_chat_unknown_session_404(client):
    resp = await client.post("/v1/chat", json={"prompt": "x", "session_id": "missing"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "session_not_found"


@pytest.mark.anyio
async def test_chat_rejects_bad_permission_mode(client):
    resp = await client.post("/v1/chat", json={"prompt": "x", "permission_mode": "chaos"})
    assert resp.status_code == 422  # pydantic Literal validation


@pytest.mark.anyio
async def test_chat_stream_emits_normalized_frames(client):
    resp = await client.post("/v1/chat/stream", json={"prompt": "a b"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = parse_sse(resp.text)
    types = [f["type"] for f in frames]
    assert types[0] == "session.started"
    assert "text.delta" in types
    assert "tool.started" in types and "tool.completed" in types
    assert types[-1] == "turn.completed"
    session_started = frames[0]
    completed = frames[-1]
    assert completed["session_id"].startswith("echo-")

    # stream session is resolvable afterwards
    info = (await client.get(f"/v1/sessions/{session_started['session_id']}")).json()
    assert info["turns"] == 1


@pytest.mark.anyio
async def test_stream_session_reuse_across_calls(client):
    first = parse_sse((await client.post("/v1/chat/stream", json={"prompt": "s1"})).text)
    sid = first[0]["session_id"]
    second = parse_sse(
        (await client.post("/v1/chat/stream", json={"prompt": "s2", "session_id": sid})).text
    )
    assert second[0]["session_id"] == sid


@pytest.mark.anyio
async def test_empty_prompt_rejected(client):
    resp = await client.post("/v1/chat", json={"prompt": ""})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_app_state_exposed(app):
    assert app.state.channel.name == "echo"
    assert app.state.store is not None
