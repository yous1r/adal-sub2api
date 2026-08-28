"""Tests for the adal-cloud channel: pure mapping functions + device-flow + SSE parsing.

Network calls are stubbed; the live proxy is exercised only through the
channel's end-to-end smoke (run manually with SUB2API_CHANNEL=adal-cloud).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from sub2api.channels.adal_cloud import (
    ADAL_APP_URL,
    ADAL_PROXY_URL,
    AdalCloudChannel,
    PROVIDER_BASE_URLS,
    anthropic_request,
    catalog_models,
    device_flow_login,
    fetch_catalog,
    openai_request,
    parse_anthropic_sse,
    parse_openai_sse,
    provider_for_model,
    read_token,
    register_session,
    target_for_request,
    token_needs_refresh,
    upstream_model_id,
)
from sub2api.core.channel import ChannelConfig
from sub2api.core.errors import AuthError, UpstreamError
from sub2api.core.types import ChatRequest, TextDelta, ThoughtDelta


# -- catalog / provider mapping --------------------------------------------


SAMPLE_CATALOG = {
    "default_model": "anthropic-claude-sonnet-5",
    "models": [
        {
            "key": "anthropic-claude-sonnet-5",
            "model_id": "claude-sonnet-5",
            "provider": "anthropic",
            "model_client": "AnthropicAPIClient",
        },
        {
            "key": "openai-gpt-5.6-terra",
            "model_id": "gpt-5.6-terra",
            "provider": "openai",
            "model_client": "OpenAIClient",
        },
        {
            "key": "zai-glm-5.2",
            "model_id": "glm-5.2",
            "provider": "zai",
            "model_client": "ZAIAPIClient",
        },
    ],
}


def test_catalog_models_extracts_keys_in_order():
    assert catalog_models(SAMPLE_CATALOG) == (
        "anthropic-claude-sonnet-5",
        "openai-gpt-5.6-terra",
        "zai-glm-5.2",
    )


def test_catalog_models_tolerates_missing_or_garbage():
    assert catalog_models({}) == ()
    assert catalog_models({"models": "bogus"}) == ()
    assert catalog_models(None) == ()
    assert catalog_models({"models": [{"no_key": 1}, {"key": "ok"}]}) == ("ok",)


def test_provider_for_model_key_and_id():
    assert (
        provider_for_model(SAMPLE_CATALOG, "anthropic-claude-sonnet-5") == "anthropic"
    )
    assert provider_for_model(SAMPLE_CATALOG, "claude-sonnet-5") == "anthropic"
    assert provider_for_model(SAMPLE_CATALOG, "openai-gpt-5.6-terra") == "openai"
    assert provider_for_model(SAMPLE_CATALOG, "unknown-model") is None
    assert provider_for_model({}, "x") is None


def test_upstream_model_id_resolves_key_to_id():
    assert (
        upstream_model_id(SAMPLE_CATALOG, "anthropic-claude-sonnet-5")
        == "claude-sonnet-5"
    )
    assert upstream_model_id(SAMPLE_CATALOG, "claude-sonnet-5") == "claude-sonnet-5"
    # unknown model passes through unchanged
    assert upstream_model_id(SAMPLE_CATALOG, "mystery") == "mystery"
    assert upstream_model_id({}, "mystery") == "mystery"


# -- request builders ------------------------------------------------------


def test_anthropic_request_basic():
    req = ChatRequest(prompt="hi")
    body = anthropic_request("claude-sonnet-5", req)
    assert body == {
        "model": "claude-sonnet-5",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }


def test_anthropic_request_adds_thinking_when_effort_set():
    req = ChatRequest(prompt="think", thinking_effort="high")
    body = anthropic_request("claude-sonnet-5", req)
    assert body["thinking"] == {"type": "adaptive"}
    assert body["stream"] is True


def test_openai_request_basic():
    req = ChatRequest(prompt="hi")
    body = openai_request("gpt-5.6-terra", req)
    assert body == {
        "model": "gpt-5.6-terra",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }


def test_openai_request_passes_reasoning_effort():
    req = ChatRequest(prompt="hi", thinking_effort="max")
    body = openai_request("gpt-5.6-terra", req)
    assert body["reasoning_effort"] == "max"


# -- SSE parsing ----------------------------------------------------------


def test_parse_anthropic_text_delta():
    line = 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"PONG"}}'
    events = parse_anthropic_sse(line)
    assert len(events) == 1
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "PONG"


def test_parse_anthropic_thinking_delta():
    line = 'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"reasoning"}}'
    events = parse_anthropic_sse(line)
    assert len(events) == 1
    assert isinstance(events[0], ThoughtDelta)
    assert events[0].text == "reasoning"


def test_parse_anthropic_ignores_non_data_and_noise():
    assert parse_anthropic_sse("") == []
    assert parse_anthropic_sse("event: message_start") == []
    assert parse_anthropic_sse("data: not json") == []
    assert parse_anthropic_sse('data: {"type":"message_start"}') == []
    assert parse_anthropic_sse('data: {"type":"message_stop"}') == []


def test_parse_anthropic_error_raises_upstream():
    line = 'data: {"type":"error","error":{"message":"quota exhausted"}}'
    with pytest.raises(UpstreamError, match="quota exhausted"):
        parse_anthropic_sse(line)


def test_parse_openai_text_delta():
    line = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
    events = parse_openai_sse(line)
    assert len(events) == 1
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "hi"


def test_parse_openai_reasoning_content_becomes_thought():
    line = 'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}'
    events = parse_openai_sse(line)
    assert len(events) == 1
    assert isinstance(events[0], ThoughtDelta)
    assert events[0].text == "thinking"


def test_parse_openai_ignores_non_data_and_noise():
    assert parse_openai_sse("") == []
    assert parse_openai_sse(": comment") == []
    assert parse_openai_sse("data: [DONE]") == []
    assert parse_openai_sse("data: not json") == []
    assert parse_openai_sse('data: {"choices":[]}') == []


def test_parse_openai_error_raises_upstream():
    line = 'data: {"error":{"message":"rate limited"}}'
    with pytest.raises(UpstreamError, match="rate limited"):
        parse_openai_sse(line)


# -- token logic ---------------------------------------------------------


def test_read_token_returns_token(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({"access_token": "tok-123", "expiry_date": 0}))
    assert read_token(p) == "tok-123"


def test_read_token_missing_or_garbage(tmp_path):
    assert read_token(tmp_path / "absent.json") is None
    p = tmp_path / "bad.json"
    p.write_text("not json")
    assert read_token(p) is None
    p2 = tmp_path / "empty.json"
    p2.write_text(json.dumps({"access_token": ""}))
    assert read_token(p2) is None


def test_token_needs_refresh_for_missing():
    assert token_needs_refresh(None) is True
    assert token_needs_refresh("") is True


def test_token_needs_refresh_for_expired_and_valid():
    # craft a JWT with exp 100s in the future
    import base64

    def b64(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")

    header = b64({"alg": "RS256", "typ": "JWT"})
    now = int(time.time())
    valid = f"{header}.{b64({'exp': now + 3600})}.sig"
    expiring = f"{header}.{b64({'exp': now + 10})}.sig"
    expired = f"{header}.{b64({'exp': now - 10})}.sig"
    assert token_needs_refresh(valid) is False
    assert token_needs_refresh(expiring) is True
    assert token_needs_refresh(expired) is True


def test_token_needs_refresh_unparseable_is_trusted():
    assert token_needs_refresh("not-a-jwt") is False


# -- fetch_catalog / register_session (network stubs) --------------------


def test_fetch_catalog_stubs_network(monkeypatch):
    class FakeResp:
        def __init__(self, body):
            self._body = body.encode()

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        "sub2api.channels.adal_cloud.urlopen",
        lambda req, timeout=20: FakeResp(json.dumps(SAMPLE_CATALOG)),
    )
    cat = fetch_catalog()
    assert cat["default_model"] == "anthropic-claude-sonnet-5"
    assert catalog_models(cat)[0] == "anthropic-claude-sonnet-5"


def test_fetch_catalog_returns_empty_on_network_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("no net")

    monkeypatch.setattr("sub2api.channels.adal_cloud.urlopen", boom)
    assert fetch_catalog() == {}


def test_register_session_posts_and_succeeds(monkeypatch):
    calls = []

    class FakeResp:
        def read(self):
            return b'{"id":"abc"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=20):
        calls.append(req)
        return FakeResp()

    monkeypatch.setattr("sub2api.channels.adal_cloud.urlopen", fake_urlopen)
    register_session(token="tok", session_id="s1")
    assert len(calls) == 1
    assert calls[0].full_url == f"{ADAL_APP_URL}/api/client-sessions/start"
    assert calls[0].get_header("Authorization") == "Bearer tok"
    body = json.loads(calls[0].data)
    assert body["session_id"] == "s1"
    assert body["client_entrypoint"] == "adal"


def test_register_session_raises_auth_on_http_error(monkeypatch):
    from urllib.error import HTTPError
    from io import BytesIO

    def fake_urlopen(req, timeout=20):
        raise HTTPError(
            req.full_url, 401, "Unauthorized", {}, BytesIO(b'{"error":"x"}')
        )

    monkeypatch.setattr("sub2api.channels.adal_cloud.urlopen", fake_urlopen)
    with pytest.raises(AuthError, match="HTTP 401"):
        register_session(token="tok", session_id="s1")


# -- device flow --------------------------------------------------------


def test_device_flow_login_completes_on_authorized(monkeypatch, tmp_path):
    init = {
        "device_code": "dc-1",
        "user_code": "ABCD-EFGH",
        "verification_url": "https://adal.sylph.ai/verify",
        "expires_in": 600,
    }
    pending_seen = []

    def fake_initiate(app_url=ADAL_APP_URL, timeout=15.0):
        return init

    def fake_poll(device_code, app_url=ADAL_APP_URL, timeout=15.0):
        return {"status": "authorized", "token": "fresh-tok"}

    monkeypatch.setattr(
        "sub2api.channels.adal_cloud.initiate_device_flow", fake_initiate
    )
    monkeypatch.setattr("sub2api.channels.adal_cloud.poll_device_flow", fake_poll)
    monkeypatch.setattr("sub2api.channels.adal_cloud.time.sleep", lambda s: None)
    creds = tmp_path / "creds.json"
    tok = device_flow_login(
        on_pending=lambda r: pending_seen.append(r), creds_path=creds
    )
    assert tok == "fresh-tok"
    assert pending_seen == [init]
    saved = json.loads(creds.read_text())
    assert saved["access_token"] == "fresh-tok"


def test_device_flow_login_raises_on_expired(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sub2api.channels.adal_cloud.initiate_device_flow",
        lambda *a, **k: {
            "device_code": "dc",
            "user_code": "x",
            "verification_url": "u",
            "expires_in": 1,
        },
    )
    monkeypatch.setattr(
        "sub2api.channels.adal_cloud.poll_device_flow",
        lambda *a, **k: {"status": "expired", "token": None},
    )
    with pytest.raises(AuthError, match="expired"):
        device_flow_login(creds_path=tmp_path / "c.json", max_attempts=1)


# -- channel lifecycle (no network) ------------------------------------


def test_adal_cloud_channel_class_attrs():
    assert AdalCloudChannel.name == "adal-cloud"
    assert "anthropic-claude-sonnet-5" in AdalCloudChannel.models
    ch = AdalCloudChannel(ChannelConfig())
    assert ch.runtime_available() is True


def test_route_for_defaults_to_anthropic():
    ch = AdalCloudChannel(ChannelConfig())
    ch._catalog = SAMPLE_CATALOG
    sub, target, model = ch._route_for(
        ChatRequest(prompt="hi", model="anthropic-claude-sonnet-5")
    )
    assert sub == "/v1/messages"
    assert target == "https://api.anthropic.com"
    assert model == "claude-sonnet-5"


def test_route_for_openai():
    ch = AdalCloudChannel(ChannelConfig())
    ch._catalog = SAMPLE_CATALOG
    sub, target, model = ch._route_for(
        ChatRequest(prompt="hi", model="openai-gpt-5.6-terra")
    )
    assert sub == "/v1/chat/completions"
    assert target == "https://api.openai.com"
    assert model == "gpt-5.6-terra"


def test_route_for_unknown_model_falls_back_to_anthropic():
    ch = AdalCloudChannel(ChannelConfig())
    ch._catalog = SAMPLE_CATALOG
    sub, target, model = ch._route_for(
        ChatRequest(prompt="hi", model="totally-unknown")
    )
    # unknown provider -> default anthropic routing; model passes through unchanged
    assert sub == "/v1/messages"
    assert target == "https://api.anthropic.com"


def test_require_token_raises_when_unresolved():
    ch = AdalCloudChannel(ChannelConfig())
    ch._token = None
    with pytest.raises(Exception, match="no auth token"):
        ch._require_token()


@pytest.mark.anyio
async def test_ensure_registered_dedupes_and_calls_register(monkeypatch):
    ch = AdalCloudChannel(ChannelConfig())
    ch._token = "tok"
    ch._registered = set()
    calls = []

    async def fake_to_thread(func, **kw):
        # simulate asyncio.to_thread: call sync func
        calls.append(kw)
        func(**kw)

    monkeypatch.setattr("sub2api.channels.adal_cloud.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr(
        "sub2api.channels.adal_cloud.register_session", lambda **kw: None
    )
    await ch._ensure_registered("s1")
    await ch._ensure_registered("s1")  # already registered: no second call
    assert len(calls) == 1
    assert calls[0]["session_id"] == "s1"


# -- passthrough routing ------------------------------------------------


def test_target_for_request_messages_always_anthropic():
    # /v1/messages always targets Anthropic, regardless of model
    assert (
        target_for_request("/v1/messages", {"model": "anything"}, SAMPLE_CATALOG)
        == PROVIDER_BASE_URLS["anthropic"]
    )


def test_target_for_request_chat_completions_by_provider():
    assert (
        target_for_request(
            "/v1/chat/completions", {"model": "openai-gpt-5.6-terra"}, SAMPLE_CATALOG
        )
        == PROVIDER_BASE_URLS["openai"]
    )
    assert (
        target_for_request(
            "/v1/chat/completions", {"model": "zai-glm-5.2"}, SAMPLE_CATALOG
        )
        == PROVIDER_BASE_URLS["zai"]
    )
    assert (
        target_for_request(
            "/v1/chat/completions", {"model": "gpt-5.6-terra"}, SAMPLE_CATALOG
        )
        == PROVIDER_BASE_URLS["openai"]
    )


def test_target_for_request_chat_completions_defaults_to_openai():
    # unknown model, no catalog -> openai default
    assert (
        target_for_request("/v1/chat/completions", {"model": "mystery"}, {})
        == PROVIDER_BASE_URLS["openai"]
    )


def test_target_for_request_responses_routes_like_chat_completions():
    # /v1/responses uses the same provider routing as /v1/chat/completions
    assert (
        target_for_request(
            "/v1/responses", {"model": "openai-gpt-5.6-terra"}, SAMPLE_CATALOG
        )
        == PROVIDER_BASE_URLS["openai"]
    )
    assert (
        target_for_request("/v1/responses", {"model": "zai-glm-5.2"}, SAMPLE_CATALOG)
        == PROVIDER_BASE_URLS["zai"]
    )
    assert (
        target_for_request("/v1/responses", {"model": "gpt-5.6-terra"}, SAMPLE_CATALOG)
        == PROVIDER_BASE_URLS["openai"]
    )


def test_target_for_request_responses_defaults_to_openai():
    # unknown model, no catalog -> openai default (same as chat/completions)
    assert (
        target_for_request("/v1/responses", {"model": "mystery"}, {})
        == PROVIDER_BASE_URLS["openai"]
    )


def test_resolve_target_delegates_responses():
    ch = AdalCloudChannel(ChannelConfig())
    ch._catalog = SAMPLE_CATALOG
    assert (
        ch.resolve_target("/v1/responses", {"model": "openai-gpt-5.6-terra"})
        == "https://api.openai.com"
    )


def test_channel_proxy_headers_include_session_and_target():
    ch = AdalCloudChannel(ChannelConfig())
    ch._token = "tok-123"
    h = ch.proxy_headers("sid-1", "https://api.anthropic.com")
    assert h["Authorization"] == "Bearer tok-123"
    assert h["X-Session-ID"] == "sid-1"
    assert h["X-Target-URL"] == "https://api.anthropic.com"
    assert h["Content-Type"] == "application/json"


def test_channel_resolve_target_delegates_to_helper():
    ch = AdalCloudChannel(ChannelConfig())
    ch._catalog = SAMPLE_CATALOG
    assert (
        ch.resolve_target("/v1/messages", {"model": "claude-sonnet-5"})
        == "https://api.anthropic.com"
    )
    assert (
        ch.resolve_target("/v1/chat/completions", {"model": "openai-gpt-5.6-terra"})
        == "https://api.openai.com"
    )


@pytest.mark.anyio
async def test_proxy_session_id_registers_once_and_reuses(monkeypatch):
    ch = AdalCloudChannel(ChannelConfig())
    ch._token = "tok"
    ch._registered = set()
    calls = []

    async def fake_to_thread(func, **kw):
        calls.append(kw["session_id"])
        func(**kw)

    monkeypatch.setattr("sub2api.channels.adal_cloud.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr(
        "sub2api.channels.adal_cloud.register_session", lambda **kw: None
    )
    sid1 = await ch.proxy_session_id()
    sid2 = await ch.proxy_session_id()
    assert sid1 == sid2 and sid1.startswith("sub2api-")
    assert len(calls) == 1  # registered once, then cached


def test_channel_proxy_url_class_attr():
    ch = AdalCloudChannel(ChannelConfig())
    assert ch.proxy_url == ADAL_PROXY_URL
