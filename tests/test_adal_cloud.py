"""Tests for the adal-cloud channel: pure mapping functions + device-flow + SSE parsing.

Network calls are stubbed; the live proxy is exercised only through the
channel's end-to-end smoke (run manually with SUB2API_CHANNEL=adal-cloud).
"""

from __future__ import annotations

import json
import time
from pathlib import Path


import pytest

from sub2api.core.pool import AccountConfig, PoolConfig

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


@pytest.mark.anyio
async def test_channel_refresh_updates_catalog_and_pool(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    channel = mod.AdalCloudChannel(ChannelConfig())
    channel._started = True
    channel._pool = mod.AccountPool(
        PoolConfig(accounts=[AccountConfig(token="old", session_id="old-sid")])
    )
    await channel._pool.start()
    old_cfg = PoolConfig(accounts=[AccountConfig(token="old", session_id="old-sid")])
    new_cfg = PoolConfig(accounts=[AccountConfig(token="new", session_id="new-sid")])
    channel._pool_signature = channel._pool_signature_for(old_cfg)
    catalogs = iter(
        [
            {"models": [{"key": "model-a", "provider": "anthropic", "model_id": "a"}]},
        ]
    )
    monkeypatch.setattr(mod, "fetch_catalog", lambda *args: next(catalogs))
    monkeypatch.setattr(mod, "load_pool_config", lambda: new_cfg)

    await channel.refresh()

    assert channel.models == ("model-a",)
    assert channel._catalog["models"][0]["model_id"] == "a"
    assert channel._pool.slots[0].token == "new"
    assert channel._pool.slots[0].session_id == "new-sid"


@pytest.mark.anyio
async def test_channel_refresh_keeps_old_catalog_on_fetch_failure(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    channel = mod.AdalCloudChannel(ChannelConfig())
    channel._started = True
    channel._catalog = {"models": [{"key": "old", "provider": "anthropic"}]}
    channel.models = ("old",)
    monkeypatch.setattr(mod, "fetch_catalog", lambda *args: {})
    await channel.refresh()
    assert channel.models == ("old",)
    assert channel._catalog["models"][0]["key"] == "old"


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


@pytest.mark.anyio
async def test_health_reflects_reloaded_pool(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    channel = mod.AdalCloudChannel(ChannelConfig())
    channel._started = True
    initial = PoolConfig(accounts=[AccountConfig(token="old", session_id="old-sid")])
    channel._pool = mod.AccountPool(initial)
    await channel._pool.start()
    channel._pool_signature = channel._pool_signature_for(initial)
    updated = PoolConfig(
        accounts=[
            AccountConfig(token="new-a", session_id="new-a-sid"),
            AccountConfig(token="new-b", session_id="new-b-sid"),
        ]
    )
    monkeypatch.setattr(mod, "fetch_catalog", lambda *args: {})
    monkeypatch.setattr(mod, "load_pool_config", lambda: updated)

    health = await channel.health()

    assert health["models"] == list(channel.models)
    assert health["pool"]["enabled"] is True
    assert health["pool"]["size"] == 2
    assert [slot["session_id"] for slot in health["pool"]["slots"]] == [
        "new-a-sid",
        "new-b-sid",
    ]


@pytest.mark.anyio
async def test_health_marks_models_unavailable_when_all_accounts_cooldown(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    channel = mod.AdalCloudChannel(ChannelConfig())
    channel._started = True
    config = PoolConfig(
        max_failures=1,
        accounts=[AccountConfig(token="token", session_id="sid")],
    )
    channel._pool = mod.AccountPool(config)
    await channel._pool.start()
    channel._pool_signature = channel._pool_signature_for(config)
    channel._pool.slots[0].disabled_until = time.monotonic() + 60
    monkeypatch.setattr(mod, "fetch_catalog", lambda *args: {})
    monkeypatch.setattr(mod, "load_pool_config", lambda: config)

    health = await channel.health()

    assert health["pool"]["healthy_accounts"] == 0
    assert health["pool"]["available_capacity"] == 0
    assert health["pool"]["models_available"] is False


@pytest.mark.anyio
async def test_health_disables_pool_after_accounts_removed(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    channel = mod.AdalCloudChannel(ChannelConfig())
    channel._started = True
    config = PoolConfig(accounts=[AccountConfig(token="token", session_id="sid")])
    channel._pool = mod.AccountPool(config)
    await channel._pool.start()
    channel._pool_signature = channel._pool_signature_for(config)
    monkeypatch.setattr(mod, "fetch_catalog", lambda *args: {})
    monkeypatch.setattr(mod, "load_pool_config", lambda: None)

    health = await channel.health()

    assert health["pool"] == {"enabled": False}
    assert channel._pool is None


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
async def test_acquire_slot_single_account_registers_once_and_reuses(monkeypatch):
    ch = AdalCloudChannel(ChannelConfig())
    ch._token = "tok"
    ch._registered = set()
    ch._pool = None
    ch._proxy_sid = None
    calls = []

    async def fake_to_thread(func, **kw):
        calls.append(kw["session_id"])
        func(**kw)

    monkeypatch.setattr("sub2api.channels.adal_cloud.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr(
        "sub2api.channels.adal_cloud.register_session", lambda **kw: None
    )
    slot1, sid1 = await ch.acquire_slot()
    slot2, sid2 = await ch.acquire_slot()
    assert sid1 == sid2 and sid1.startswith("sub2api-")
    assert slot1 is None and slot2 is None  # single-account mode
    assert len(calls) == 1  # registered once, then cached


def test_channel_proxy_url_class_attr():
    ch = AdalCloudChannel(ChannelConfig())
    assert ch.proxy_url == ADAL_PROXY_URL


# -- pool integration --------------------------------------------------------


@pytest.mark.anyio
async def test_acquire_slot_pool_mode_distributes_across_accounts(monkeypatch):
    """When a pool is configured, acquire_slot round-robins across accounts."""
    from sub2api.core.pool import AccountConfig, AccountPool, PoolConfig

    ch = AdalCloudChannel(ChannelConfig())
    ch._token = "fallback-tok"
    ch._registered = set()
    ch._proxy_sid = None
    cfg = PoolConfig(
        strategy="round-robin",
        accounts=[
            AccountConfig(token="tok-a", session_id="sess-a", max_concurrent=5),
            AccountConfig(token="tok-b", session_id="sess-b", max_concurrent=5),
        ],
    )
    ch._pool = AccountPool(cfg)
    await ch._pool.start()

    monkeypatch.setattr(
        "sub2api.channels.adal_cloud.register_session", lambda **kw: None
    )

    results = []
    for _ in range(4):
        slot, sid = await ch.acquire_slot()
        results.append((slot.session_id, slot.token))
        await ch.release_slot(slot)
    # Round-robin: a, b, a, b
    assert results[0][0] == "sess-a"
    assert results[1][0] == "sess-b"
    assert results[2][0] == "sess-a"
    assert results[3][0] == "sess-b"
    await ch._pool.close()


def test_proxy_headers_uses_slot_token_in_pool_mode():
    """proxy_headers picks the slot's token when slot is provided."""
    from sub2api.core.pool import AccountSlot

    ch = AdalCloudChannel(ChannelConfig())
    ch._token = "fallback-tok"
    slot = AccountSlot(token="pool-tok", session_id="sess-x", max_concurrent=2)
    headers = ch.proxy_headers("sess-x", "https://api.openai.com", "openai", slot=slot)
    assert headers["Authorization"] == "Bearer pool-tok"
    assert headers["X-Session-ID"] == "sess-x"
    assert headers["X-Provider"] == "openai"


def test_proxy_headers_falls_back_to_channel_token():
    """Without a slot, proxy_headers uses the channel-level token."""
    ch = AdalCloudChannel(ChannelConfig())
    ch._token = "chan-tok"
    headers = ch.proxy_headers("sess-1", "https://api.openai.com", "", slot=None)
    assert headers["Authorization"] == "Bearer chan-tok"


# -- cookie mint / runtime token refresh ------------------------------------


def _make_jwt(claims: dict) -> str:
    import base64

    def b64(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")

    header = b64({"alg": "RS256", "typ": "JWT"})
    return f"{header}.{b64(claims)}.sig"


def _mint_resp(status_code: int = 200, payload: dict | None = None):
    class FakeResp:
        def __init__(self):
            self.status_code = status_code

        def json(self):
            return payload or {}

    return FakeResp()


def test_mint_token_with_cookies_success(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    seen = {}

    def fake_post(url, **kw):
        seen["url"] = url
        seen["cookies"] = kw.get("cookies")
        return _mint_resp(payload={"jwt": "fresh.jwt.sig"})

    monkeypatch.setattr(mod.httpx, "post", fake_post)
    token = _make_jwt({"exp": int(time.time()) - 5, "sid": "sess_abc"})
    fresh, err = mod.mint_token_with_cookies(
        token,
        [
            {
                "name": "__client",
                "value": "c-val",
                "domain": ".clerk.example",
                "path": "/",
            }
        ],
    )
    assert err is None
    assert fresh == "fresh.jwt.sig"
    assert "sessions/sess_abc/tokens" in seen["url"]


def test_mint_token_with_cookies_surfaces_clerk_error_code(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    monkeypatch.setattr(
        mod.httpx,
        "post",
        lambda url, **kw: _mint_resp(
            403, {"errors": [{"code": "user_banned", "message": "User banned"}]}
        ),
    )
    token = _make_jwt({"exp": int(time.time()) - 5, "sid": "sess_abc"})
    assert mod.mint_token_with_cookies(token, [{"name": "__client", "value": "c"}]) == (
        None,
        "user_banned",
    )


def test_mint_token_with_cookies_no_cookies_or_unparseable_token():
    import sub2api.channels.adal_cloud as mod

    assert mod.mint_token_with_cookies(_make_jwt({"sid": "s"}), []) == (
        None,
        "no_cookies",
    )
    assert mod.mint_token_with_cookies(
        "garbage", [{"name": "__client", "value": "c"}]
    ) == (None, "no_sid")


def test_refresh_token_with_cookies_falls_back_to_original(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    monkeypatch.setattr(
        mod, "mint_token_with_cookies", lambda *a, **k: (None, "user_banned")
    )
    assert mod.refresh_token_with_cookies("orig", [{"name": "__client", "value": "c"}]) == "orig"


@pytest.mark.anyio
async def test_refresh_slot_auth_marks_dead_on_ban(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    ch = mod.AdalCloudChannel(ChannelConfig())
    slot = mod.AccountSlot(
        token=_make_jwt({"exp": int(time.time()) - 10, "sid": "sess_x"}),
        session_id="s1",
        cookies=[{"name": "__client", "value": "c"}],
    )
    monkeypatch.setattr(
        mod, "mint_token_with_cookies", lambda *a, **k: (None, "user_banned")
    )
    assert await ch.refresh_slot_auth(slot, force=True) is False
    assert slot.dead_reason == "user_banned"
    assert slot.disabled_until > time.monotonic()


@pytest.mark.anyio
async def test_refresh_slot_auth_transient_failure_short_cooldown(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    ch = mod.AdalCloudChannel(ChannelConfig())
    slot = mod.AccountSlot(
        token=_make_jwt({"exp": int(time.time()) - 10, "sid": "sess_x"}),
        session_id="s1",
        cookies=[{"name": "__client", "value": "c"}],
    )
    monkeypatch.setattr(
        mod, "mint_token_with_cookies", lambda *a, **k: (None, "transient")
    )
    assert await ch.refresh_slot_auth(slot, force=True) is False
    # transient: parked briefly, but not permanently marked dead
    assert slot.dead_reason == ""
    assert 0 < slot.disabled_until - time.monotonic() <= 20


@pytest.mark.anyio
async def test_refresh_slot_auth_updates_token_on_success(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    ch = mod.AdalCloudChannel(ChannelConfig())
    slot = mod.AccountSlot(
        token=_make_jwt({"exp": int(time.time()) - 10, "sid": "sess_x"}),
        session_id="s1",
        cookies=[{"name": "__client", "value": "c"}],
    )
    monkeypatch.setattr(
        mod, "mint_token_with_cookies", lambda *a, **k: ("fresh.jwt.sig", None)
    )
    assert await ch.refresh_slot_auth(slot, force=True) is True
    assert slot.token == "fresh.jwt.sig"
    assert slot.dead_reason == ""
    assert slot.disabled_until == 0.0
    assert slot.fail_count == 0


@pytest.mark.anyio
async def test_refresh_slot_auth_fresh_token_short_circuits(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    ch = mod.AdalCloudChannel(ChannelConfig())
    slot = mod.AccountSlot(
        token=_make_jwt({"exp": int(time.time()) + 3600, "sid": "sess_x"}),
        session_id="s1",
        cookies=[{"name": "__client", "value": "c"}],
    )

    def boom(*a, **k):
        raise AssertionError("mint must not be called for a fresh token")

    monkeypatch.setattr(mod, "mint_token_with_cookies", boom)
    assert await ch.refresh_slot_auth(slot) is True


@pytest.mark.anyio
async def test_ensure_slot_token_skips_when_no_cookies(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    ch = mod.AdalCloudChannel(ChannelConfig())
    slot = mod.AccountSlot(
        token=_make_jwt({"exp": int(time.time()) - 10, "sid": "sess_x"}),
        session_id="s1",
        cookies=None,
    )

    def boom(*a, **k):
        raise AssertionError("mint must not be called without cookies")

    monkeypatch.setattr(mod, "mint_token_with_cookies", boom)
    await ch._ensure_slot_token(slot)


@pytest.mark.anyio
async def test_acquire_slot_pool_mode_refreshes_stale_token(monkeypatch):
    """acquire_slot lazily re-mints an expired JWT before handing out the slot."""
    import sub2api.channels.adal_cloud as mod

    ch = mod.AdalCloudChannel(ChannelConfig())
    ch._token = "fallback"
    ch._registered = {"sess-x"}  # skip registration
    cfg = PoolConfig(
        accounts=[
            AccountConfig(
                token=_make_jwt(
                    {"exp": int(time.time()) - 10, "sid": "sess_x"}
                ),
                session_id="sess-x",
                cookies=[{"name": "__client", "value": "c"}],
            )
        ]
    )
    ch._pool = mod.AccountPool(cfg)
    await ch._pool.start()
    monkeypatch.setattr(
        mod, "mint_token_with_cookies", lambda *a, **k: ("fresh.jwt.sig", None)
    )
    slot, sid = await ch.acquire_slot()
    assert slot is not None
    assert slot.token == "fresh.jwt.sig"
    await ch.release_slot(slot)
    await ch._pool.close()


@pytest.mark.anyio
async def test_refresh_slot_auth_single_account_rereads_creds(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    ch = mod.AdalCloudChannel(ChannelConfig())
    ch._token = "old-token"
    ch._client = None
    fresh = _make_jwt({"exp": int(time.time()) + 3600})
    monkeypatch.setattr(mod, "read_token", lambda: fresh)
    assert await ch.refresh_slot_auth(None) is True
    assert ch._token == fresh
    # second call: creds unchanged -> nothing to do
    assert await ch.refresh_slot_auth(None) is False


@pytest.mark.anyio
async def test_health_reports_dead_accounts(monkeypatch):
    import sub2api.channels.adal_cloud as mod

    channel = mod.AdalCloudChannel(ChannelConfig())
    channel._started = True
    config = PoolConfig(
        accounts=[
            AccountConfig(token="tok-a", session_id="sid-a"),
            AccountConfig(token="tok-b", session_id="sid-b"),
        ]
    )
    channel._pool = mod.AccountPool(config)
    await channel._pool.start()
    channel._pool_signature = channel._pool_signature_for(config)
    channel._pool.slots[0].dead_reason = "user_banned"
    monkeypatch.setattr(mod, "fetch_catalog", lambda *args: {})
    monkeypatch.setattr(mod, "load_pool_config", lambda: config)

    health = await channel.health()
    assert health["pool"]["dead_accounts"] == 1
    assert health["pool"]["slots"][0]["dead_reason"] == "user_banned"


# -- passthrough 401 retry ---------------------------------------------------


def _passthrough_channel(mod, handler):
    """Build a started adal-cloud channel wired to a mock transport."""
    import httpx as _httpx

    ch = mod.AdalCloudChannel(ChannelConfig())
    ch._started = True
    ch._catalog = SAMPLE_CATALOG
    cfg = PoolConfig(
        accounts=[
            AccountConfig(
                token=_make_jwt(
                    {"exp": int(time.time()) + 3600, "sid": "sess_x"}
                ),
                session_id="sess-x",
                cookies=[{"name": "__client", "value": "c"}],
            )
        ]
    )
    ch._pool = mod.AccountPool(cfg)
    ch._client = _httpx.AsyncClient(transport=_httpx.MockTransport(handler))
    ch._registered = {"sess-x"}
    return ch, cfg


@pytest.mark.anyio
async def test_messages_passthrough_retries_once_after_401(monkeypatch):
    """First proxy call 401s -> token re-mints -> retry succeeds end-to-end."""
    import httpx as _httpx

    import sub2api.channels.adal_cloud as mod
    import sub2api.server.app as app_mod

    calls = {"n": 0}

    def handler(request: _httpx.Request) -> _httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _httpx.Response(401, json={"error_type": "auth_invalid"})
        return _httpx.Response(
            200, json={"ok": True, "auth": request.headers.get("authorization", "")}
        )

    ch, cfg = _passthrough_channel(mod, handler)
    monkeypatch.setattr(app_mod, "create_channel", lambda name, cfg_: ch)
    monkeypatch.setattr(mod, "fetch_catalog", lambda *args: {})
    monkeypatch.setattr(mod, "load_pool_config", lambda: cfg)
    monkeypatch.setattr(
        mod, "mint_token_with_cookies", lambda *a, **k: ("fresh.jwt.sig", None)
    )

    from sub2api.core.config import AppSettings

    settings = AppSettings(
        channel="adal-cloud",
        host="testserver",
        port=0,
        channel_config=ChannelConfig(workspace="."),
    )
    app = app_mod.create_app(settings)
    transport = _httpx.ASGITransport(app=app)
    async with _httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        resp = await c.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    assert resp.json()["auth"] == "Bearer fresh.jwt.sig"
    assert calls["n"] == 2


@pytest.mark.anyio
async def test_messages_passthrough_no_retry_when_mint_fails(monkeypatch):
    """When the re-mint fails (e.g. banned), the 401 passes through untouched."""
    import httpx as _httpx

    import sub2api.channels.adal_cloud as mod
    import sub2api.server.app as app_mod

    calls = {"n": 0}

    def handler(request: _httpx.Request) -> _httpx.Response:
        calls["n"] += 1
        return _httpx.Response(401, json={"error_type": "auth_invalid"})

    ch, cfg = _passthrough_channel(mod, handler)
    monkeypatch.setattr(app_mod, "create_channel", lambda name, cfg_: ch)
    monkeypatch.setattr(mod, "fetch_catalog", lambda *args: {})
    monkeypatch.setattr(mod, "load_pool_config", lambda: cfg)
    monkeypatch.setattr(
        mod, "mint_token_with_cookies", lambda *a, **k: (None, "user_banned")
    )

    from sub2api.core.config import AppSettings

    settings = AppSettings(
        channel="adal-cloud",
        host="testserver",
        port=0,
        channel_config=ChannelConfig(workspace="."),
    )
    app = app_mod.create_app(settings)
    transport = _httpx.ASGITransport(app=app)
    async with _httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        resp = await c.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 401
    assert resp.json()["error_type"] == "auth_invalid"
    assert calls["n"] == 1
    # the dead account is parked so the scheduler stops routing to it
    assert ch._pool.slots[0].dead_reason == "user_banned"


@pytest.mark.anyio
async def test_messages_passthrough_stream_retries_after_401(monkeypatch):
    """Streaming passthrough also re-mints and retries once on a 401."""
    import httpx as _httpx

    import sub2api.channels.adal_cloud as mod
    import sub2api.server.app as app_mod

    calls = {"n": 0}

    def handler(request: _httpx.Request) -> _httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _httpx.Response(401, json={"error_type": "auth_invalid"})

        async def sse():
            yield b'data: {"ok": true}\n\n'

        return _httpx.Response(
            200,
            content=sse(),
            headers={"content-type": "text/event-stream"},
        )

    ch, cfg = _passthrough_channel(mod, handler)
    monkeypatch.setattr(app_mod, "create_channel", lambda name, cfg_: ch)
    monkeypatch.setattr(mod, "fetch_catalog", lambda *args: {})
    monkeypatch.setattr(mod, "load_pool_config", lambda: cfg)
    monkeypatch.setattr(
        mod, "mint_token_with_cookies", lambda *a, **k: ("fresh.jwt.sig", None)
    )

    from sub2api.core.config import AppSettings

    settings = AppSettings(
        channel="adal-cloud",
        host="testserver",
        port=0,
        channel_config=ChannelConfig(workspace="."),
    )
    app = app_mod.create_app(settings)
    transport = _httpx.ASGITransport(app=app)
    async with _httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        resp = await c.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    assert '{"ok": true}' in resp.text
    assert calls["n"] == 2
