"""Tests for the subscription-usage feature (GET /v1/usage).

The AdaL platform publishes live credits at two unauthenticated,
clerk-id-keyed endpoints; every network call here is served by an
``httpx.MockTransport`` so the suite stays offline.
"""

from __future__ import annotations

import base64
import json
import time

import httpx
import pytest

import sub2api.channels.adal_cloud as mod
import sub2api.server.app as app_mod
from sub2api.channels.adal_cloud import (
    ADAL_APP_URL,
    USAGE_UNIT,
    aggregate_quota,
    clerk_id_from_token,
    fetch_account_quota,
    fetch_tiers,
    tier_display_name,
)
from sub2api.core.channel import ChannelConfig
from sub2api.core.config import AppSettings
from sub2api.core.pool import AccountConfig, PoolConfig

USER_UUID = "822c5e02-06d6-4427-a216-9b9d2b134945"
CLERK_ID = "user_3IlZEUxw3JiDzwM2eBHrcTYCLPw"

TIERS = {
    "free": {"tier": "free", "display_name": "Free", "monthly_credits": 2.0},
    "pro": {"tier": "pro", "display_name": "Pro", "monthly_credits": 30.0},
}


def _jwt(claims: dict) -> str:
    def seg(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")

    return f"{seg({'alg': 'RS256'})}.{seg(claims)}.sig"


def _token(clerk_id: str = CLERK_ID, *, ttl: int = 3600) -> str:
    return _jwt({"sub": clerk_id, "sid": "sess_x", "exp": int(time.time()) + ttl})


def _platform_handler(
    *,
    users: dict[str, dict] | None = None,
    subs: dict[str, dict] | None = None,
    tiers: dict | None = TIERS,
    calls: dict[str, int] | None = None,
):
    """MockTransport handler for the platform's user/subscription/tier APIs."""
    users = (
        users if users is not None else {CLERK_ID: {"id": USER_UUID, "email": "a@b.c"}}
    )
    subs = (
        subs
        if subs is not None
        else {
            USER_UUID: {
                "tier": "pro",
                "status": "trialing",
                "billing_interval": "monthly",
                "monthly_credits": 80.0,
                "credits_used_this_period": 0.624921,
                "credits_remaining": 79.375079,
                "is_trialing": True,
                "current_period_end": "2026-09-09T09:24:29Z",
            }
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls[request.url.path] = calls.get(request.url.path, 0) + 1
        path = request.url.path
        if path == "/api/subscription/tiers":
            if tiers is None:
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={"tiers": tiers})
        if path.startswith("/api/user/by-clerk-id/"):
            user = users.get(path.rsplit("/", 1)[-1])
            if user is None:
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json=user)
        if path.startswith("/api/subscription/user/"):
            sub = subs.get(path.rsplit("/", 1)[-1])
            if sub is None:
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json=sub)
        return httpx.Response(404, json={"detail": "Not Found"})

    return handler


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# -- clerk id extraction -----------------------------------------------------


def test_clerk_id_from_token_reads_sub_claim():
    assert clerk_id_from_token(_token()) == CLERK_ID


def test_clerk_id_from_token_empty_for_garbage():
    assert clerk_id_from_token("not-a-jwt") == ""
    assert clerk_id_from_token("") == ""
    assert clerk_id_from_token(_jwt({"sid": "s"})) == ""


def test_clerk_id_survives_expired_token():
    """Quota lookups key off `sub`, so an expired JWT still resolves."""
    assert clerk_id_from_token(_token(ttl=-9000)) == CLERK_ID


# -- fetch_account_quota -----------------------------------------------------


@pytest.mark.anyio
async def test_fetch_account_quota_resolves_credits():
    async with _mock_client(_platform_handler()) as c:
        row = await fetch_account_quota(c, _token())
    assert row["ok"] is True
    assert row["email"] == "a@b.c"
    assert row["tier"] == "pro"
    assert row["status"] == "trialing"
    assert row["total"] == 80.0
    assert row["used"] == pytest.approx(0.624921)
    assert row["remaining"] == pytest.approx(79.375079)
    assert row["period_end"] == "2026-09-09T09:24:29Z"


@pytest.mark.anyio
async def test_fetch_account_quota_requires_token():
    async with _mock_client(_platform_handler()) as c:
        assert (await fetch_account_quota(c, ""))["error"] == "no auth token"
        row = await fetch_account_quota(c, "garbage")
    assert row == {"ok": False, "error": "token carries no clerk id"}


@pytest.mark.anyio
async def test_fetch_account_quota_reports_unknown_user():
    async with _mock_client(_platform_handler(users={})) as c:
        row = await fetch_account_quota(c, _token())
    assert row["ok"] is False
    assert "HTTP 404" in row["error"]


@pytest.mark.anyio
async def test_fetch_account_quota_reports_missing_subscription():
    async with _mock_client(_platform_handler(subs={})) as c:
        row = await fetch_account_quota(c, _token())
    assert row == {"ok": False, "error": "no subscription", "email": "a@b.c"}


@pytest.mark.anyio
async def test_fetch_account_quota_survives_transport_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    async with _mock_client(boom) as c:
        row = await fetch_account_quota(c, _token())
    assert row["ok"] is False
    assert "user lookup failed" in row["error"]


@pytest.mark.anyio
async def test_fetch_account_quota_targets_platform_endpoints():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return _platform_handler()(request)

    async with _mock_client(handler) as c:
        await fetch_account_quota(c, _token())
    assert seen == [
        f"{ADAL_APP_URL}/api/user/by-clerk-id/{CLERK_ID}",
        f"{ADAL_APP_URL}/api/subscription/user/{USER_UUID}",
    ]


# -- tiers -------------------------------------------------------------------


@pytest.mark.anyio
async def test_fetch_tiers_returns_catalog_and_tolerates_failure():
    async with _mock_client(_platform_handler()) as c:
        assert (await fetch_tiers(c))["pro"]["display_name"] == "Pro"
    async with _mock_client(_platform_handler(tiers=None)) as c:
        assert await fetch_tiers(c) == {}


def test_tier_display_name_falls_back_to_id():
    assert tier_display_name(TIERS, "pro") == "Pro"
    assert tier_display_name(TIERS, "max_plus") == "max_plus"
    assert tier_display_name({}, "pro") == "pro"


# -- aggregate_quota ---------------------------------------------------------


def _row(**over) -> dict:
    row = {
        "ok": True,
        "email": "a@b.c",
        "tier": "pro",
        "status": "active",
        "total": 30.0,
        "used": 4.0,
        "remaining": 26.0,
        "period_end": "2026-10-01T00:00:00Z",
    }
    row.update(over)
    return row


def test_aggregate_quota_single_account_flat_shape():
    payload = aggregate_quota([_row()], TIERS, channel="adal-cloud")
    assert payload["isValid"] is True
    assert "invalidMessage" not in payload
    assert payload["planName"] == "Pro"
    assert (payload["total"], payload["used"], payload["remaining"]) == (
        30.0,
        4.0,
        26.0,
    )
    assert payload["unit"] == USAGE_UNIT
    assert "active" in payload["extra"]
    assert "resets 2026-10-01T00:00:00Z" in payload["extra"]
    assert payload["sub2api"]["accounts_resolved"] == 1


def test_aggregate_quota_sums_pool_and_labels_duplicate_tiers():
    rows = [
        _row(),
        _row(remaining=26.5, used=3.5),
        _row(tier="free", total=2.0, used=0.5, remaining=1.5),
    ]
    payload = aggregate_quota(rows, TIERS, pool_enabled=True)
    assert payload["total"] == 62.0
    assert payload["used"] == 8.0
    assert payload["remaining"] == 54.0
    assert payload["planName"] == "Free, Pro x2"
    assert "3/3 accounts" in payload["extra"]


def test_aggregate_quota_reports_parked_accounts_but_keeps_credits():
    rows = [_row(), _row(dead_reason="user_banned")]
    payload = aggregate_quota(rows, TIERS, pool_enabled=True)
    assert payload["isValid"] is True
    assert payload["total"] == 60.0
    assert "1 parked (user_banned)" in payload["extra"]
    assert payload["sub2api"]["accounts_parked"] == 1


def test_aggregate_quota_invalid_when_every_account_parked():
    rows = [_row(dead_reason="user_banned"), _row(dead_reason="signed_out")]
    payload = aggregate_quota(rows, TIERS, pool_enabled=True)
    assert payload["isValid"] is False
    assert (
        payload["invalidMessage"] == "all 2 account(s) parked: signed_out, user_banned"
    )


def test_aggregate_quota_invalid_surfaces_lookup_errors():
    rows = [{"ok": False, "error": "no subscription"}]
    payload = aggregate_quota(rows, TIERS)
    assert payload["isValid"] is False
    assert payload["invalidMessage"] == "no subscription"
    assert payload["total"] == 0.0


def test_aggregate_quota_invalid_without_accounts():
    payload = aggregate_quota([], TIERS)
    assert payload["isValid"] is False
    assert payload["invalidMessage"] == "no adal account configured"
    assert payload["planName"] == ""


def test_aggregate_quota_tolerates_non_numeric_credits():
    payload = aggregate_quota([_row(total=None, used="x", remaining=[])], TIERS)
    assert (payload["total"], payload["used"], payload["remaining"]) == (0.0, 0.0, 0.0)


# -- channel.usage -----------------------------------------------------------


def _channel(handler, *, pool: PoolConfig | None = None) -> mod.AdalCloudChannel:
    ch = mod.AdalCloudChannel(ChannelConfig())
    ch._started = True
    ch._client = _mock_client(handler)
    if pool is not None:
        ch._pool = mod.AccountPool(pool)
    else:
        ch._token = _token()
    return ch


@pytest.mark.anyio
async def test_channel_usage_single_account():
    ch = _channel(_platform_handler())
    payload = await ch.usage()
    assert payload["isValid"] is True
    assert payload["remaining"] == pytest.approx(79.375079)
    assert payload["sub2api"]["pool_enabled"] is False
    await ch.close()


@pytest.mark.anyio
async def test_channel_usage_caches_until_refresh():
    calls: dict[str, int] = {}
    ch = _channel(_platform_handler(calls=calls))
    await ch.usage()
    hits = calls[f"/api/user/by-clerk-id/{CLERK_ID}"]
    await ch.usage()
    assert calls[f"/api/user/by-clerk-id/{CLERK_ID}"] == hits  # served from cache
    await ch.usage(refresh=True)
    assert calls[f"/api/user/by-clerk-id/{CLERK_ID}"] == hits + 1
    await ch.close()


@pytest.mark.anyio
async def test_channel_usage_pool_mode_marks_parked_slot():
    other = "user_second"
    users = {
        CLERK_ID: {"id": USER_UUID, "email": "a@b.c"},
        other: {"id": "uuid-2", "email": "b@b.c"},
    }
    subs = {
        USER_UUID: {
            "tier": "pro",
            "monthly_credits": 30.0,
            "credits_remaining": 26.0,
            "credits_used_this_period": 4.0,
            "status": "active",
        },
        "uuid-2": {
            "tier": "free",
            "monthly_credits": 2.0,
            "credits_remaining": 2.0,
            "credits_used_this_period": 0.0,
            "status": "active",
        },
    }
    cfg = PoolConfig(
        accounts=[
            AccountConfig(token=_token(), session_id="sid-a"),
            AccountConfig(token=_token(other), session_id="sid-b"),
        ]
    )
    ch = _channel(_platform_handler(users=users, subs=subs), pool=cfg)
    await ch._pool.start()
    ch._pool.slots[1].dead_reason = "user_banned"

    payload = await ch.usage()

    assert payload["isValid"] is True
    assert payload["total"] == 32.0
    assert payload["planName"] == "Free, Pro"
    assert payload["sub2api"]["accounts_parked"] == 1
    assert "1 parked (user_banned)" in payload["extra"]
    await ch._pool.close()
    await ch.close()


@pytest.mark.anyio
async def test_channel_usage_without_credentials_is_invalid():
    ch = mod.AdalCloudChannel(ChannelConfig())
    ch._started = True
    ch._client = _mock_client(_platform_handler())
    payload = await ch.usage()
    assert payload["isValid"] is False
    assert payload["invalidMessage"] == "no adal account configured"
    await ch.close()


# -- GET /v1/usage -----------------------------------------------------------


def _app(monkeypatch, ch, *, api_key: str | None = None):
    monkeypatch.setattr(app_mod, "create_channel", lambda name, cfg: ch)
    monkeypatch.setattr(mod, "fetch_catalog", lambda *a: {})
    monkeypatch.setattr(mod, "load_pool_config", lambda: None)
    settings = AppSettings(
        channel="adal-cloud",
        host="testserver",
        port=0,
        api_key=api_key,
        channel_config=ChannelConfig(workspace="."),
    )
    return app_mod.create_app(settings)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


@pytest.mark.anyio
async def test_usage_endpoint_returns_extractor_fields(monkeypatch):
    ch = _channel(_platform_handler())
    async with _client(_app(monkeypatch, ch)) as c:
        resp = await c.get("/v1/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {
        "isValid",
        "planName",
        "total",
        "used",
        "remaining",
        "unit",
        "extra",
    }
    assert body["isValid"] is True
    assert body["unit"] == "USD"
    assert isinstance(body["remaining"], (int, float))
    await ch.close()


@pytest.mark.anyio
async def test_usage_endpoint_aliases_match(monkeypatch):
    ch = _channel(_platform_handler())
    async with _client(_app(monkeypatch, ch)) as c:
        canonical = (await c.get("/v1/usage")).json()
        for alias in ("/usage", "/v1/v1/usage"):
            resp = await c.get(alias)
            assert resp.status_code == 200
            assert resp.json()["remaining"] == canonical["remaining"]
    await ch.close()


@pytest.mark.anyio
async def test_usage_endpoint_refresh_param_bypasses_cache(monkeypatch):
    calls: dict[str, int] = {}
    ch = _channel(_platform_handler(calls=calls))
    async with _client(_app(monkeypatch, ch)) as c:
        await c.get("/v1/usage")
        hits = calls[f"/api/user/by-clerk-id/{CLERK_ID}"]
        await c.get("/v1/usage")
        assert calls[f"/api/user/by-clerk-id/{CLERK_ID}"] == hits
        await c.get("/v1/usage?refresh=1")
    assert calls[f"/api/user/by-clerk-id/{CLERK_ID}"] == hits + 1
    await ch.close()


@pytest.mark.anyio
async def test_usage_endpoint_requires_api_key(monkeypatch):
    ch = _channel(_platform_handler())
    app = _app(monkeypatch, ch, api_key="sk-secret")
    async with _client(app) as c:
        assert (await c.get("/v1/usage")).status_code == 401
        ok = await c.get("/v1/usage", headers={"Authorization": "Bearer sk-secret"})
    assert ok.status_code == 200
    assert ok.json()["isValid"] is True
    await ch.close()


@pytest.mark.anyio
async def test_usage_endpoint_501_for_channel_without_subscription(client):
    """The default echo channel reports no subscription usage."""
    resp = await client.get("/v1/usage")
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == "channel_not_supported"
