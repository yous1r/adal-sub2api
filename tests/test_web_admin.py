"""Tests for the ``--web`` admin surface and the hardened API-key guard.

Everything here runs over ASGI with no network: the one endpoint that would
call out (``/admin/api/accounts/{sid}/quota``) is served through an
``httpx.MockTransport`` returning the ``/api/credits/balance`` payload
recorded from the live platform.

The admin router hands out AdaL bearer tokens and Clerk cookies, so the
central assertion is that every JSON route refuses an unkeyed caller and that
listings never render a full secret.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

import sub2api.server.app as app_mod
from sub2api.channels.adal_cloud import clerk_id_from_token
from sub2api.core.channel import ChannelConfig
from sub2api.core.config import AppSettings
from sub2api.server.routes.web import extract_credentials, fingerprint, redact

API_KEY = "sk-admin-secret"

# Recorded verbatim from GET https://adal.sylph.ai/api/credits/balance.
BALANCE_PAYLOAD = {
    "total": 80.0,
    "monthly_allocation": 80.0,
    "monthly_used": 0.624921,
    "monthly_usage_percentage": 0.78,
    "weekly_usage": {
        "spent": 19.461245,
        "limit": 20.0,
        "remaining": 0.538755,
        "usage_percentage": 97.31,
        "resets_at": "2026-09-09T09:24:29Z",
        "is_usage_limited": False,
        "limit_reason": "",
    },
}

ACCOUNTS_JSON = {
    "strategy": "least-connections",
    "max_failures": 5,
    "cooldown_seconds": 30.0,
    "accounts": [
        {
            "token": "header.payload.signature-one",
            "session_id": "sess_alpha",
            "max_concurrent": 3,
            "cookies": [{"name": "__client", "value": "cookie-alpha"}],
            "email": "alpha@example.com",
        },
        {
            "token": "header.payload.signature-two",
            "session_id": "sess_beta",
            "max_concurrent": 2,
            "cookies": [{"name": "__client", "value": "cookie-beta"}],
            "email": "beta@example.com",
        },
    ],
}


class _StubChannel:
    """Minimal channel: enough surface for the admin router and lifespan."""

    name = "stub"
    display_name = "Stub"
    models = ()

    def __init__(self) -> None:
        self.refreshed = 0
        # Single-account mode pins the creds-file account's bearer on the
        # shared client.  Keeping that here is what makes the cross-account
        # rule in ``_balance`` bite: an import of a *different* account must
        # still resolve its own email.
        self._client = httpx.AsyncClient(
            transport=httpx.MockTransport(_balance),
            headers={"Authorization": f"Bearer {_jwt('user_PREEXISTING')}"},
        )

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        await self._client.aclose()

    async def refresh(self) -> None:
        self.refreshed += 1

    async def health(self) -> dict:
        return {"channel": self.name, "ready": True, "pool": {"enabled": False}}


def _jwt(sub: str) -> str:
    """A JWT-shaped token whose ``sub`` claim is ``sub``; signature is inert.

    The signature is padded to a realistic RS256 length because the paste
    importer locates tokens by shape, and a 3-character stub would not be a
    faithful stand-in for a Clerk token.
    """

    def seg(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    claims = {"sub": sub, "sid": f"sess_{sub}", "iat": 1790933258}
    return f"{seg({'alg': 'RS256'})}.{seg(claims)}.{'s1gnatur3' * 38}"


# Recorded shapes: GET /api/user/by-clerk-id/{clerk_id} and
# GET /api/subscription/user/{uuid}, plus the bearer-authenticated credits
# endpoint.  The user lookup reproduces one measured access rule: it answers
# 403 ``Cannot query other users`` when the bearer names a different account,
# so a caller must authenticate as the account it is asking about.
def _balance(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    bearer = request.headers.get("authorization", "")
    if path == "/api/credits/balance":
        if not bearer.startswith("Bearer "):
            return httpx.Response(401, json={"error_type": "auth_required"})
        return httpx.Response(200, json=BALANCE_PAYLOAD)
    if path.startswith("/api/user/by-clerk-id/"):
        clerk_id = path.rsplit("/", 1)[-1]
        subject = (
            clerk_id_from_token(bearer.removeprefix("Bearer "))
            if bearer.startswith("Bearer ")
            else ""
        )
        if subject and subject != clerk_id:
            return httpx.Response(
                403,
                json={
                    "error_type": "auth_forbidden",
                    "message": "Cannot query other users",
                },
            )
        return httpx.Response(
            200,
            json={
                "id": f"uuid-{clerk_id}",
                "email": f"{clerk_id}@example.com",
                "clerk_id": clerk_id,
            },
        )
    if path.startswith("/api/subscription/user/"):
        return httpx.Response(200, json={"tier": "free", "status": "active"})
    return httpx.Response(404, json={"error": "unexpected path"})


@pytest.fixture
def admin_app(monkeypatch, tmp_path):
    channel = _StubChannel()
    monkeypatch.setattr(app_mod, "create_channel", lambda name, cfg: channel)
    settings = AppSettings(
        channel="stub",
        host="testserver",
        port=0,
        api_key=API_KEY,
        web=True,
        db=str(tmp_path / "admin.sqlite3"),
        channel_config=ChannelConfig(workspace="."),
    )
    return app_mod.create_app(settings)


@pytest.fixture
async def admin_client(admin_app):
    # ASGITransport does not run lifespan, so the store is opened the same way
    # the lifespan does, keeping app.state identical to a live server's.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=admin_app), base_url="http://testserver"
    ) as client:
        from sub2api.core.store import Store

        store = Store(admin_app.state.settings.db)
        admin_app.state.usage_store = store
        try:
            yield client
        finally:
            await store.close()
            await admin_app.state.channel.close()


KEYED = {"x-api-key": API_KEY}


# -- redaction ---------------------------------------------------------------


def test_fingerprint_reveals_only_length_and_last_four():
    assert fingerprint("header.payload.signature") == "24 chars …ture"


def test_fingerprint_short_secret_hides_everything():
    assert fingerprint("abcd") == "4 chars"


def test_fingerprint_empty_is_empty():
    assert fingerprint("") == ""
    assert fingerprint(None) == ""


def test_fingerprint_cookies_reports_count_only():
    assert fingerprint([{"name": "__client", "value": "secret"}]) == "1 cookie(s)"


def test_redact_replaces_token_and_cookies_only():
    row = {
        "session_id": "s1",
        "token": "aaaa.bbbb.cccc",
        "cookies": [{"name": "__client", "value": "v"}],
        "email": "a@b.c",
    }
    safe = redact(row)
    assert safe["session_id"] == "s1"
    assert safe["email"] == "a@b.c"
    assert "aaaa" not in safe["token"]
    assert safe["cookies"] == "1 cookie(s)"
    # The source row is untouched: redaction must not destroy live state.
    assert row["token"] == "aaaa.bbbb.cccc"


# -- auth --------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/admin/api/accounts"),
        ("POST", "/admin/api/accounts"),
        ("POST", "/admin/api/accounts/paste"),
        ("PATCH", "/admin/api/accounts/sess_alpha"),
        ("DELETE", "/admin/api/accounts/sess_alpha"),
        ("POST", "/admin/api/accounts/import"),
        ("GET", "/admin/api/accounts/export"),
        ("GET", "/admin/api/accounts/sess_alpha/quota"),
        ("POST", "/admin/api/device/start"),
        ("POST", "/admin/api/device/claim"),
        ("POST", "/admin/api/reload"),
    ],
)
async def test_every_admin_api_route_requires_the_key(admin_client, method, path):
    resp = await admin_client.request(method, path, json={})
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


@pytest.mark.anyio
async def test_admin_accepts_bearer_as_well_as_x_api_key(admin_client):
    resp = await admin_client.get(
        "/admin/api/accounts", headers={"Authorization": f"Bearer {API_KEY}"}
    )
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_admin_rejects_wrong_key(admin_client):
    resp = await admin_client.get("/admin/api/accounts", headers={"x-api-key": "nope"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_admin_page_is_self_contained_html(admin_client):
    resp = await admin_client.get("/admin")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert "<!doctype html>" in body
    # No CDN, no template engine, no build step.
    assert "http://" not in body.replace("http://127.0.0.1", "")
    assert "src=" not in body


# -- account CRUD ------------------------------------------------------------


@pytest.mark.anyio
async def test_accounts_listing_redacts_secrets(admin_client):
    await admin_client.post(
        "/admin/api/accounts/import", headers=KEYED, json=ACCOUNTS_JSON
    )
    resp = await admin_client.get("/admin/api/accounts", headers=KEYED)
    assert resp.status_code == 200
    rows = resp.json()["accounts"]
    assert [r["session_id"] for r in rows] == ["sess_alpha", "sess_beta"]
    for row, token in zip(rows, ("...signature-one", "...signature-two")):
        # Only the length and the last four characters survive; the JWT body
        # that would authenticate as the account never leaves the process.
        assert row["token"] == f"28 chars …{token[-4:]}"
        assert "payload" not in row["token"]
        assert row["cookies"] == "1 cookie(s)"


@pytest.mark.anyio
async def test_import_list_export_round_trip(admin_client):
    imported = await admin_client.post(
        "/admin/api/accounts/import", headers=KEYED, json=ACCOUNTS_JSON
    )
    assert imported.json() == {"imported": 2, "skipped": 0}

    exported = await admin_client.get("/admin/api/accounts/export", headers=KEYED)
    assert exported.status_code == 200
    assert "attachment" in exported.headers["content-disposition"]
    payload = json.loads(exported.text)
    assert payload["strategy"] == "least-connections"
    assert payload["max_failures"] == 5
    assert payload["cooldown_seconds"] == 30.0
    assert payload["accounts"] == ACCOUNTS_JSON["accounts"]


@pytest.mark.anyio
async def test_import_skips_entries_without_credentials(admin_client):
    payload = {"accounts": [{"session_id": "no-token"}, {"token": "no-sid"}]}
    resp = await admin_client.post(
        "/admin/api/accounts/import", headers=KEYED, json=payload
    )
    assert resp.json() == {"imported": 0, "skipped": 2}


@pytest.mark.anyio
async def test_import_rejects_non_object_payload(admin_client):
    resp = await admin_client.post(
        "/admin/api/accounts/import", headers=KEYED, json=[1, 2, 3]
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"


@pytest.mark.anyio
async def test_upsert_then_patch_then_delete(admin_client):
    created = await admin_client.post(
        "/admin/api/accounts",
        headers=KEYED,
        json={
            "session_id": "sess_new",
            "token": "a.b.c",
            "email": "new@example.com",
            "max_concurrent": 7,
        },
    )
    assert created.json() == {"session_id": "sess_new", "ok": True}

    patched = await admin_client.patch(
        "/admin/api/accounts/sess_new", headers=KEYED, json={"max_concurrent": 2}
    )
    assert patched.status_code == 200
    rows = (await admin_client.get("/admin/api/accounts", headers=KEYED)).json()
    row = next(r for r in rows["accounts"] if r["session_id"] == "sess_new")
    assert row["max_concurrent"] == 2
    # A patch that omits the token must not wipe it.
    exported = json.loads(
        (await admin_client.get("/admin/api/accounts/export", headers=KEYED)).text
    )
    assert (
        next(a for a in exported["accounts"] if a["session_id"] == "sess_new")["token"]
        == "a.b.c"
    )

    deleted = await admin_client.delete("/admin/api/accounts/sess_new", headers=KEYED)
    assert deleted.json() == {"session_id": "sess_new", "deleted": True}
    rows = (await admin_client.get("/admin/api/accounts", headers=KEYED)).json()
    assert all(r["session_id"] != "sess_new" for r in rows["accounts"])


@pytest.mark.anyio
async def test_upsert_requires_session_id_and_token(admin_client):
    resp = await admin_client.post(
        "/admin/api/accounts", headers=KEYED, json={"session_id": "only-sid"}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"


@pytest.mark.anyio
async def test_patch_and_delete_unknown_account_are_404(admin_client):
    patched = await admin_client.patch(
        "/admin/api/accounts/ghost", headers=KEYED, json={"max_concurrent": 1}
    )
    assert patched.status_code == 404
    deleted = await admin_client.delete("/admin/api/accounts/ghost", headers=KEYED)
    assert deleted.status_code == 404


# -- quota + reload ----------------------------------------------------------


@pytest.mark.anyio
async def test_quota_returns_live_balance_payload(admin_client):
    await admin_client.post(
        "/admin/api/accounts/import", headers=KEYED, json=ACCOUNTS_JSON
    )
    resp = await admin_client.get("/admin/api/accounts/sess_alpha/quota", headers=KEYED)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["weekly_usage"]["remaining"] == 0.538755
    assert payload["weekly_usage"]["is_usage_limited"] is False
    assert payload["monthly_allocation"] == 80.0


@pytest.mark.anyio
async def test_quota_unknown_account_is_404(admin_client):
    resp = await admin_client.get("/admin/api/accounts/ghost/quota", headers=KEYED)
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_reload_refreshes_the_channel(admin_client):
    channel = admin_client._transport.app.state.channel  # type: ignore[attr-defined]
    before = channel.refreshed
    resp = await admin_client.post("/admin/api/reload", headers=KEYED)
    assert resp.status_code == 200
    assert resp.json()["reloaded"] is True
    assert channel.refreshed == before + 1


# -- one-field import: paste + device code -----------------------------------


TOKEN_A = _jwt("user_AAA")


@pytest.fixture
def adal_stub(monkeypatch):
    """Stub the network half of the AdaL auth module on the package object.

    The routes resolve ``register_session``/``initiate_device_flow``/
    ``poll_device_flow`` through the package module at call time, so rebinding
    attributes here is what a live import would hit.
    """
    import sub2api.channels.adal_cloud as adal

    calls: dict[str, list] = {"register": [], "poll": []}

    def register_session(*, token, session_id, **kw):
        calls["register"].append((token, session_id))

    def initiate_device_flow(*a, **kw):
        return {
            "device_code": "dev-code-secret",
            "user_code": "ABC-DEF-9",
            "expires_in": 600,
            "verification_url": "https://adal.sylph.ai/auth/device-verify",
        }

    def poll_device_flow(device_code, *a, **kw):
        calls["poll"].append(device_code)
        return calls.get("next_poll") or {"status": "pending", "token": None}

    monkeypatch.setattr(adal, "register_session", register_session)
    monkeypatch.setattr(adal, "initiate_device_flow", initiate_device_flow)
    monkeypatch.setattr(adal, "poll_device_flow", poll_device_flow)
    return calls


def test_extract_credentials_finds_a_bare_jwt():
    token, cookies = extract_credentials(f"  {TOKEN_A}  ")
    assert token == TOKEN_A
    assert cookies == []


def test_extract_credentials_reads_the_creds_file_shape():
    blob = json.dumps({"access_token": TOKEN_A, "expiry_date": 1790933318778})
    assert extract_credentials(blob)[0] == TOKEN_A


def test_extract_credentials_finds_a_token_inside_noise():
    noise = f"curl -H 'Authorization: Bearer {TOKEN_A}' https://api.adal.sylph.ai/"
    assert extract_credentials(noise)[0] == TOKEN_A


def test_extract_credentials_harvests_nested_cookies():
    blob = json.dumps(
        {
            "accounts": [
                {
                    "token": TOKEN_A,
                    "cookies": [{"name": "__client", "value": "c1"}],
                }
            ]
        }
    )
    token, cookies = extract_credentials(blob)
    assert token == TOKEN_A
    assert cookies == [{"name": "__client", "value": "c1"}]


def test_extract_credentials_rejects_a_jwt_without_a_subject():
    # Shape alone is not enough: a token that cannot identify an account is
    # useless to the pool, so it must not be imported as one.
    raw = base64.urlsafe_b64encode(b'{"iss":"https://clerk.adal.sylph.ai"}')
    body = raw.decode().rstrip("=")
    assert extract_credentials(f"eyJhbGciOiJSUzI1NiJ9.{body}.{'s1gnatur3' * 38}") == (
        "",
        [],
    )


def test_extract_credentials_ignores_text_without_a_token():
    assert extract_credentials("no credentials here") == ("", [])
    assert extract_credentials("") == ("", [])


@pytest.mark.anyio
async def test_paste_import_needs_only_the_token(admin_client, adal_stub):
    resp = await admin_client.post(
        "/admin/api/accounts/paste", headers=KEYED, json={"text": TOKEN_A}
    )
    assert resp.status_code == 200
    body = resp.json()
    # The operator supplied no session_id, email, cookies or max_concurrent.
    assert body["session_id"].startswith("sub2api-pool-")
    assert body["email"] == "user_AAA@example.com"
    assert body["registered"] is True
    assert body["updated"] is False
    assert adal_stub["register"] == [(TOKEN_A, body["session_id"])]

    rows = (await admin_client.get("/admin/api/accounts", headers=KEYED)).json()
    assert [r["session_id"] for r in rows["accounts"]] == [body["session_id"]]
    row = rows["accounts"][0]
    assert row["status"] == "alive"
    assert row["max_concurrent"] == 4
    # No cookies: a device/paste row cannot be re-minted through Clerk.
    assert row["cookies"] == "0 cookie(s)"


@pytest.mark.anyio
async def test_paste_import_reimport_updates_one_row(admin_client, adal_stub):
    first = (
        await admin_client.post(
            "/admin/api/accounts/paste", headers=KEYED, json={"text": TOKEN_A}
        )
    ).json()
    await admin_client.patch(
        f"/admin/api/accounts/{first['session_id']}",
        headers=KEYED,
        json={"max_concurrent": 9},
    )
    # A second, fresher token for the same Clerk subject.
    again = await admin_client.post(
        "/admin/api/accounts/paste", headers=KEYED, json={"text": _jwt("user_AAA")}
    )
    assert again.json()["session_id"] == first["session_id"]
    assert again.json()["updated"] is True
    rows = (await admin_client.get("/admin/api/accounts", headers=KEYED)).json()
    assert len(rows["accounts"]) == 1
    # A re-import must not silently reset a hand-tuned concurrency limit.
    assert rows["accounts"][0]["max_concurrent"] == 9


@pytest.mark.anyio
async def test_paste_import_keeps_the_token_when_registration_fails(
    admin_client, adal_stub, monkeypatch
):
    import sub2api.channels.adal_cloud as adal
    from sub2api.core.errors import AuthError

    def boom(**kw):
        raise AuthError("session registration failed: HTTP 403")

    monkeypatch.setattr(adal, "register_session", boom)
    resp = await admin_client.post(
        "/admin/api/accounts/paste", headers=KEYED, json={"text": TOKEN_A}
    )
    body = resp.json()
    assert body["registered"] is False
    assert "403" in body["detail"]
    exported = json.loads(
        (await admin_client.get("/admin/api/accounts/export", headers=KEYED)).text
    )
    assert exported["accounts"][0]["token"] == TOKEN_A
    rows = (await admin_client.get("/admin/api/accounts", headers=KEYED)).json()
    assert rows["accounts"][0]["status"] == "unknown"


@pytest.mark.anyio
async def test_paste_import_rejects_text_without_a_token(admin_client, adal_stub):
    resp = await admin_client.post(
        "/admin/api/accounts/paste", headers=KEYED, json={"text": "hello"}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "no_credentials"
    rows = (await admin_client.get("/admin/api/accounts", headers=KEYED)).json()
    assert rows["accounts"] == []


@pytest.mark.anyio
async def test_device_start_hides_the_device_code(admin_client, adal_stub):
    resp = await admin_client.post("/admin/api/device/start", headers=KEYED, json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_code"] == "ABC-DEF-9"
    assert body["expires_in"] == 600
    assert body["verification_url"].endswith("/auth/device-verify")
    # The device code grants a bearer: it must never reach the browser.
    assert "dev-code-secret" not in resp.text
    assert body["flow_id"] != "dev-code-secret"


@pytest.mark.anyio
async def test_device_claim_pending_writes_nothing(admin_client, adal_stub):
    started = (
        await admin_client.post("/admin/api/device/start", headers=KEYED, json={})
    ).json()
    resp = await admin_client.post(
        "/admin/api/device/claim", headers=KEYED, json={"flow_id": started["flow_id"]}
    )
    assert resp.json() == {"status": "pending"}
    assert adal_stub["poll"] == ["dev-code-secret"]
    rows = (await admin_client.get("/admin/api/accounts", headers=KEYED)).json()
    assert rows["accounts"] == []


@pytest.mark.anyio
async def test_device_claim_authorized_imports_the_account(admin_client, adal_stub):
    started = (
        await admin_client.post("/admin/api/device/start", headers=KEYED, json={})
    ).json()
    adal_stub["next_poll"] = {"status": "authorized", "token": TOKEN_A}
    resp = await admin_client.post(
        "/admin/api/device/claim", headers=KEYED, json={"flow_id": started["flow_id"]}
    )
    body = resp.json()
    assert body["status"] == "ok"
    assert body["session_id"].startswith("sub2api-pool-")
    assert body["email"] == "user_AAA@example.com"
    assert body["pool_reloaded"] is True
    rows = (await admin_client.get("/admin/api/accounts", headers=KEYED)).json()
    assert len(rows["accounts"]) == 1
    assert rows["accounts"][0]["session_id"] == body["session_id"]

    # The flow is single-use: claiming it twice must not mint a second row.
    replay = await admin_client.post(
        "/admin/api/device/claim", headers=KEYED, json={"flow_id": started["flow_id"]}
    )
    assert replay.status_code == 404
    assert replay.json()["error"]["code"] == "unknown_flow"


@pytest.mark.anyio
async def test_device_claim_denied_drops_the_flow(admin_client, adal_stub):
    started = (
        await admin_client.post("/admin/api/device/start", headers=KEYED, json={})
    ).json()
    adal_stub["next_poll"] = {"status": "denied", "token": None}
    resp = await admin_client.post(
        "/admin/api/device/claim", headers=KEYED, json={"flow_id": started["flow_id"]}
    )
    assert resp.json() == {"status": "denied"}
    again = await admin_client.post(
        "/admin/api/device/claim", headers=KEYED, json={"flow_id": started["flow_id"]}
    )
    assert again.status_code == 404


@pytest.mark.anyio
async def test_device_claim_unknown_flow_is_404(admin_client, adal_stub):
    resp = await admin_client.post(
        "/admin/api/device/claim", headers=KEYED, json={"flow_id": "nope"}
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_device_claim_expired_flow_is_410(admin_client, adal_stub):
    started = (
        await admin_client.post("/admin/api/device/start", headers=KEYED, json={})
    ).json()
    flows = admin_client._transport.app.state.device_flows  # type: ignore[attr-defined]
    flows[started["flow_id"]]["expires_at"] = 0.0
    resp = await admin_client.post(
        "/admin/api/device/claim", headers=KEYED, json={"flow_id": started["flow_id"]}
    )
    assert resp.status_code == 410
    assert resp.json()["error"]["code"] == "flow_expired"
    assert flows == {}


@pytest.mark.anyio
async def test_device_start_reports_an_upstream_failure(admin_client, monkeypatch):
    import sub2api.channels.adal_cloud as adal
    from sub2api.core.errors import AuthError

    def boom(*a, **kw):
        raise AuthError("device flow initiate failed: HTTP 503")

    monkeypatch.setattr(adal, "initiate_device_flow", boom)
    resp = await admin_client.post("/admin/api/device/start", headers=KEYED, json={})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "device_flow_failed"


# -- mounting ----------------------------------------------------------------


@pytest.mark.anyio
async def test_admin_absent_without_web_flag(monkeypatch, tmp_path):
    channel = _StubChannel()
    monkeypatch.setattr(app_mod, "create_channel", lambda name, cfg: channel)
    settings = AppSettings(
        channel="stub",
        host="testserver",
        port=0,
        api_key=API_KEY,
        web=False,
        db=str(tmp_path / "off.sqlite3"),
        channel_config=ChannelConfig(workspace="."),
    )
    app = app_mod.create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        for path in ("/admin", "/admin/api/accounts"):
            assert (await client.get(path, headers=KEYED)).status_code == 404
    await channel.close()
