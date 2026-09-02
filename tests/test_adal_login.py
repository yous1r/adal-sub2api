"""Tests for the CLI-free automated login (sub2api.adal_login)."""

from __future__ import annotations

import json
import time

import httpx
import pytest

import sub2api.adal_login as mod
from sub2api.core.errors import AuthError


class FakeClient:
    """Scripted stand-in for httpx.Client used by adal_login."""

    def __init__(self, responses, gets=None):
        self._responses = list(responses)
        self._gets = list(gets or [])
        self.posts: list[tuple[str, dict | None]] = []
        self.gets: list[str] = []
        self.cookies = httpx.Cookies()
        self.headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, data=None, json=None, **kw):
        self.posts.append((url, data))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, **kw):
        self.gets.append(url)
        item = self._gets.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request("POST", "https://clerk.test/v1/client"),
    )


def _happy_responses() -> list[httpx.Response]:
    return [
        _ok({"response": {}}),  # client creation
        _ok({"response": {"id": "sia_1"}}),  # sign_in creation
        _ok(
            {
                "response": {
                    "status": "complete",
                    "created_session_id": "sess_9",
                }
            }
        ),  # first factor
        _ok({"jwt": "fresh.jwt.sig"}),  # tokens
    ]


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Patch module collaborators; capture calls; return the recorder."""
    seen: dict = {
        "register": [],
        "saved": [],
        "clients": [],
        "gets": [],
        "responses": [],
    }

    monkeypatch.setattr(mod, "load_cached_session", lambda: "sub2api-fixed")
    monkeypatch.setattr(
        mod, "save_cached_session", lambda sid: seen["saved"].append(sid)
    )
    monkeypatch.setattr(
        mod,
        "register_session",
        lambda **kw: seen["register"].append(kw),
    )

    def fake_client(*a, **kw):
        client = FakeClient(seen["responses"], seen["gets"])
        client.headers.update(kw.get("headers") or {})
        seen["clients"].append(client)
        return client

    monkeypatch.setattr(mod.httpx, "Client", fake_client)
    seen["creds"] = tmp_path / "creds.json"
    return seen


def test_email_code_login_full_flow(patched):
    patched["responses"] = [
        _ok({"response": {}}),  # client creation
        _ok(
            {
                "response": {
                    "id": "sia_2",
                    "supported_first_factors": [
                        {"strategy": "password"},
                        {"strategy": "email_code", "email_address_id": "ea_1"},
                    ],
                }
            }
        ),  # sign_in creation
        _ok({"response": {"status": "needs_first_factor"}}),  # prepare
        _ok(
            {"response": {"status": "complete", "created_session_id": "sess_7"}}
        ),  # attempt
        _ok({"jwt": "fresh.jwt.sig"}),  # tokens
    ]
    codes = []
    out = mod.email_code_login(
        "user@example.com",
        code_fetcher=lambda since: codes.append(since) or "654321",
        register=True,
        creds_path=patched["creds"],
    )

    assert out["token"] == "fresh.jwt.sig"
    assert out["clerk_session_id"] == "sess_7"
    assert out["session_id"] == "sub2api-fixed"
    assert patched["register"] == [
        {"token": "fresh.jwt.sig", "session_id": "sub2api-fixed"}
    ]
    assert len(codes) == 1  # fetcher received the initiation timestamp
    posts = patched["clients"][0].posts
    assert posts[2][1]["strategy"] == "email_code"
    assert posts[2][1]["email_address_id"] == "ea_1"
    assert posts[3][1] == {
        "strategy": "email_code",
        "code": "654321",
        "_clerk_js_version": mod._CLERK_JS_VERSION,
    }
    saved = json.loads(patched["creds"].read_text())
    assert saved["access_token"] == "fresh.jwt.sig"


def test_email_code_login_missing_factor(patched):
    patched["responses"] = [
        _ok({"response": {}}),
        _ok({"response": {"id": "sia_2", "supported_first_factors": []}}),
    ]
    with pytest.raises(AuthError, match="email_code factor not available"):
        mod.email_code_login(
            "user@example.com",
            code_fetcher=lambda since: "123456",
            register=True,
            creds_path=patched["creds"],
        )
    assert patched["register"] == []


def test_email_code_login_wrong_code(patched):
    patched["responses"] = [
        _ok({"response": {}}),
        _ok(
            {
                "response": {
                    "id": "sia_2",
                    "supported_first_factors": [
                        {"strategy": "email_code", "email_address_id": "ea_1"}
                    ],
                }
            }
        ),
        _ok({"response": {"status": "needs_first_factor"}}),
        httpx.Response(
            422,
            json={
                "errors": [
                    {
                        "code": "form_code_incorrect",
                        "message": "Incorrect code.",
                    }
                ]
            },
            request=httpx.Request("POST", "https://clerk.test/v1/client"),
        ),
    ]
    with pytest.raises(AuthError, match="form_code_incorrect"):
        mod.email_code_login(
            "user@example.com",
            code_fetcher=lambda since: "000000",
            register=True,
            creds_path=patched["creds"],
        )


def test_fetch_code_from_moemail(patched, monkeypatch):
    now_ms = int(time.time() * 1000)
    patched["gets"] = [
        _ok({"emails": [{"id": "mb1", "email": "user@example.com"}]}),
        _ok(
            {
                "messages": [
                    {
                        "id": "m1",
                        "createdAt": now_ms,
                        "subject": "Sign in code",
                    }
                ]
            }
        ),
        _ok({"content": "<p>Your verification code is <b>998877</b></p>"}),
    ]
    code = mod.fetch_code_from_moemail(
        "user@example.com",
        base_url="http://moemail.test/",
        api_key="key-1",
        since=time.time() - 30,
        timeout=5,
    )
    assert code == "998877"
    client = patched["clients"][0]
    assert client.gets == [
        "http://moemail.test/api/emails",
        "http://moemail.test/api/emails/mb1",
        "http://moemail.test/api/emails/mb1/m1",
    ]
    assert client.headers["X-API-Key"] == "key-1"


def test_fetch_code_from_moemail_skips_stale(patched, monkeypatch):
    old_ms = int((time.time() - 600) * 1000)
    now_ms = int(time.time() * 1000)
    patched["gets"] = [
        _ok({"emails": [{"id": "mb1", "email": "user@example.com"}]}),
        _ok(
            {
                "messages": [
                    {"id": "old", "createdAt": old_ms, "subject": "old code"},
                    {"id": "new", "createdAt": now_ms, "subject": "new code"},
                ]
            }
        ),
        # the stale message is skipped WITHOUT a detail fetch; only the
        # fresh one is read back
        _ok({"content": "fresh code 222333"}),
    ]
    code = mod.fetch_code_from_moemail(
        "user@example.com",
        base_url="http://moemail.test",
        api_key="key-1",
        since=time.time() - 30,
        timeout=5,
    )
    assert code == "222333"


def test_fetch_code_from_moemail_no_mailbox(patched):
    patched["gets"] = [_ok({"emails": [{"id": "mb1", "email": "other@example.com"}]})]
    with pytest.raises(AuthError, match="no mailbox found"):
        mod.fetch_code_from_moemail(
            "user@example.com",
            base_url="http://moemail.test",
            api_key="key-1",
            since=time.time() - 30,
            timeout=5,
        )


def test_password_login_full_flow(patched):
    patched["responses"] = _happy_responses()
    out = mod.password_login(
        "user@example.com",
        "secret",
        register=True,
        creds_path=patched["creds"],
    )

    assert out["token"] == "fresh.jwt.sig"
    assert out["clerk_session_id"] == "sess_9"
    assert out["session_id"] == "sub2api-fixed"
    # registration called with the fresh token and the reused session id
    assert patched["register"] == [
        {"token": "fresh.jwt.sig", "session_id": "sub2api-fixed"}
    ]
    # session id persisted for reuse
    assert patched["saved"] == ["sub2api-fixed"]
    # creds file holds the token
    saved = json.loads(patched["creds"].read_text())
    assert saved["access_token"] == "fresh.jwt.sig"
    # the four Clerk endpoints were hit in order
    urls = [u for u, _ in patched["clients"][0].posts]
    assert "/v1/client" in urls[0]
    assert "/v1/client/sign_ins" in urls[1]
    assert "/attempt_first_factor" in urls[2]
    assert "/tokens" in urls[3]
    # the password was sent to the factor endpoint only
    assert patched["clients"][0].posts[2][1]["password"] == "secret"


def test_password_login_surfaces_clerk_error_code(patched):
    patched["responses"] = [
        _ok({"response": {}}),
        _ok({"response": {"id": "sia_1"}}),
        httpx.Response(
            422,
            json={
                "errors": [
                    {
                        "code": "form_password_incorrect",
                        "message": "Password is incorrect.",
                    }
                ]
            },
            request=httpx.Request("POST", "https://clerk.test/v1/client"),
        ),
    ]
    with pytest.raises(AuthError, match="form_password_incorrect"):
        mod.password_login(
            "user@example.com",
            "bad",
            register=True,
            creds_path=patched["creds"],
        )
    assert patched["register"] == []  # nothing registered on failure


def test_password_login_identifier_not_found(patched):
    patched["responses"] = [
        _ok({"response": {}}),
        _ok({"response": {"id": "sia_1"}}),
        httpx.Response(
            404,
            json={
                "errors": [
                    {
                        "code": "form_identifier_not_found",
                        "message": "Couldn't find your account.",
                    }
                ]
            },
            request=httpx.Request("POST", "https://clerk.test/v1/client"),
        ),
    ]
    with pytest.raises(AuthError, match="form_identifier_not_found"):
        mod.password_login(
            "ghost@example.com",
            "whatever",
            register=True,
            creds_path=patched["creds"],
        )


def test_password_login_incomplete_factor(patched):
    patched["responses"] = [
        _ok({"response": {}}),
        _ok({"response": {"id": "sia_1"}}),
        _ok({"response": {"status": "needs_second_factor"}}),
    ]
    with pytest.raises(AuthError, match="needs_second_factor"):
        mod.password_login(
            "user@example.com",
            "secret",
            register=True,
            creds_path=patched["creds"],
        )


def test_password_login_can_skip_registration(patched):
    patched["responses"] = _happy_responses()
    mod.password_login(
        "user@example.com", "secret", register=False, creds_path=patched["creds"]
    )
    assert patched["register"] == []
