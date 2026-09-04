"""Clerk identity plumbing for the AdaL cloud channel.

Separated from the channel proper because token minting, the device-code
flow and session registration are the only parts that talk to the platform's
*identity* surface rather than its proxy: they are stubbed independently in
the test suite, and :mod:`sub2api.adal_login` reuses them without dragging in
the account pool or httpx streaming.

Every I/O helper resolves its collaborators through the package namespace
(``_pkg``) instead of a module global, because a test that rebinds
``sub2api.channels.adal_cloud.urlopen`` (or ``initiate_device_flow`` /
``poll_device_flow`` / ``mint_token_with_cookies``) must be seen by the code
that calls it.
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request

import httpx

from ...core.errors import AuthError

# The package module object is registered in ``sys.modules`` before its
# ``__init__`` body runs, so binding it here is safe during package import;
# every attribute lookup through it happens later, at call time.
_pkg = sys.modules[__package__]

ADAL_APP_URL = "https://adal.sylph.ai"
CREDS_PATH = Path.home() / ".adal" / "adal_oauth_creds.json"
# Refresh a token this many seconds before its JWT exp to avoid mid-turn 401s.
TOKEN_REFRESH_SKEW = 60
SESSION_PATH = Path.home() / ".adal" / "adal_session.json"

CLERK_BASE = "https://clerk.adal.sylph.ai"

# Headers that mirror the official client so Clerk treats the mint call as a
# first-party browser/CLI request rather than an anonymous one.
CLERK_MINT_HEADERS = {
    "Origin": "https://adal.sylph.ai",
    "Referer": "https://adal.sylph.ai/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) adal-cli/1.7.2",
}

# Clerk error codes (plus local reasons) that mean the JWT can never be
# re-minted from these credentials; the slot is parked for DEAD_COOLDOWN_S
# instead of being retried on every request.
DEAD_MINT_ERRORS = frozenset({"user_banned", "signed_out", "no_cookies", "no_sid"})
DEAD_COOLDOWN_S = 3600.0
TRANSIENT_COOLDOWN_S = 15.0


def mint_token_with_cookies(
    token: str,
    cookies: list[dict[str, str]] | None,
    *,
    timeout: float = 15.0,
) -> tuple[str | None, str | None]:
    """Mint a fresh Clerk JWT using the persisted ``__client`` cookie.

    The JWT's TTL is 60 seconds, but the Clerk session itself lives much
    longer.  ``POST /v1/client/sessions/{sid}/tokens`` with the session
    cookie mints a new bearer without the interactive device flow.

    Returns ``(fresh_jwt, None)`` on success, or ``(None, error_code)`` on
    failure where ``error_code`` is a Clerk error code (e.g. ``user_banned``,
    ``signed_out``) or one of ``no_cookies`` / ``no_sid`` / ``transient``.
    """
    if not cookies:
        return None, "no_cookies"
    clerk_sid = _jwt_claims(token).get("sid", "")
    if not clerk_sid:
        return None, "no_sid"

    jar = httpx.Cookies()
    for c in cookies:
        jar.set(
            c["name"], c["value"], domain=c.get("domain", ""), path=c.get("path", "/")
        )

    try:
        r = httpx.post(
            f"{CLERK_BASE}/v1/client/sessions/{clerk_sid}/tokens",
            data={"organization_id": "", "token": ""},
            cookies=jar,
            headers=CLERK_MINT_HEADERS,
            timeout=timeout,
        )
    except Exception:
        return None, "transient"
    if r.status_code == 200:
        fresh = r.json().get("jwt")
        if fresh:
            return fresh, None
    try:
        code = r.json()["errors"][0]["code"]
    except Exception:
        code = f"http_{r.status_code}"
    return None, code


def refresh_token_with_cookies(
    token: str,
    cookies: list[dict[str, str]],
    *,
    timeout: float = 15.0,
) -> str:
    """Back-compat wrapper around :func:`mint_token_with_cookies`.

    Returns the fresh JWT, or the original token if the refresh fails (the
    proxy keys off session_id, so an expired bearer may still work).
    """
    fresh, _err = _pkg.mint_token_with_cookies(token, cookies, timeout=timeout)
    return fresh or token


def load_cached_session() -> str | None:
    """Return a previously registered session id from disk, or None."""
    try:
        data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        sid = data.get("session_id")
        if isinstance(sid, str) and sid:
            return sid
    except (OSError, ValueError, TypeError):
        pass
    return None


def save_cached_session(session_id: str) -> None:
    """Persist a registered session id for reuse across restarts."""
    try:
        SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        SESSION_PATH.write_text(
            json.dumps({"session_id": session_id}), encoding="utf-8"
        )
    except OSError:
        pass


def _jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT payload's claims, or ``{}`` when unparseable."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError, TypeError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _jwt_exp(token: str) -> int | None:
    """Return the JWT ``exp`` (unix seconds) or ``None`` if unparseable."""
    try:
        return int(_jwt_claims(token)["exp"])
    except (KeyError, ValueError, TypeError):
        return None


def clerk_id_from_token(token: str) -> str:
    """Return the account's Clerk user id (JWT ``sub``), or ``""``."""
    sub = _jwt_claims(token).get("sub")
    return sub if isinstance(sub, str) else ""


def read_token(path: Path | None = None) -> str | None:
    """Read the cached Clerk access token, or ``None`` if absent/unparsable."""
    try:
        data = json.loads((path or CREDS_PATH).read_text(encoding="utf-8"))
        tok = data.get("access_token")
        return tok if isinstance(tok, str) and tok else None
    except (OSError, ValueError, TypeError):
        return None


def token_needs_refresh(token: str | None, *, now: int | None = None) -> bool:
    """True when ``token`` is missing or expires within the skew window."""
    if not token:
        return True
    exp = _jwt_exp(token)
    if exp is None:
        return False  # can't tell; trust it until the proxy rejects it
    return exp - (now if now is not None else int(time.time())) <= TOKEN_REFRESH_SKEW


def initiate_device_flow(
    app_url: str = ADAL_APP_URL, timeout: float = 15.0
) -> dict[str, Any]:
    """Start the OAuth device-code flow; return the initiate response dict."""
    req = Request(
        f"{app_url}/api/auth/device/initiate",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _pkg.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:  # pragma: no cover - network path
        raise AuthError(f"device flow initiate failed: HTTP {exc.code}") from exc
    except OSError as exc:
        raise AuthError(f"device flow initiate failed: {exc}") from exc


def poll_device_flow(
    device_code: str, app_url: str = ADAL_APP_URL, timeout: float = 15.0
) -> dict[str, Any]:
    """Poll the device flow once; returns ``{status, token?, ...}``."""
    req = Request(
        f"{app_url}/api/auth/device/poll",
        data=json.dumps({"device_code": device_code}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _pkg.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:  # pragma: no cover - network path
        raise AuthError(f"device flow poll failed: HTTP {exc.code}") from exc
    except OSError as exc:
        raise AuthError(f"device flow poll failed: {exc}") from exc


def device_flow_login(
    *,
    app_url: str = ADAL_APP_URL,
    poll_interval: float = 2.0,
    max_attempts: int = 300,
    on_pending=None,
    creds_path: Path | None = None,
) -> str:
    """Run the interactive device-code flow and persist the resulting token.

    ``on_pending(initiate_response)`` is called once with the verification URL
    and user code so a CLI/embedder can surface them to the human; it defaults
    to printing to stderr.  Blocks until the user authorizes (or the device
    code expires).  Returns the fresh access token and writes it to
    ``creds_path`` (default ``~/.adal/adal_oauth_creds.json``).
    """
    init = _pkg.initiate_device_flow(app_url)
    if on_pending is not None:
        on_pending(init)
    else:  # pragma: no cover - human-in-the-loop default
        import sys

        sys.stderr.write(
            f"[adal-cloud] Open {init['verification_url']} and enter code "
            f"{init['user_code']}\n"
        )
    device_code = init["device_code"]
    for _ in range(max_attempts):
        result = _pkg.poll_device_flow(device_code, app_url)
        if result.get("token"):
            token = result["token"]
            exp = _jwt_exp(token)
            path = creds_path or CREDS_PATH
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {"access_token": token, "expiry_date": (exp or 0) * 1000}
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass  # persist best-effort; in-memory token still works
            return token
        if result.get("status") in ("expired", "denied"):
            raise AuthError(f"device flow {result.get('status')}")
        time.sleep(poll_interval)
    raise AuthError("device flow timed out waiting for user authorization")


def register_session(
    *,
    token: str,
    session_id: str,
    app_url: str = ADAL_APP_URL,
    source: str = "cli",
    client_surface: str = "cli",
    client_entrypoint: str = "adal",
    client_version: str | None = "1.7.2",
    os_name: str | None = "linux",
    terminal_program: str | None = None,
    timeout: float = 15.0,
) -> None:
    """Register ``session_id`` with the platform so the proxy accepts it.

    Idempotent: the remote simply upserts on the (user, session_id) pair.
    Retries on transient network errors (the registration is on the
    request hot path, so a single timeout must not 500 the whole turn).
    Raises :class:`AuthError` on a non-2xx response or repeated timeouts.
    """
    body = {
        "session_id": session_id,
        "source": source,
        "client_surface": client_surface,
        "client_entrypoint": client_entrypoint,
    }
    if client_version:
        body["client_version"] = client_version
    if os_name:
        body["os"] = os_name
    if terminal_program:
        body["terminal_program"] = terminal_program
    req = Request(
        f"{app_url}/api/client-sessions/start",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            with _pkg.urlopen(req, timeout=timeout) as resp:
                resp.read()
            return
        except HTTPError as exc:
            raise AuthError(f"session registration failed: HTTP {exc.code}") from exc
        except OSError as exc:
            last_exc = exc
    raise AuthError(f"session registration failed: {last_exc}") from last_exc
