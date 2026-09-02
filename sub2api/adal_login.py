"""CLI-free AdaL account bootstrap: automated login + session registration.

Modes (``python -m sub2api.adal_login``):

* **Password login** (fully automated, no browser, no ``adal`` CLI):
  ``--email you@example.com --password ...``
  Signs in to the Clerk tenant over plain HTTP, mints the short-lived
  bearer JWT, registers a session with the platform, and persists the
  token plus the Clerk cookies so the server can re-mint silently.

* **Email-code login** (for passwordless accounts):
  ``--email you@example.com [--moemail-url URL --moemail-key KEY]``
  Requests a Clerk email verification code and completes sign-in with it.
  With MoeMail flags the code is fetched from the mailbox API
  automatically; otherwise it is prompted for interactively.

* **Device flow** (default, one browser click):
  runs the same OAuth device flow the ``adal`` CLI uses.

The adal CLI is never required for the ``adal-cloud`` channel — it is only
needed by the legacy ``adal-backend`` local-runtime channel.  Account
*sign-up* (creating a new account) is captcha-protected and stays manual;
this tool automates everything after a valid account exists.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import httpx

from .channels.adal_cloud import (
    ADAL_APP_URL,
    CLERK_BASE,
    CREDS_PATH,
    _jwt_exp,
    load_cached_session,
    mint_token_with_cookies,
    register_session,
    save_cached_session,
)
from .core.errors import AuthError

# Advisory version tag Clerk's web client sends; the tenant does not pin it.
_CLERK_JS_VERSION = "5.56.0"


def _record_cookies(jar: dict[tuple[str, str, str], dict[str, str]], resp: httpx.Response) -> None:
    for cookie in resp.cookies.jar:
        name = cookie.name if isinstance(cookie.name, str) else cookie.name.decode()
        value = cookie.value or ""
        if not isinstance(value, str):
            value = value.decode()
        jar[(name, cookie.domain or "", cookie.path or "/")] = {
            "name": name,
            "value": value,
            "domain": cookie.domain or "",
            "path": cookie.path or "/",
        }


def _persist(path: Path, token: str, cookies: list[dict[str, str]]) -> None:
    payload = {"access_token": token, "expiry_date": (_jwt_exp(token) or 0) * 1000}
    if cookies:
        payload["cookies"] = cookies
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not persist creds: {exc}", file=sys.stderr)


def _clerk_payload(resp: httpx.Response) -> dict:
    """Parse a Clerk API response, raising :class:`AuthError` on errors."""
    try:
        data = resp.json()
    except ValueError:
        raise AuthError(f"clerk: non-json response (HTTP {resp.status_code})") from None
    if resp.status_code >= 400:
        errors = data.get("errors") or []
        if errors:
            code = errors[0].get("code", "unknown")
            msg = errors[0].get("long_message") or errors[0].get("message", "")
            raise AuthError(f"clerk {code}: {msg}")
        raise AuthError(f"clerk: HTTP {resp.status_code}")
    return data


def _mint_register_persist(
    client: httpx.Client,
    jar: dict[tuple[str, str, str], dict[str, str]],
    attempt_response: dict,
    *,
    clerk_base: str,
    register: bool,
    creds_path: Path | None,
    session_id: str,
) -> dict:
    """Shared login tail: mint the bearer, register the session, persist."""
    clerk_sid = attempt_response["created_session_id"]
    r = client.post(
        f"{clerk_base}/v1/client/sessions/{clerk_sid}/tokens",
        data={
            "organization_id": "",
            "token": "",
            "_clerk_js_version": _CLERK_JS_VERSION,
        },
    )
    _record_cookies(jar, r)
    token = _clerk_payload(r)["jwt"]
    cookies = list(jar.values())
    if register:
        register_session(token=token, session_id=session_id)
    _persist(creds_path or CREDS_PATH, token, cookies)
    save_cached_session(session_id)
    return {
        "token": token,
        "clerk_session_id": clerk_sid,
        "session_id": session_id,
        "cookies": cookies,
    }


def password_login(
    email: str,
    password: str,
    *,
    clerk_base: str = CLERK_BASE,
    register: bool = True,
    creds_path: Path | None = None,
    timeout: float = 20.0,
) -> dict:
    """Automated single-account registration without the adal CLI.

    Signs in with email+password via Clerk's web flow, mints a fresh
    bearer, registers a session with the platform, and persists the token
    and Clerk cookies.  Returns a dict with the token, the Clerk session
    id, the registered session id, and the captured cookies.
    """
    session_id = load_cached_session() or f"sub2api-{uuid.uuid4().hex[:12]}"
    jar: dict[tuple[str, str, str], dict[str, str]] = {}
    with httpx.Client(timeout=timeout) as client:
        # 1. Browser-style client — creates the session the __client cookie
        #    keys off (the same cookie mint_token_with_cookies needs later).
        r = client.post(
            f"{clerk_base}/v1/client",
            data={"_clerk_js_version": _CLERK_JS_VERSION},
        )
        _record_cookies(jar, r)
        _clerk_payload(r)
        # 2. Create a sign-in for the identifier.
        r = client.post(
            f"{clerk_base}/v1/client/sign_ins",
            data={
                "identifier": email,
                "_clerk_js_version": _CLERK_JS_VERSION,
            },
        )
        _record_cookies(jar, r)
        sign_in = _clerk_payload(r)["response"]
        sign_in_id = sign_in["id"]
        # 3. First factor: password.  (The tenant enables email_code as an
        #    alternative, which would need a mailbox — out of scope.)
        r = client.post(
            f"{clerk_base}/v1/client/sign_ins/{sign_in_id}/attempt_first_factor",
            data={
                "strategy": "password",
                "password": password,
                "_clerk_js_version": _CLERK_JS_VERSION,
            },
        )
        _record_cookies(jar, r)
        res = _clerk_payload(r)["response"]
        status = res.get("status")
        if status != "complete":
            raise AuthError(f"clerk sign-in stopped at status={status}")
        return _mint_register_persist(
            client,
            jar,
            res,
            clerk_base=clerk_base,
            register=register,
            creds_path=creds_path,
            session_id=session_id,
        )


def email_code_login(
    email: str,
    *,
    code_fetcher: Callable[[float], str] | None = None,
    clerk_base: str = CLERK_BASE,
    register: bool = True,
    creds_path: Path | None = None,
    timeout: float = 20.0,
) -> dict:
    """Sign in to a passwordless account with an email verification code.

    ``code_fetcher(since_epoch)`` must return the 6-digit code sent to
    ``email`` after the request was initiated at ``since_epoch``; pass
    :func:`fetch_code_from_moemail` wired to a MoeMail instance for full
    automation, or leave ``None`` for an interactive prompt.
    """
    session_id = load_cached_session() or f"sub2api-{uuid.uuid4().hex[:12]}"
    jar: dict[tuple[str, str, str], dict[str, str]] = {}
    with httpx.Client(timeout=timeout) as client:
        r = client.post(
            f"{clerk_base}/v1/client",
            data={"_clerk_js_version": _CLERK_JS_VERSION},
        )
        _record_cookies(jar, r)
        _clerk_payload(r)
        r = client.post(
            f"{clerk_base}/v1/client/sign_ins",
            data={
                "identifier": email,
                "_clerk_js_version": _CLERK_JS_VERSION,
            },
        )
        _record_cookies(jar, r)
        sign_in = _clerk_payload(r)["response"]
        sign_in_id = sign_in["id"]
        factors = sign_in.get("supported_first_factors") or []
        ea_id = next(
            (
                f.get("email_address_id")
                for f in factors
                if isinstance(f, dict)
                and f.get("strategy") == "email_code"
                and f.get("email_address_id")
            ),
            None,
        )
        if ea_id is None:
            raise AuthError(
                "clerk: email_code factor not available for this account "
                "(no email_address_id in supported_first_factors)"
            )
        since = time.time()
        r = client.post(
            f"{clerk_base}/v1/client/sign_ins/{sign_in_id}/prepare_first_factor",
            data={
                "strategy": "email_code",
                "email_address_id": ea_id,
                "_clerk_js_version": _CLERK_JS_VERSION,
            },
        )
        _record_cookies(jar, r)
        _clerk_payload(r)
        if code_fetcher is not None:
            code = code_fetcher(since)
        else:
            code = input(f"Enter the verification code sent to {email}: ").strip()
        r = client.post(
            f"{clerk_base}/v1/client/sign_ins/{sign_in_id}/attempt_first_factor",
            data={
                "strategy": "email_code",
                "code": code,
                "_clerk_js_version": _CLERK_JS_VERSION,
            },
        )
        _record_cookies(jar, r)
        res = _clerk_payload(r)["response"]
        if res.get("status") != "complete":
            raise AuthError(f"clerk sign-in stopped at status={res.get('status')}")
        return _mint_register_persist(
            client,
            jar,
            res,
            clerk_base=clerk_base,
            register=register,
            creds_path=creds_path,
            session_id=session_id,
        )


# -- MoeMail integration -----------------------------------------------------


_CODE_RE = re.compile(r"\b(\d{6})\b")


def _find_mailbox(client: httpx.Client, base_url: str, email: str) -> str:
    r = client.get(f"{base_url}/api/emails")
    if r.status_code >= 400:
        raise AuthError(f"moemail: list mailboxes failed: HTTP {r.status_code}")
    try:
        data = r.json()
    except ValueError:
        raise AuthError("moemail: non-json response listing mailboxes") from None
    items = data if isinstance(data, list) else (
        next((v for v in data.values() if isinstance(v, list)), [])
    )
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("email") or item.get("address") or "") == email:
            if item.get("id"):
                return str(item["id"])
    raise AuthError(f"moemail: no mailbox found for {email}")


def _list_messages(client: httpx.Client, base_url: str, mailbox_id: str) -> list:
    r = client.get(f"{base_url}/api/emails/{mailbox_id}")
    if r.status_code >= 400:
        return []
    try:
        data = r.json()
    except ValueError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def _msg_epoch(message: dict) -> float:
    for key in ("createdAt", "created_at", "receivedAt"):
        value = message.get(key)
        if value is None:
            continue
        try:
            if isinstance(value, (int, float)):
                return value / 1000 if value > 1e12 else float(value)
            text = str(value)
            if text.isdigit():
                number = float(text)
                return number / 1000 if number > 1e12 else number
            from datetime import datetime

            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
    return 0.0  # unknown time: treated as fresh


def _extract_code(detail: object) -> str | None:
    if isinstance(detail, dict):
        text = str(
            detail.get("content") or detail.get("text") or detail.get("html") or ""
        )
    elif isinstance(detail, str):
        text = detail
    else:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    match = _CODE_RE.search(text)
    return match.group(1) if match else None


def fetch_code_from_moemail(
    email: str,
    *,
    base_url: str,
    api_key: str,
    since: float,
    timeout: float = 120.0,
    interval: float = 3.0,
) -> str:
    """Poll a self-hosted MoeMail instance for the Clerk verification code.

    ``since`` is the unix second just before the sign-in was initiated;
    messages older than that are ignored so stale codes are never used.
    """
    base = base_url.rstrip("/")
    with httpx.Client(timeout=15.0, headers={"X-API-Key": api_key}) as client:
        mailbox_id = _find_mailbox(client, base, email)
        deadline = time.monotonic() + timeout
        while True:
            for message in _list_messages(client, base, mailbox_id):
                if not isinstance(message, dict) or not message.get("id"):
                    continue
                epoch = _msg_epoch(message)
                if epoch and epoch < since - 60:
                    continue  # predates this sign-in attempt: stale code
                r = client.get(f"{base}/api/emails/{mailbox_id}/{message['id']}")
                if r.status_code >= 400:
                    continue
                try:
                    detail = r.json()
                except ValueError:
                    continue
                code = _extract_code(detail)
                if code:
                    return code
            if time.monotonic() >= deadline:
                break
            time.sleep(interval)
    raise AuthError(
        f"moemail: no verification code for {email} arrived within {int(timeout)}s"
    )


# -- device flow -------------------------------------------------------------


def login(
    *,
    app_url: str = ADAL_APP_URL,
    creds_path: Path | None = None,
    poll_interval: float = 2.0,
    max_attempts: int = 300,
) -> str:
    """Run the interactive device flow; persist token + cookies."""
    path = creds_path or CREDS_PATH
    jar: dict[tuple[str, str, str], dict[str, str]] = {}
    with httpx.Client(timeout=15.0) as client:
        r = client.post(f"{app_url}/api/auth/device/initiate", json={})
        r.raise_for_status()
        _record_cookies(jar, r)
        init = r.json()
        print(
            f"Open {init['verification_url']} and enter code {init['user_code']}",
            file=sys.stderr,
        )
        device_code = init["device_code"]
        for _ in range(max_attempts):
            pr = client.post(
                f"{app_url}/api/auth/device/poll",
                json={"device_code": device_code},
            )
            pr.raise_for_status()
            _record_cookies(jar, pr)
            result = pr.json()
            if result.get("token"):
                token = result["token"]
                _persist(path, token, list(jar.values()))
                return token
            if result.get("status") in ("expired", "denied"):
                raise SystemExit(f"device flow {result.get('status')}")
            time.sleep(poll_interval)
    raise SystemExit("device flow timed out waiting for user authorization")


# -- CLI ---------------------------------------------------------------------


def _cookies_can_mint(token: str) -> bool:
    """Best-effort: confirm the persisted cookies can mint a fresh JWT."""
    try:
        data = json.loads(CREDS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    cookies = data.get("cookies") or []
    if not cookies:
        return False
    fresh, _err = mint_token_with_cookies(token, cookies)
    return bool(fresh)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sub2api.adal_login",
        description="CLI-free AdaL login + session registration",
    )
    parser.add_argument("--email", help="account email")
    parser.add_argument("--password", help="account password (password login)")
    parser.add_argument(
        "--moemail-url",
        help="self-hosted MoeMail base URL (enables automatic code retrieval)",
    )
    parser.add_argument("--moemail-key", help="MoeMail X-API-Key")
    parser.add_argument(
        "--code-timeout",
        type=float,
        default=120.0,
        help="seconds to wait for the verification email (default 120)",
    )
    parser.add_argument(
        "--device",
        action="store_true",
        help="force the interactive device flow even if credentials are given",
    )
    args = parser.parse_args()

    if args.device or not args.email:
        token = login()
        exp = _jwt_exp(token)
        print(f"Login OK (token TTL 60s; expires at unix {exp}).")
        _report_cookie_state(token)
        return

    if args.password:
        out = password_login(args.email, args.password)
        print(f"Login OK; session {out['session_id']} registered (password login).")
    else:
        if args.moemail_url and args.moemail_key:

            def fetcher(since: float) -> str:
                return fetch_code_from_moemail(
                    args.email,
                    base_url=args.moemail_url,
                    api_key=args.moemail_key,
                    since=since,
                    timeout=args.code_timeout,
                )

        else:
            fetcher = None  # interactive prompt
        out = email_code_login(args.email, code_fetcher=fetcher)
        print(f"Login OK; session {out['session_id']} registered (email-code login).")
    _report_cookie_state(out["token"])


def _report_cookie_state(token: str) -> None:
    if _cookies_can_mint(token):
        print("Clerk cookies captured: the server will refresh tokens silently.")
    else:
        print(
            "No usable Clerk cookie captured — re-run this command when the "
            "token stops working.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
