"""AdaL channel backed by the cloud proxy API (no local runtime).

This channel talks **directly** to AdaL's hosted proxy at
``api.adal.sylph.ai`` — there is no ``adal`` CLI, no ``adal --sdk-runtime``
subprocess, and no local ``adal-backend`` binary involved.  The user's
subscription quota is consumed through the same proxy the official clients
use; sub2api merely re-implements the thin client side of that protocol.

Protocol (reverse-engineered from the bundled ``adal-backend``):

  1. authenticate — read a Clerk JWT from ``~/.adal/adal_oauth_creds.json``
     (or ``SUB2API_AUTH_TOKEN``); if absent/expired, run the OAuth **device
     code** flow against ``adal.sylph.ai`` and persist the fresh token.
  2. register session — ``POST {ADAL_APP_URL}/api/client-sessions/start``
     with ``{session_id, source, ...}`` and ``Authorization: Bearer <jwt>``.
     The remote stores the ``session_id`` keyed to the user; one registration
     per session_id is enough and is cached for the channel's lifetime.
  3. chat — forward the request to ``{ADAL_PROXY_URL}/proxy/<path>`` with
     ``X-Session-ID`` and ``X-Target-URL`` headers.  The proxy enforces
     subscription billing and relays to the upstream provider (Anthropic via
     AWS Bedrock, OpenAI, Google, Z.AI, …).  Two shapes are supported:

       * Anthropic native: ``/proxy/v1/messages`` → target ``api.anthropic.com``
       * OpenAI native:    ``/proxy/v1/chat/completions`` → target ``api.openai.com``

  4. model catalog — ``GET {ADAL_PROXY_URL}/proxy/models/catalog`` (no auth)
     yields the live model list keyed by ``provider`` → ``model_client``.

The provider→target base-URL map mirrors ``adal_backend``'s
``_CLIENT_REGISTRY`` so every catalog model routes to its real upstream.

Docs: https://docs.sylph.ai/cloud-agents/overview
"""

from __future__ import annotations

import base64
import uuid
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, AsyncIterator, ClassVar
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import httpx

from ..core.channel import BaseChannel
from ..core.errors import AuthError, RuntimeMissingError, UpstreamError
from ..core.pool import AccountPool, AccountSlot, PoolConfig, load_pool_config
from ..core.registry import register
from ..core.types import (
    ChatRequest,
    Event,
    MessageCompleted,
    TextDelta,
    ThoughtDelta,
    TurnCompleted,
    TurnFailed,
)

ADAL_APP_URL = "https://adal.sylph.ai"
ADAL_PROXY_URL = "https://api.adal.sylph.ai"
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
    fresh, _err = mint_token_with_cookies(token, cookies, timeout=timeout)
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


# provider (from the catalog) -> (proxy sub-path, upstream target base URL).
# Anthropic-style providers POST /proxy/v1/messages; OpenAI-style providers
# POST /proxy/v1/chat/completions.  Targets are the bare host (the proxy keeps
# the /v1/... suffix from the inbound path).
PROVIDER_TARGETS: dict[str, tuple[str, str]] = {
    "anthropic": ("/v1/messages", "https://api.anthropic.com"),
    "openai": ("/v1/chat/completions", "https://api.openai.com"),
    "google": ("/v1/chat/completions", "https://generativelanguage.googleapis.com"),
    "zai": ("/v1/messages", "https://api.z.ai/api/paas/v4"),
    "deepseek": ("/v1/chat/completions", "https://api.deepseek.com"),
    "kimi": ("/v1/chat/completions", "https://api.moonshot.ai/v1"),
    "minimax": ("/v1/messages", "https://api.minimax.io/anthropic"),
    "xai": ("/v1/chat/completions", "https://api.x.ai/v1"),
    "qwen": (
        "/v1/chat/completions",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    ),
    "meta": ("/v1/chat/completions", "https://api.meta.ai/v1"),
}


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
        with urlopen(req, timeout=timeout) as resp:
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
        with urlopen(req, timeout=timeout) as resp:
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
    init = initiate_device_flow(app_url)
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
        result = poll_device_flow(device_code, app_url)
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


def fetch_catalog(
    proxy_url: str = ADAL_PROXY_URL, timeout: float = 20.0
) -> dict[str, Any]:
    """Fetch the live model catalog (no auth required)."""
    req = Request(f"{proxy_url}/proxy/models/catalog")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (HTTPError, OSError, ValueError):
        return {}


def catalog_models(catalog: dict[str, Any]) -> tuple[str, ...]:
    """Extract model keys from a catalog response, in catalog order."""
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(models, list):
        return ()
    keys = tuple(m["key"] for m in models if isinstance(m, dict) and m.get("key"))
    return keys


def provider_for_model(catalog: dict[str, Any], model: str) -> str | None:
    """Look up the provider for a model key/id in the catalog."""
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(models, list):
        return None
    for m in models:
        if not isinstance(m, dict):
            continue
        if m.get("key") == model or m.get("model_id") == model:
            return m.get("provider")
    return None


def upstream_model_id(catalog: dict[str, Any], model: str) -> str:
    """Resolve a request model (key or id) to the upstream model_id."""
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if isinstance(models, list):
        for m in models:
            if isinstance(m, dict) and (
                m.get("key") == model or m.get("model_id") == model
            ):
                return m.get("model_id") or model
    return model


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
            with urlopen(req, timeout=timeout) as resp:
                resp.read()
            return
        except HTTPError as exc:
            raise AuthError(f"session registration failed: HTTP {exc.code}") from exc
        except OSError as exc:
            last_exc = exc
    raise AuthError(f"session registration failed: {last_exc}") from last_exc


# -- subscription quota ------------------------------------------------------
# The platform publishes live subscription state at two endpoints keyed by the
# internal user id:
#
#   GET /api/user/by-clerk-id/{clerk_id} -> {"id": <uuid>, "email", "tier", …}
#   GET /api/subscription/user/{uuid}    -> {"tier", "status",
#                                            "monthly_credits",
#                                            "credits_used_this_period",
#                                            "credits_remaining",
#                                            "current_period_end", …}
#
# Neither requires a bearer: the account is identified by the ``sub`` claim of
# its JWT, so even an expired token resolves its own quota.  Credits are
# dollar-denominated (the free tier's 2.0 credits is described upstream as
# "$2/month"), hence ``USAGE_UNIT``.

USAGE_CACHE_TTL = 30.0  # cc-switch polls on a per-minute timer
TIERS_CACHE_TTL = 600.0  # the tier catalog is effectively static
USAGE_FETCH_CONCURRENCY = 8  # parallel account lookups for large pools
USAGE_UNIT = "USD"


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


async def fetch_tiers(
    client: httpx.AsyncClient,
    *,
    app_url: str = ADAL_APP_URL,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Fetch the tier catalog (``{tier_id: {...}}``); ``{}`` on any failure."""
    try:
        resp = await client.get(
            f"{app_url}/api/subscription/tiers",
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        tiers = resp.json().get("tiers") if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError, AttributeError):
        return {}
    return tiers if isinstance(tiers, dict) else {}


def tier_display_name(tiers: dict[str, Any], tier: str) -> str:
    """Human label for a tier id, falling back to the id itself."""
    entry = tiers.get(tier) if isinstance(tiers, dict) else None
    if isinstance(entry, dict):
        name = entry.get("display_name")
        if isinstance(name, str) and name:
            return name
    return tier


async def fetch_account_quota(
    client: httpx.AsyncClient,
    token: str,
    *,
    app_url: str = ADAL_APP_URL,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Resolve one account's subscription credits from its JWT.

    Returns ``{"ok": True, ...}`` with the credit figures, or
    ``{"ok": False, "error": <reason>}``.  Never raises: a quota lookup must
    not be able to fail the gateway's usage endpoint.
    """
    if not token:
        return {"ok": False, "error": "no auth token"}
    clerk_id = clerk_id_from_token(token)
    if not clerk_id:
        return {"ok": False, "error": "token carries no clerk id"}
    headers = {"Accept": "application/json"}
    try:
        resp = await client.get(
            f"{app_url}/api/user/by-clerk-id/{clerk_id}",
            headers=headers,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"user lookup failed: {exc}"}
    if resp.status_code != 200:
        return {"ok": False, "error": f"user lookup failed: HTTP {resp.status_code}"}
    try:
        user = resp.json()
        user_id = str(user["id"])
    except (ValueError, KeyError, TypeError):
        return {"ok": False, "error": "user lookup returned no id"}
    email = str(user.get("email") or "")
    try:
        resp = await client.get(
            f"{app_url}/api/subscription/user/{user_id}",
            headers=headers,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error": f"subscription lookup failed: {exc}",
            "email": email,
        }
    if resp.status_code == 404:
        return {"ok": False, "error": "no subscription", "email": email}
    if resp.status_code != 200:
        return {
            "ok": False,
            "error": f"subscription lookup failed: HTTP {resp.status_code}",
            "email": email,
        }
    try:
        sub = resp.json()
    except ValueError:
        return {
            "ok": False,
            "error": "subscription lookup returned non-JSON",
            "email": email,
        }
    return {
        "ok": True,
        "email": email,
        "tier": str(sub.get("tier") or user.get("subscription_tier") or ""),
        "status": str(sub.get("status") or ""),
        "billing_interval": str(sub.get("billing_interval") or ""),
        "total": _as_float(sub.get("monthly_credits")),
        "used": _as_float(sub.get("credits_used_this_period")),
        "remaining": _as_float(sub.get("credits_remaining")),
        "is_trialing": bool(sub.get("is_trialing")),
        "period_end": str(sub.get("current_period_end") or ""),
    }


def _invalid_message(
    rows: list[dict[str, Any]],
    resolved: list[dict[str, Any]],
    parked: list[dict[str, Any]],
) -> str:
    """Explain why the gateway cannot currently spend its subscription."""
    if not rows:
        return "no adal account configured"
    if not resolved:
        errors = sorted({str(r.get("error") or "unknown error") for r in rows})
        return "; ".join(errors[:3])
    reasons = sorted({str(r["dead_reason"]) for r in parked})
    return f"all {len(rows)} account(s) parked: {', '.join(reasons)}"


def aggregate_quota(
    rows: list[dict[str, Any]],
    tiers: dict[str, Any],
    *,
    channel: str = "",
    pool_enabled: bool = False,
) -> dict[str, Any]:
    """Fold per-account quota rows into one flat cc-switch usage payload.

    The top-level keys are exactly the fields a cc-switch usage-script
    extractor reads (``isValid``, ``invalidMessage``, ``planName``, ``used``,
    ``total``, ``remaining``, ``unit``, ``extra``).  sub2api's own detail is
    nested under ``sub2api``, which cc-switch ignores.

    Credit figures sum every account whose subscription resolved, including
    parked ones — ``isValid``/``invalidMessage`` carry the "cannot spend it"
    signal instead of silently zeroing the numbers.
    """
    resolved = [r for r in rows if r.get("ok")]
    parked = [r for r in rows if r.get("dead_reason")]
    counts = Counter(
        name
        for name in (
            tier_display_name(tiers, str(r.get("tier") or "")) for r in resolved
        )
        if name
    )
    period_ends = sorted(str(r.get("period_end") or "") for r in resolved)
    notes: list[str] = []
    if len(resolved) == 1 and resolved[0].get("status"):
        notes.append(str(resolved[0]["status"]))
    elif pool_enabled:
        notes.append(f"{len(resolved)}/{len(rows)} accounts")
    if parked:
        reasons = sorted({str(r["dead_reason"]) for r in parked})
        notes.append(f"{len(parked)} parked ({', '.join(reasons)})")
    next_reset = next((end for end in period_ends if end), "")
    if next_reset:
        notes.append(f"resets {next_reset}")
    is_valid = bool(resolved) and len(parked) < len(rows)
    payload: dict[str, Any] = {
        "isValid": is_valid,
        "planName": ", ".join(
            name if count == 1 else f"{name} x{count}"
            for name, count in sorted(counts.items())
        ),
        "total": round(sum(_as_float(r.get("total")) for r in resolved), 6),
        "used": round(sum(_as_float(r.get("used")) for r in resolved), 6),
        "remaining": round(sum(_as_float(r.get("remaining")) for r in resolved), 6),
        "unit": USAGE_UNIT,
        "extra": " · ".join(notes),
    }
    if not is_valid:
        payload["invalidMessage"] = _invalid_message(rows, resolved, parked)
    payload["sub2api"] = {
        "channel": channel,
        "pool_enabled": pool_enabled,
        "accounts": len(rows),
        "accounts_resolved": len(resolved),
        "accounts_parked": len(parked),
        "queried_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detail": rows,
    }
    return payload


# provider (catalog id) -> upstream target base URL (bare host; proxy keeps
# the /v1/... suffix from the inbound path).  Used for passthrough routing.
PROVIDER_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "google": "https://generativelanguage.googleapis.com",
    "zai": "https://api.z.ai/api/paas/v4",
    "deepseek": "https://api.deepseek.com",
    "kimi": "https://api.moonshot.ai/v1",
    "minimax": "https://api.minimax.io/anthropic",
    "xai": "https://api.x.ai/v1",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "meta": "https://api.meta.ai/v1",
}


def target_for_request(path: str, body: dict[str, Any], catalog: dict[str, Any]) -> str:
    """Resolve the upstream ``X-Target-URL`` for a passthrough request.

    Routing by path + model:

    - ``/v1/messages`` is Anthropic-native → always targets the Anthropic host.
    - ``/v1/chat/completions`` and ``/v1/responses`` are OpenAI-native; the
      provider is inferred from the request's ``model`` field via the
      catalog, defaulting to OpenAI.  ``/v1/responses`` is the modern
      stateful Responses endpoint (reasoning models, background mode,
      polling) and routes to the same upstream hosts as chat/completions.
    - Any other path defaults to Anthropic.
    """
    if path.endswith("/messages"):
        return PROVIDER_BASE_URLS["anthropic"]
    model = str(body.get("model") or "")
    provider = provider_for_model(catalog, model)
    if provider is None and model:
        provider = model.split("-", 1)[0] if "-" in model else None
    return PROVIDER_BASE_URLS.get(provider or "", PROVIDER_BASE_URLS["openai"])


def anthropic_request(model: str, request: ChatRequest) -> dict[str, Any]:
    """Build an Anthropic ``/v1/messages`` body for one sub2api turn."""
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": True,
    }
    if request.thinking_effort:
        # "adaptive" lets the model/proxy decide thinking depth from effort;
        # the proxy maps effort to provider-native thinking config server-side.
        body["thinking"] = {"type": "adaptive"}
    return body


def openai_request(model: str, request: ChatRequest) -> dict[str, Any]:
    """Build an OpenAI ``/v1/chat/completions`` body for one sub2api turn."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": True,
    }
    if request.thinking_effort:
        body["reasoning_effort"] = request.thinking_effort
    return body


def parse_anthropic_sse(raw: str) -> list[Event]:
    """Translate one Anthropic SSE event line into normalized events.

    Anthropic streams ``event: <type>\\ndata: <json>`` pairs.  This parser is
    driven by the ``data:`` payload's ``type`` and delta shape; the ``event:``
    line is informational.
    """
    if not raw or not raw.startswith("data:"):
        return []
    payload = raw[5:].strip()
    if not payload:
        return []
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return []
    typ = obj.get("type")
    if typ == "content_block_delta":
        delta = obj.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta":
            return [TextDelta(text=delta.get("text", ""))]
        if dtype in ("thinking_delta", "signature_delta"):
            return [ThoughtDelta(text=delta.get("thinking", delta.get("text", "")))]
        return []
    if typ == "message_stop":
        return []  # terminal handled by caller via the non-stream envelope
    if typ == "error":
        err = obj.get("error") or {}
        raise UpstreamError(err.get("message", "anthropic upstream error"))
    return []


def parse_openai_sse(raw: str) -> list[Event]:
    """Translate one OpenAI chat-completion SSE chunk into normalized events."""
    if not raw or not raw.startswith("data:"):
        return []
    payload = raw[5:].strip()
    if not payload:
        return []
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if obj.get("error"):
        raise UpstreamError(str(obj["error"].get("message", "openai upstream error")))
    choices = obj.get("choices") or []
    if not choices:
        return []
    delta = choices[0].get("delta") or {}
    events: list[Event] = []
    if delta.get("reasoning_content"):
        events.append(ThoughtDelta(text=delta["reasoning_content"]))
    if delta.get("content"):
        events.append(TextDelta(text=delta["content"]))
    return events


# -- prompt cache helpers ----------------------------------------------------


def _has_cache_control(system: Any) -> bool:
    """True when *system* already carries ``cache_control`` somewhere."""
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                return True
    elif isinstance(system, dict):
        return "cache_control" in system
    elif isinstance(system, str):
        # Plain string system — no cache_control possible.
        pass
    return False


def _inject_cache_control(system: Any) -> None:
    """Inject ``cache_control: {type:"ephemeral"}`` on the last block of
    *system* in place.  Handles string, dict, and list-of-dicts formats.
    """
    ephemeral: dict[str, str] = {"type": "ephemeral"}
    if isinstance(system, str):
        # Can't mutate a string; caller should have converted it.
        return
    if isinstance(system, dict):
        system["cache_control"] = ephemeral
        return
    if isinstance(system, list) and system:
        last = system[-1]
        if isinstance(last, dict):
            last["cache_control"] = ephemeral


@register
class AdalCloudChannel(BaseChannel):
    """Direct cloud-proxy channel: no ``adal`` install required.

    The user authenticates once (cached JWT or interactive device flow); each
    sub2api session registers its id with the platform lazily, then every
    turn streams from ``api.adal.sylph.ai/proxy/*`` with the subscription
    footing the bill.
    """

    name: ClassVar[str] = "adal-cloud"
    proxy_url: ClassVar[str] = ADAL_PROXY_URL
    display_name: ClassVar[str] = "AdaL (cloud proxy)"
    models: ClassVar[tuple[str, ...]] = (
        "anthropic-claude-sonnet-5",
        "anthropic-claude-opus-5",
        "openai-gpt-5.6-terra",
        "google-gemini-3.7-flash",
        "zai-glm-5.2",
        "deepseek-deepseek-v4-pro",
        "kimi-kimi-k3",
    )

    _token: str | None
    _catalog: dict[str, Any]
    _registered: set[str]
    _client: httpx.AsyncClient | None
    _pool: AccountPool | None
    _proxy_sid: str | None
    _usage_cache: tuple[float, dict[str, Any]] | None
    _tiers_cache: tuple[float, dict[str, Any]] | None

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._token = None
        self._catalog = {}
        self._registered = set()
        self._client = None
        self._pool = None
        self._proxy_sid = None
        self._refresh_lock = asyncio.Lock()
        self._pool_signature: tuple[Any, ...] | None = None
        self._slot_locks: dict[str, asyncio.Lock] = {}
        self._usage_cache = None
        self._tiers_cache = None
        self._usage_lock = asyncio.Lock()

    @staticmethod
    def _pool_signature_for(config: PoolConfig | None) -> tuple[Any, ...] | None:
        if config is None:
            return None
        return (
            config.strategy,
            config.max_failures,
            config.cooldown_seconds,
            tuple(
                (a.token, a.session_id, a.max_concurrent, repr(a.cookies))
                for a in config.accounts
            ),
        )

    async def refresh(self) -> None:
        """Refresh the catalog and adopt changed pool definitions safely."""
        if not self.started:
            return
        async with self._refresh_lock:
            config = await asyncio.to_thread(load_pool_config)
            signature = self._pool_signature_for(config)
            pool_changed = signature != self._pool_signature
            if not pool_changed:
                if self._pool is None:
                    catalog = await asyncio.to_thread(fetch_catalog, self.proxy_url)
                    models = catalog_models(catalog)
                    if models:
                        self._catalog = catalog
                        self.models = models
                return
            if config is None or not config.accounts:
                if self._pool is not None and not await self._pool.close_if_idle():
                    return
                if self._pool is not None:
                    await self._pool.close()
                self._pool = None
                self._pool_signature = signature
                self._registered.clear()
                if self._client is not None and self._token:
                    self._client.headers["Authorization"] = f"Bearer {self._token}"
                return
            if self._pool is None:
                self._pool = AccountPool(config)
                await self._pool.start()
            elif not await self._pool.reconfigure(config):
                return
            catalog = await asyncio.to_thread(fetch_catalog, self.proxy_url)
            models = catalog_models(catalog)
            if models:
                self._catalog = catalog
                self.models = models
            self._pool_signature = signature
            active_sessions = {slot.session_id for slot in self._pool.slots}
            self._registered.intersection_update(active_sessions)
            for slot in self._pool.slots:
                if slot.pre_registered:
                    self._registered.add(slot.session_id)
                elif slot.session_id not in self._registered:
                    try:
                        await asyncio.to_thread(
                            register_session,
                            token=slot.token,
                            session_id=slot.session_id,
                        )
                        self._registered.add(slot.session_id)
                    except AuthError:
                        pass

    # -- token lifecycle ---------------------------------------------------

    def _slot_lock(self, session_id: str) -> asyncio.Lock:
        """One refresh lock per session id so concurrent requests don't
        stampede Clerk with duplicate mint calls for the same account."""
        lock = self._slot_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._slot_locks[session_id] = lock
        return lock

    async def _mint_slot_token(self, slot: AccountSlot) -> bool:
        """Re-mint ``slot``'s JWT via its Clerk cookies and update its state.

        Returns True when a fresh token is now on the slot.  On a permanent
        failure (banned user, dead cookies) the slot is parked for
        ``DEAD_COOLDOWN_S`` with a ``dead_reason`` so the scheduler stops
        routing requests to it; transient failures get a short cooldown.
        """
        fresh, err = await asyncio.to_thread(
            mint_token_with_cookies, slot.token, slot.cookies or []
        )
        if fresh:
            slot.token = fresh
            slot.fail_count = 0
            slot.disabled_until = 0.0
            slot.dead_reason = ""
            return True
        now = time.monotonic()
        if err in DEAD_MINT_ERRORS:
            slot.dead_reason = err
            slot.disabled_until = now + DEAD_COOLDOWN_S
        else:
            slot.disabled_until = now + TRANSIENT_COOLDOWN_S
        return False

    async def _ensure_slot_token(self, slot: AccountSlot) -> None:
        """Lazily re-mint the slot's JWT when it is expired or near expiry.

        Clerk JWTs carry a 60-second TTL; this keeps a long-running server
        working without restarts.  Slots without cookies (nothing to mint
        from) and already-fresh tokens are skipped.  Failures park the slot
        (see :meth:`_mint_slot_token`).
        """
        if not slot.cookies or not token_needs_refresh(slot.token):
            return
        async with self._slot_lock(slot.session_id):
            if token_needs_refresh(slot.token):
                await self._mint_slot_token(slot)

    async def refresh_slot_auth(
        self, slot: AccountSlot | None, *, force: bool = False
    ) -> bool:
        """Re-mint auth after the proxy rejected a request (HTTP 401).

        Returns True when a usable token is now available and the caller
        should retry once with rebuilt headers.  ``force`` bypasses the
        freshness check — used when the bearer looked valid but the proxy
        still rejected it (e.g. unparseable exp, clock drift).

        In single-account mode (``slot is None``) the cached creds file is
        re-read instead; an external ``adal`` login may have refreshed it.
        """
        if slot is None:
            return await self._refresh_single_account_token()
        async with self._slot_lock(slot.session_id):
            if (
                not force
                and not slot.dead_reason
                and not token_needs_refresh(slot.token)
            ):
                return True  # a concurrent request already refreshed it
            return await self._mint_slot_token(slot)

    async def _refresh_single_account_token(self) -> bool:
        """Swap in a fresher token from the creds file, if one appeared."""
        tok = await asyncio.to_thread(read_token)
        if not tok or tok == self._token or token_needs_refresh(tok):
            return False
        self._token = tok
        if self._client is not None:
            self._client.headers["Authorization"] = f"Bearer {tok}"
        return True

    # -- lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        catalog = await asyncio.to_thread(fetch_catalog)
        models = catalog_models(catalog)
        if models:
            self.models = models
        self._catalog = catalog
        # Resolve auth token: explicit config wins, then cached creds. If the
        # cached token is expired we still try it — the proxy may accept it
        # (it keys off the session_id, not the bearer, for /proxy/*). Only run
        # the interactive device flow when no token exists at all.
        token = self.config.auth_token or read_token()
        if not token:
            token = await asyncio.to_thread(device_flow_login)
        self._token = token
        # Multi-account pool: when configured, the pool owns per-account tokens
        # and sessions. The shared httpx client is token-agnostic (the
        # Authorization header is set per-request from the acquired slot).
        pool_cfg = await asyncio.to_thread(load_pool_config)
        if pool_cfg is not None and pool_cfg.accounts:
            self._pool_signature = self._pool_signature_for(pool_cfg)
            self._pool = AccountPool(pool_cfg)
            await self._pool.start()
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(600.0, connect=15.0, read=600.0),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )
            # JWTs carry a 60s TTL, so bulk-refreshing at startup is pointless
            # (they'd all be stale again within a minute).  Tokens are
            # re-minted lazily per request in _ensure_slot_token, and again
            # with refresh_slot_auth when the proxy answers 401.

            # Register only sessions the config did not already register.
            # adal-registrar calls /api/client-sessions/start during signup
            # and writes the session_id into accounts.json; those skip here.
            unregistered = [s for s in self._pool.slots if not s.pre_registered]

            async def _register_slot(slot):
                try:
                    await asyncio.to_thread(
                        register_session,
                        token=slot.token,
                        session_id=slot.session_id,
                    )
                    self._registered.add(slot.session_id)
                except AuthError:
                    pass  # proxy will upsert on first request

            if unregistered:
                await asyncio.gather(*(_register_slot(s) for s in unregistered))
            # Pre-registered sessions are assumed live; mark them so the first
            # request does not try to re-register.
            for slot in self._pool.slots:
                if slot.pre_registered:
                    self._registered.add(slot.session_id)
            return
        # Single-account fallback: embed the token in the shared client.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=15.0, read=600.0),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        # Pre-register a passthrough session.  Reuse a cached session id
        # from disk if available (avoids generating a new id each restart),
        # but always re-register it — the proxy upserts on
        # (user, session_id), and the remote may have dropped the session.
        cached_sid = load_cached_session()
        self._proxy_sid = cached_sid or f"sub2api-{uuid.uuid4().hex[:12]}"
        try:
            await asyncio.to_thread(
                register_session, token=token, session_id=self._proxy_sid
            )
            self._registered.add(self._proxy_sid)
            save_cached_session(self._proxy_sid)
        except AuthError:
            pass  # will retry on first request

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def runtime_available(self) -> bool:
        # No local runtime needed; only a reachable network + a token source.
        return True

    # -- turn pipeline -----------------------------------------------------

    async def _ensure_registered(self, session_id: str) -> None:
        if session_id in self._registered:
            return
        token = self._require_token()
        await asyncio.to_thread(register_session, token=token, session_id=session_id)
        self._registered.add(session_id)

    def _require_token(self) -> str:
        if not self._token:
            raise RuntimeMissingError(
                "adal-cloud: no auth token resolved (set SUB2API_AUTH_TOKEN or run "
                "`adal` once to populate ~/.adal/adal_oauth_creds.json)"
            )
        return self._token

    def _route_for(self, request: ChatRequest) -> tuple[str, str, str]:
        """Return ``(proxy_sub_path, target_base_url, upstream_model_id)``."""
        model = request.model or (self.models[0] if self.models else "")
        provider = provider_for_model(self._catalog, model)
        if provider is None and model:
            # Fall back to the model key prefix (e.g. "anthropic-..." -> anthropic).
            provider = model.split("-", 1)[0] if "-" in model else None
        target = PROVIDER_TARGETS.get(provider or "")
        if target is None:
            # Default to Anthropic routing for the canonical AdaL models.
            target = PROVIDER_TARGETS["anthropic"]
        sub_path, target_url = target
        upstream_model = upstream_model_id(self._catalog, model)
        return sub_path, target_url, upstream_model

    async def _chat(self, request: ChatRequest) -> AsyncIterator[Event]:
        assert self._client is not None
        await self.refresh()
        # sub2api allocates the native session id; reuse it across turns.
        session_id = request.native_session_id or f"sub2api-{request.session_id}"
        await self._ensure_registered(session_id)
        sub_path, target_url, upstream_model = self._route_for(request)
        url = f"{ADAL_PROXY_URL}/proxy{sub_path}"
        is_anthropic = sub_path.endswith("/messages")
        body = (
            anthropic_request(upstream_model, request)
            if is_anthropic
            else openai_request(upstream_model, request)
        )
        headers: dict[str, str] = {
            "X-Session-ID": session_id,
            "X-Target-URL": target_url,
        }
        slot: AccountSlot | None = None
        if self._pool is not None:
            slot = await self._pool.acquire()
            await self._ensure_slot_token(slot)
            await self._ensure_registered(slot.session_id)
            headers["Authorization"] = f"Bearer {slot.token}"
            headers["X-Session-ID"] = slot.session_id
        full_text: list[str] = []
        try:
            async with self._client.stream(
                "POST", url, json=body, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", errors="replace")[
                        :500
                    ]
                    raise UpstreamError(f"proxy HTTP {resp.status_code}: {detail}")
                parser = parse_anthropic_sse if is_anthropic else parse_openai_sse
                async for line in resp.aiter_lines():
                    events = parser(line)
                    for event in events:
                        if isinstance(event, TextDelta):
                            full_text.append(event.text)
                        yield event
        except UpstreamError:
            if slot is not None:
                await self._pool.release(slot, success=False)
            raise
        except httpx.HTTPError as exc:
            if slot is not None:
                await self._pool.release(slot, success=False)
            raise UpstreamError(f"proxy transport error: {exc}") from exc
        if slot is not None:
            await self._pool.release(slot, success=True)
        if not full_text:
            raise UpstreamError("proxy stream ended without any text content")

    # -- passthrough (native format, no event normalization) ---------------

    async def acquire_slot(self) -> tuple[AccountSlot | None, str]:
        """Acquire a pool slot for a passthrough request."""
        await self.refresh()
        if self._pool is not None:
            slot = await self._pool.acquire()
            await self._ensure_slot_token(slot)
            await self._ensure_registered(slot.session_id)
            return slot, slot.session_id
        # Single-account fallback.
        if not self._proxy_sid:
            self._proxy_sid = f"sub2api-{uuid.uuid4().hex[:12]}"
            try:
                await self._ensure_registered(self._proxy_sid)
                save_cached_session(self._proxy_sid)
            except AuthError:
                pass
        return None, self._proxy_sid

    async def release_slot(
        self, slot: AccountSlot | None, *, success: bool = True
    ) -> None:
        """Release a previously acquired slot.  No-op for single-account mode."""
        if slot is not None and self._pool is not None:
            await self._pool.release(slot, success=success)

    def proxy_headers(
        self,
        session_id: str,
        target_url: str,
        provider: str = "",
        slot: AccountSlot | None = None,
    ) -> dict[str, str]:
        """Headers to forward to ``api.adal.sylph.ai/proxy/*``.

        When ``slot`` is provided (pool mode) the per-account token is used;
        otherwise the channel-level token is used (single-account mode).

        Mirrors the header set that ``adal-backend``'s ``init_proxy_client``
        sends: Authorization, X-Session-ID, X-Target-URL, X-Provider, and
        the X-Client-* identity headers (Source/Entrypoint/OS/Version).
        Values match what the ``adal`` CLI sends to ``set_context`` so the
        proxy sees a legitimate client.
        """
        bearer = slot.token if slot is not None else self._require_token()
        headers = {
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
            "X-Session-ID": session_id,
            "X-Target-URL": target_url,
            "X-Client-Source": "cli",
            "X-Client-Entrypoint": "adal",
            "X-Client-OS": "linux",
            "X-Client-Version": "1.7.2",
        }
        if provider:
            headers["X-Provider"] = provider
        return headers

    def resolve_target(self, path: str, body: dict[str, Any]) -> str:
        """Upstream target base URL for a passthrough request."""
        return target_for_request(path, body, self._catalog)

    def resolve_provider(self, body: dict[str, Any]) -> str:
        """Return the catalog provider for the request model, or ''."""
        model = str(body.get("model") or "")
        return provider_for_model(self._catalog, model) or ""

    def rewrite_body(self, raw: bytes) -> bytes:
        """Rewrite a passthrough request body for the upstream proxy.

        Applies compatibility rewrites so CLIProxyAPI / Claude Code /
        legacy OpenAI clients can target any catalog model without
        knowing the upstream's parameter quirks:

        1. ``model``: catalog ``key`` → upstream ``model_id``
           (e.g. ``openai-gpt-5.6-terra`` → ``gpt-5.6-terra``).
        2. ``max_tokens`` → ``max_completion_tokens`` for OpenAI-native
           models that reject the legacy parameter
           (gpt-5.6-* etc.); Anthropic models keep ``max_tokens``.
        3. Claude Code special tool types (``text_editor_20250429``,
           ``bash_20250124``) → ``custom`` with an ``input_schema``.
        4. **Anthropic prompt cache**: inject ``cache_control:
           {type:"ephemeral"}`` on the system prompt's last block when
           the client didn't send any — this lets non-Anthropic-native
           clients (CLIProxyAPI, raw curl) benefit from the upstream's
           prompt caching.  Claude Code already injects its own
           ``cache_control`` so this is idempotent.
        5. **OpenAI prompt cache**: inject ``prompt_cache_key`` when
           missing on Responses API requests so the upstream's prefix
           cache is keyed consistently across retries.

        All other client fields pass through verbatim.  Returns the
        original bytes unchanged if nothing was rewritten.
        """
        if not raw:
            return raw
        try:
            body = json.loads(raw)
        except (ValueError, TypeError):
            return raw
        if not isinstance(body, dict):
            return raw
        changed = False
        # 1. model key -> upstream model_id
        model = str(body.get("model") or "")
        upstream = upstream_model_id(self._catalog, model)
        if upstream and upstream != model:
            body["model"] = upstream
            changed = True
        # 2. max_tokens -> max_completion_tokens for OpenAI-native models
        provider = provider_for_model(self._catalog, model)
        if (
            provider == "openai"
            and "max_tokens" in body
            and "max_completion_tokens" not in body
        ):
            body["max_completion_tokens"] = body.pop("max_tokens")
            changed = True
        # 3. Remap Claude Code special tool types to "custom" — the upstream
        # AdaL proxy rejects text_editor_20250429 / bash_20250124 as unsupported
        # tool types.  These are Anthropic-native server tool types that the
        # proxy doesn't implement; remapping to "custom" with the same name
        # and an input_schema lets the model still use them as function calls.
        tools = body.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                t = tool.get("type", "")
                if t in ("text_editor_20250429", "bash_20250124"):
                    tool["type"] = "custom"
                    if "input_schema" not in tool:
                        tool["input_schema"] = {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": True,
                        }
                    changed = True
        # 4. Anthropic prompt cache: inject cache_control on system prompt
        # if no cache_control exists anywhere in the system field.  This
        # maximises cache hits for clients that don't natively emit
        # cache_control (CLIProxyAPI, raw curl).  Claude Code already adds
        # its own cache_control so this is a no-op for it.
        if provider == "anthropic" or (not provider and "messages" in body):
            system = body.get("system")
            if system is not None and not _has_cache_control(system):
                if isinstance(system, str):
                    # Convert plain string to content-block array so we
                    # can attach cache_control to the last block.
                    system = [{"type": "text", "text": system}]
                    body["system"] = system
                _inject_cache_control(system)
                changed = True
        # 5. OpenAI prompt cache: inject prompt_cache_key on Responses API
        # requests that don't have one, so the upstream prefix cache keys
        # consistently across retries with the same model+input.
        if provider == "openai" and "input" in body and "prompt_cache_key" not in body:
            body["prompt_cache_key"] = "sub2api"
            changed = True
        if changed:
            return json.dumps(body).encode()
        return raw

    # -- usage / quota -----------------------------------------------------

    def _quota_accounts(self) -> list[tuple[str, str]]:
        """``(token, dead_reason)`` for every account backing this channel."""
        if self._pool is not None:
            return [(slot.token, slot.dead_reason) for slot in self._pool.slots]
        if self._token:
            return [(self._token, "")]
        return []

    async def _tiers(self, client: httpx.AsyncClient) -> dict[str, Any]:
        now = time.monotonic()
        if self._tiers_cache is not None and now < self._tiers_cache[0]:
            return self._tiers_cache[1]
        tiers = await fetch_tiers(client)
        if tiers:
            self._tiers_cache = (time.monotonic() + TIERS_CACHE_TTL, tiers)
        return tiers

    async def _collect_usage(self, client: httpx.AsyncClient) -> dict[str, Any]:
        accounts = self._quota_accounts()
        gate = asyncio.Semaphore(USAGE_FETCH_CONCURRENCY)

        async def one(token: str, dead_reason: str) -> dict[str, Any]:
            async with gate:
                row = await fetch_account_quota(client, token)
            if dead_reason:
                row["dead_reason"] = dead_reason
            return row

        tiers, *rows = await asyncio.gather(
            self._tiers(client), *(one(t, d) for t, d in accounts)
        )
        return aggregate_quota(
            rows,
            tiers,
            channel=self.name,
            pool_enabled=self._pool is not None,
        )

    async def usage(self, *, refresh: bool = False) -> dict[str, Any]:
        """Live subscription credits, shaped for a cc-switch usage script.

        Every configured account is resolved concurrently and folded into one
        flat payload (see :func:`aggregate_quota`).  Results are cached for
        ``USAGE_CACHE_TTL`` seconds so a client polling on a timer cannot turn
        into upstream request amplification; ``refresh=True`` bypasses it.
        """
        await self.start()
        deadline_ok = (
            not refresh
            and self._usage_cache is not None
            and time.monotonic() < self._usage_cache[0]
        )
        if deadline_ok:
            return self._usage_cache[1]
        async with self._usage_lock:
            if (
                not refresh
                and self._usage_cache is not None
                and time.monotonic() < self._usage_cache[0]
            ):
                return self._usage_cache[1]
            client = self._client
            if client is not None:
                payload = await self._collect_usage(client)
            else:
                async with httpx.AsyncClient() as temp:
                    payload = await self._collect_usage(temp)
            self._usage_cache = (time.monotonic() + USAGE_CACHE_TTL, payload)
            return payload

    # -- health -----------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        await self.refresh()
        base = await super().health()
        base["proxy_url"] = ADAL_PROXY_URL
        base["token_present"] = bool(self._token)
        base["registered_sessions"] = len(self._registered)
        if self._pool is not None:
            slots = self._pool.slots
            healthy = [slot for slot in slots if slot.is_healthy]
            base["pool"] = {
                "enabled": True,
                "size": self._pool.size,
                "healthy_accounts": len(healthy),
                "dead_accounts": sum(1 for s in slots if s.dead_reason),
                "available_capacity": sum(slot.available_capacity for slot in slots),
                "models_available": bool(self.models) and bool(healthy),
                "slots": self._pool.snapshot(),
            }
        else:
            base["pool"] = {"enabled": False}
        base["prompt_cache"] = {
            "auto_inject": True,
            "anthropic": "cache_control:ephemeral",
            "openai": "prompt_cache_key:sub2api",
        }
        return base
