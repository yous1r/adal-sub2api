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

import uuid
import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator, ClassVar
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import httpx

from ..core.channel import BaseChannel
from ..core.errors import AuthError, RuntimeMissingError, UpstreamError
from ..core.pool import AccountPool, AccountSlot, load_pool_config
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


def _jwt_exp(token: str) -> int | None:
    """Return the JWT ``exp`` (unix seconds) or ``None`` if unparseable."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return int(json.loads(__import__("base64").urlsafe_b64decode(payload))["exp"])
    except (IndexError, ValueError, KeyError, TypeError):
        return None


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

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._token = None
        self._catalog = {}
        self._registered = set()
        self._client = None
        self._pool = None
        self._proxy_sid = None

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
            self._pool = AccountPool(pool_cfg)
            await self._pool.start()
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(600.0, connect=15.0, read=600.0),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )

            # Pre-register every pool session id with its own token. These are
            # independent network calls, so fire them all at once; serial
            # registration made startup take N×15s on a large pool.
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

            await asyncio.gather(*(_register_slot(s) for s in self._pool.slots))
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
        """Acquire a pool slot for a passthrough request.

        Returns ``(slot, session_id)``.  When the pool is active the slot
        must be released via ``release_slot`` after the request completes
        (use ``success=`` to report the outcome).  When no pool is configured
        ``(None, self._proxy_sid)`` is returned and no release is needed.
        """
        if self._pool is not None:
            slot = await self._pool.acquire()
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

    # -- health -----------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        base = await super().health()
        base["proxy_url"] = ADAL_PROXY_URL
        base["token_present"] = bool(self._token)
        base["registered_sessions"] = len(self._registered)
        base["models"] = list(self.models)
        if self._pool is not None:
            base["pool"] = {
                "enabled": True,
                "size": self._pool.size,
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
