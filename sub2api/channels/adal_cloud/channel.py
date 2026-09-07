"""The AdaL cloud-proxy channel itself: auth lifecycle, pool, transport.

This module owns the stateful half of the channel — token minting per slot,
session registration, the httpx client, the normalized turn pipeline and the
passthrough surface the server's forwarding layer drives.  The stateless
tables and helpers live in the sibling modules (:mod:`.auth`, :mod:`.catalog`,
:mod:`.routing`, :mod:`.quota`, :mod:`.sse`).

Collaborators are resolved through the package namespace (``_pkg``) rather
than captured by ``from .catalog import fetch_catalog``: the test suite stubs
network access by rebinding names on ``sub2api.channels.adal_cloud`` itself
(``fetch_catalog``, ``load_pool_config``, ``mint_token_with_cookies``,
``register_session``, ``urlopen``, …), and a captured global would keep
calling the real function.  ``_pkg`` is bound from ``sys.modules`` at import
time — the package module object exists there before its ``__init__`` body
runs, so this cannot deadlock the circular import, and every attribute lookup
through it happens later, at call time, after the re-exports are in place.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from typing import Any, AsyncIterator, ClassVar

import httpx

from ...compat.sanitize import sanitize
from ...core.channel import BaseChannel
from ...core.errors import AuthError, RuntimeMissingError, UpstreamError
from ...core.pool import AccountSlot, PoolConfig
from ...core.registry import register
from ...core.types import ChatRequest, Event, TextDelta
from .auth import (
    ADAL_APP_URL,
    DEAD_COOLDOWN_S,
    DEAD_MINT_ERRORS,
    TRANSIENT_COOLDOWN_S,
)
from .catalog import ADAL_PROXY_URL
from .quota import TIERS_CACHE_TTL, USAGE_CACHE_TTL, USAGE_FETCH_CONCURRENCY
from .routing import PROVIDER_ROUTES, RequestRoute

_pkg = sys.modules[__package__]

# Logger name stays the package's, not this module's, so existing log filters
# and the ``sanitized ...`` DEBUG line keep the record name they always had.
_log = logging.getLogger(__package__)


def _effort_config(
    catalog: dict[str, Any], model: str
) -> tuple[str | None, frozenset[str] | None]:
    """Return the catalog-declared effort path and accepted values."""
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(models, list):
        return None, None
    for entry in models:
        if not isinstance(entry, dict) or model not in (
            entry.get("key"),
            entry.get("model_id"),
        ):
            continue
        options = entry.get("config_options")
        if not isinstance(options, dict):
            return None, None
        path = options.get("effort_path")
        values = options.get("effort")
        allowed = (
            frozenset(value for value in values if isinstance(value, str))
            if isinstance(values, list)
            else None
        )
        return path if isinstance(path, str) else None, allowed
    return None, None


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
    # Populated from the live catalog by _start()/refresh(); the class default
    # stays empty so a stale hardcoded list can never advertise a model the
    # proxy no longer serves.
    models: ClassVar[tuple[str, ...]] = ()

    _token: str | None
    _catalog: dict[str, Any]
    _registered: set[str]
    _client: httpx.AsyncClient | None
    _pool: Any | None
    _proxy_sid: str | None
    _usage_cache: tuple[float, dict[str, Any]] | None
    _store: Any | None
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
        self._store = None

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
        """Adopt changed pool definitions; serve models from the DB cache.

        Called on the request path (models listing, chat, passthrough), so
        it must never touch the network: the catalog lives in the SQLite
        store (see :meth:`refresh_catalog_from_upstream`, driven by the
        background task in ``server.app``).  Pool config is re-read from
        disk/DB so admin edits apply without a restart.
        """
        if not self.started:
            return
        if not self._catalog:
            # With a store: read the cached catalog only (no network on the
            # request path).  Without one: a one-time network fetch keeps the
            # original behavior for store-less deployments and tests.
            catalog = await self._store.load_catalog() if self._store is not None else {}
            if not catalog and self._store is None:
                catalog = await asyncio.to_thread(_pkg.fetch_catalog, self.proxy_url)
            if catalog:
                self._adopt_catalog(catalog)
        async with self._refresh_lock:
            config = await asyncio.to_thread(_pkg.load_pool_config)
            signature = self._pool_signature_for(config)
            if signature == self._pool_signature:
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
                self._pool = _pkg.AccountPool(config)
                await self._pool.start()
                # Leaving single-account mode: the shared client must stop
                # carrying the old account's bearer.  Pool requests set
                # Authorization per slot, and a leftover header would make
                # every account-scoped GET (quota, credits) authenticate as
                # the *previous* account -> 403 "Cannot query other users".
                if self._client is not None:
                    self._client.headers.pop("Authorization", None)
            elif not await self._pool.reconfigure(config):
                return
            self._pool_signature = signature
            active_sessions = {slot.session_id for slot in self._pool.slots}
            self._registered.intersection_update(active_sessions)
            for slot in self._pool.slots:
                if slot.pre_registered:
                    self._registered.add(slot.session_id)
                elif slot.session_id not in self._registered:
                    try:
                        await asyncio.to_thread(
                            _pkg.register_session,
                            token=slot.token,
                            session_id=slot.session_id,
                        )
                        self._registered.add(slot.session_id)
                    except AuthError:
                        pass

    def _adopt_catalog(self, catalog: dict[str, Any]) -> None:
        """Swap in a fetched catalog; empty fetches keep the current one."""
        models = _pkg.reachable_models(catalog)
        if models:
            self._catalog = catalog
            self.models = models

    async def refresh_catalog_from_upstream(
        self, store: Any = None
    ) -> tuple[str, ...]:
        """Fetch the live catalog over the network, cache it in the store.

        Runs on the background timer (``server.app.catalog_refresh_loop``),
        never on the request path.  Returns the newly advertised model ids;
        an empty tuple means the fetch failed and the previous catalog
        (memory or store) stays authoritative.
        """
        catalog = await asyncio.to_thread(_pkg.fetch_catalog, self.proxy_url)
        if not catalog:
            return ()
        self._adopt_catalog(catalog)
        target = store if store is not None else self._store
        if target is not None:
            await target.save_catalog(catalog)
        return self.models

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
            _pkg.mint_token_with_cookies, slot.token, slot.cookies or []
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
        if not slot.cookies or not _pkg.token_needs_refresh(slot.token):
            return
        async with self._slot_lock(slot.session_id):
            if _pkg.token_needs_refresh(slot.token):
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
                and not _pkg.token_needs_refresh(slot.token)
            ):
                return True  # a concurrent request already refreshed it
            return await self._mint_slot_token(slot)

    async def _refresh_single_account_token(self) -> bool:
        """Swap in a fresher token from the creds file, if one appeared."""
        tok = await asyncio.to_thread(_pkg.read_token)
        if not tok or tok == self._token or _pkg.token_needs_refresh(tok):
            return False
        self._token = tok
        if self._client is not None:
            self._client.headers["Authorization"] = f"Bearer {tok}"
        return True

    # -- lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        # Catalog: DB cache first (instant, offline-safe), network fetch only
        # when the cache is cold.  The background task in server.app keeps it
        # fresh afterwards.
        catalog: dict[str, Any] = {}
        if self._store is not None:
            catalog = await self._store.load_catalog() or {}
        if not catalog:
            catalog = await asyncio.to_thread(_pkg.fetch_catalog)
            if catalog and self._store is not None:
                await self._store.save_catalog(catalog)
        self._adopt_catalog(catalog)
        # Resolve auth token: explicit config wins, then cached creds. If the
        # cached token is expired we still try it — the proxy may accept it
        # (it keys off the session_id, not the bearer, for /proxy/*). Only run
        # the interactive device flow when no token exists at all.
        token = self.config.auth_token or _pkg.read_token()
        if not token:
            token = await asyncio.to_thread(_pkg.device_flow_login)
        self._token = token

        # Multi-account pool: when configured, the pool owns per-account tokens
        # and sessions. The shared httpx client is token-agnostic (the
        # Authorization header is set per-request from the acquired slot).
        pool_cfg = await asyncio.to_thread(_pkg.load_pool_config)
        if pool_cfg is not None and pool_cfg.accounts:
            self._pool_signature = self._pool_signature_for(pool_cfg)
            self._pool = _pkg.AccountPool(pool_cfg)
            await self._pool.start()
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(600.0, connect=15.0, read=600.0),
                proxy=self.config.proxy or None,
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
                        _pkg.register_session,
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
            proxy=self.config.proxy or None,
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
        cached_sid = _pkg.load_cached_session()
        self._proxy_sid = cached_sid or f"sub2api-{uuid.uuid4().hex[:12]}"
        try:
            await asyncio.to_thread(
                _pkg.register_session, token=token, session_id=self._proxy_sid
            )
            self._registered.add(self._proxy_sid)
            _pkg.save_cached_session(self._proxy_sid)
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
        await asyncio.to_thread(
            _pkg.register_session, token=token, session_id=session_id
        )
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
        provider = _pkg.provider_for_model(self._catalog, model)
        if provider is None and model:
            provider = _pkg.provider_from_key(model)
        route = PROVIDER_ROUTES.get(provider or "", PROVIDER_ROUTES["anthropic"])
        return (
            route.native_path,
            route.target_url,
            _pkg.upstream_model_id(self._catalog, model),
        )

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
            _pkg.anthropic_request(upstream_model, request)
            if is_anthropic
            else _pkg.openai_request(upstream_model, request)
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
                parser = (
                    _pkg.parse_anthropic_sse if is_anthropic else _pkg.parse_openai_sse
                )
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

    async def acquire_slot(
        self, affinity_key: str | None = None
    ) -> tuple[AccountSlot | None, str]:
        """Acquire a pool slot for a passthrough request.

        ``affinity_key`` pins the request to the account that already holds
        its cacheable prefix upstream (prompt caching is per account), so a
        repeated prefix bills as a cache read instead of fresh input tokens.
        """
        await self.refresh()
        if self._pool is not None:
            slot = await self._pool.acquire(affinity_key)
            await self._ensure_slot_token(slot)
            await self._ensure_registered(slot.session_id)
            return slot, slot.session_id
        # Single-account fallback.
        if not self._proxy_sid:
            self._proxy_sid = f"sub2api-{uuid.uuid4().hex[:12]}"
            try:
                await self._ensure_registered(self._proxy_sid)
                _pkg.save_cached_session(self._proxy_sid)
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
        return _pkg.target_for_request(path, body, self._catalog)

    def resolve_provider(self, body: dict[str, Any]) -> str:
        """Return the catalog provider for the request model, or ''."""
        model = str(body.get("model") or "")
        return _pkg.provider_for_model(self._catalog, model) or ""

    def resolve_route(self, path: str, body: dict[str, Any]) -> RequestRoute | None:
        """Reachable upstream route for a passthrough request, or ``None``."""
        return _pkg.route_for_request(path, body, self._catalog)

    def rewrite_body(self, raw: bytes, path: str = "/v1/messages") -> bytes:
        """Rewrite a passthrough request body for the upstream proxy.

        Applies compatibility rewrites so Claude Code / CLIProxyAPI / legacy
        OpenAI clients can target any catalog model without knowing the
        upstream's parameter quirks:

        1. ``model``: catalog ``key`` → upstream ``model_id``
           (e.g. ``openai-gpt-5.6-terra`` → ``gpt-5.6-terra``).
        2. :func:`sub2api.compat.sanitize.sanitize` strips or rewrites every
           field measured to make the upstream SDK wrapper reject the request
           (``context_management``, ``betas``, ``metadata.session_id``,
           deprecated sampling params, unsupported tool types, ``max_tokens``
           spelling per protocol, …).  Without this, Claude Code gets one
           streaming 200 followed by a wall of 500s.
        3. **Anthropic prompt cache**: inject ``cache_control:
           {type:"ephemeral"}`` on the system prompt's last block when the
           client didn't send any — this lets non-Anthropic-native clients
           (CLIProxyAPI, raw curl) benefit from the upstream's prompt caching.
           Claude Code already injects its own, so this is idempotent.
        4. **OpenAI prompt cache**: inject ``prompt_cache_key`` when missing on
           Responses API requests so the upstream prefix cache keys
           consistently across retries.

        All other client fields pass through verbatim.  Returns the original
        bytes unchanged when nothing was rewritten.
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
        upstream = _pkg.upstream_model_id(self._catalog, model)
        if upstream and upstream != model:
            body["model"] = upstream
            changed = True
        provider = _pkg.provider_for_model(self._catalog, model) or (
            _pkg.provider_from_key(model) or ""
        )
        # 2. Protocol-specific capability sanitization from the live catalog.
        effort_path, effort_options = _effort_config(self._catalog, model)
        dropped: list[str] = []
        body = sanitize(
            body,
            protocol=_pkg.protocol_for(provider, path),
            upstream_model=upstream or model,
            dropped=dropped,
            effort_path=effort_path,
            effort_options=effort_options,
        )
        if dropped:
            changed = True
            _log.debug(
                "sanitized %s request for %s: dropped %s",
                path,
                upstream or model or "?",
                ", ".join(dropped),
            )
        # 3. Anthropic prompt cache: inject cache_control on the system prompt
        # when no cache_control exists anywhere in it.  This maximises cache
        # hits for clients that don't natively emit cache_control (CLIProxyAPI,
        # raw curl); Claude Code already adds its own, so it is a no-op there.
        if provider == "anthropic" or (not provider and "messages" in body):
            system = body.get("system")
            if system is not None and not _pkg._has_cache_control(system):
                if isinstance(system, str):
                    # Convert plain string to a content-block array so
                    # cache_control can attach to the last block.
                    system = [{"type": "text", "text": system}]
                    body["system"] = system
                _pkg._inject_cache_control(system)
                changed = True
        # 4. OpenAI prompt cache: inject prompt_cache_key on Responses API
        # requests that don't have one, so the upstream prefix cache keys
        # consistently across retries with the same model+input.
        if provider == "openai" and "input" in body and "prompt_cache_key" not in body:
            body["prompt_cache_key"] = "sub2api"
            changed = True
        if changed:
            return json.dumps(body).encode()
        return raw

    # -- usage / quota -----------------------------------------------------

    def _quota_accounts(self) -> list[tuple[str, str, str]]:
        """``(token, dead_reason, session_id)`` for every channel account.

        ``session_id`` is the pool's account identity and the join key the
        admin UI needs to pair a subscription-quota row with locally metered
        spend; single-account mode falls back to the registered session id.
        """
        if self._pool is not None:
            return [
                (slot.token, slot.dead_reason, slot.session_id)
                for slot in self._pool.slots
            ]
        if self._token:
            return [(self._token, "", self._proxy_sid or "single")]
        return []

    async def _tiers(self, client: httpx.AsyncClient) -> dict[str, Any]:
        now = time.monotonic()
        if self._tiers_cache is not None and now < self._tiers_cache[0]:
            return self._tiers_cache[1]
        tiers = await _pkg.fetch_tiers(client)
        if tiers:
            self._tiers_cache = (time.monotonic() + TIERS_CACHE_TTL, tiers)
        return tiers

    async def _collect_usage(self, client: httpx.AsyncClient) -> dict[str, Any]:
        accounts = self._quota_accounts()
        gate = asyncio.Semaphore(USAGE_FETCH_CONCURRENCY)

        async def one(token: str, dead_reason: str, session_id: str) -> dict[str, Any]:
            async with gate:
                row = await _pkg.fetch_account_quota(client, token)
            if dead_reason:
                row["dead_reason"] = dead_reason
            row["session_id"] = session_id
            return row

        tiers, *rows = await asyncio.gather(
            self._tiers(client),
            *(one(t, d, sid) for t, d, sid in accounts),
        )
        return _pkg.aggregate_quota(
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
                async with httpx.AsyncClient(proxy=self.config.proxy or None) as temp:
                    payload = await self._collect_usage(temp)
            self._usage_cache = (time.monotonic() + USAGE_CACHE_TTL, payload)
            return payload

    async def refresh_credits(self, store: Any = None) -> list[dict[str, Any]]:
        """Probe every account's live credit balance; park usage-limited ones.

        ``/api/credits/balance`` is the only authoritative source for weekly
        spend limits — the account's real limits differ from the tier
        catalogue's — and the only place ``is_usage_limited`` appears.  A
        slot whose weekly budget is exhausted is marked ``usage_limited`` so
        the pool deprioritises it instead of burning a request to discover
        the same 429.

        Returns one row per probed account (``session_id`` plus the snapshot
        columns).  Failures are skipped, never raised: this runs on a timer.
        """
        client = self._client
        if client is None:
            return []
        if self._pool is not None:
            targets = [(slot.session_id, slot.token, slot) for slot in self._pool.slots]
        elif self._token:
            targets = [(self._proxy_sid or "single", self._token, None)]
        else:
            return []
        gate = asyncio.Semaphore(USAGE_FETCH_CONCURRENCY)

        async def one(
            session_id: str, token: str, slot: AccountSlot | None
        ) -> dict[str, Any] | None:
            async with gate:
                snapshot = await _pkg.fetch_credits_balance(
                    client, token, app_url=ADAL_APP_URL
                )
            if snapshot is None:
                return None
            if slot is not None:
                slot.usage_limited = bool(snapshot["is_usage_limited"])
            if store is not None:
                await store.record_credits(session_id, **snapshot)
            return {"session_id": session_id, **snapshot}

        rows = await asyncio.gather(*(one(s, t, sl) for s, t, sl in targets))
        return [row for row in rows if row is not None]

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
                "usage_limited_accounts": sum(1 for s in slots if s.usage_limited),
                "available_capacity": sum(slot.available_capacity for slot in slots),
                "models_available": bool(self.models) and bool(healthy),
                "overflow_total": self._pool.overflow_count,
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
