"""Account pool for multi-account scheduling with health tracking.

When multiple AdaL accounts are configured the pool distributes requests
across them using round-robin or least-connections scheduling.  Each slot
tracks in-flight requests, consecutive failures, and a cooldown window so
flaky accounts are temporarily skipped without manual intervention.

The pool is async-native: ``acquire`` blocks when every slot is saturated and
resumes as soon as one becomes available.  A global ``asyncio.Semaphore``
caps total concurrency at the sum of per-slot limits.

Scheduling is **cache-affinity-first**.  Upstream prompt caching on the AdaL
proxy is per account, not per session (a brand-new session id still hit
``cache_read_input_tokens``), and a cache read costs ~10x less than a fresh
input token.  So a request carrying a known cacheable prefix goes back to the
account that already holds it, and falls through to plain scheduling only
when that account is saturated, cooling, dead, or out of credit.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .store import accounts_from_db, pool_settings_from_db

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

POOL_CONFIG_PATH = Path.home() / ".adal" / "accounts.json"

DEFAULT_MAX_CONCURRENT = 4
DEFAULT_MAX_FAILURES = 3
DEFAULT_COOLDOWN_SECONDS = 60.0

# Affinity bindings expire after the longest upstream prompt-cache TTL, and the
# map is LRU-capped so a long-lived process cannot grow it without bound.
AFFINITY_TTL_SECONDS = 3600.0
AFFINITY_MAX_ENTRIES = 4096

# Cooldown doubles per consecutive arming, capped at 2**4 = 16x the base, so a
# permanently broken account stops being retried every minute forever.
MAX_COOLDOWN_DOUBLINGS = 4


@dataclass(slots=True)
class AccountConfig:
    """One account entry loaded from config."""

    token: str
    session_id: str = ""
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    cookies: list[dict[str, str]] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AccountConfig:
        return cls(
            token=str(d["token"]),
            session_id=str(d.get("session_id", "")),
            max_concurrent=int(d.get("max_concurrent", DEFAULT_MAX_CONCURRENT)),
            cookies=d.get("cookies"),
        )


@dataclass(slots=True)
class PoolConfig:
    """Pool-wide settings."""

    strategy: Literal["round-robin", "least-connections"] = "round-robin"
    max_failures: int = DEFAULT_MAX_FAILURES
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    accounts: list[AccountConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PoolConfig:
        return cls(
            strategy=d.get("strategy", "round-robin"),
            max_failures=int(d.get("max_failures", DEFAULT_MAX_FAILURES)),
            cooldown_seconds=float(d.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS)),
            accounts=[AccountConfig.from_dict(a) for a in d.get("accounts", [])],
        )


def load_pool_config(
    env_var: str = "SUB2API_ACCOUNTS",
    path: Path | None = None,
    db_path: Path | None = None,
) -> PoolConfig | None:
    """Load multi-account config from env JSON, the SQLite store, or a file.

    Resolution order:
      1. ``SUB2API_ACCOUNTS`` env var containing a JSON object
         (``{"strategy": ..., "accounts": [{"token": "..."}]}``).
      2. The SQLite store (``db_path``, else ``$SUB2API_DB``, else
         ``~/.adal/sub2api.sqlite3``) when it exists and holds at least one
         non-``dead`` account. This is what the ``--web`` UI writes, so edits
         made there take effect without touching ``accounts.json``.
      3. ``~/.adal/accounts.json`` (or ``path``) with the same shape.
      4. ``None`` when no config exists (single-account fallback).

    The DB probe is read-only and never creates the database.

    Raises ``ValueError`` on malformed JSON.
    """
    import os

    raw = os.environ.get(env_var)
    if raw:
        return PoolConfig.from_dict(json.loads(raw))

    rows = accounts_from_db(db_path)
    if rows:
        strategy, max_failures, cooldown = pool_settings_from_db(db_path)
        return PoolConfig.from_dict(
            {
                "strategy": strategy,
                "max_failures": max_failures,
                "cooldown_seconds": cooldown,
                "accounts": rows,
            }
        )

    p = path or POOL_CONFIG_PATH
    if p.exists():
        return PoolConfig.from_dict(json.loads(p.read_text(encoding="utf-8")))

    return None


# ---------------------------------------------------------------------------
# slot state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AccountSlot:
    """Runtime state for one account in the pool."""

    token: str
    session_id: str
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    pre_registered: bool = False
    cookies: list[dict[str, str]] | None = None
    # runtime state
    in_flight: int = 0
    fail_count: int = 0
    disabled_until: float = 0.0
    total_served: int = 0
    # set when the account is parked for a permanent reason (banned user,
    # dead cookies); cleared on a successful token re-mint
    dead_reason: str = ""
    # weekly_usage.is_usage_limited from the credit sync: the account still
    # authenticates but every request would be rejected for lack of credit
    usage_limited: bool = False
    # consecutive cooldown armings, for exponential backoff
    cooldown_strikes: int = 0

    @property
    def is_schedulable(self) -> bool:
        """Healthy, not permanently parked, and has spare capacity."""
        return (
            not self.dead_reason
            and not self.usage_limited
            and self.available_capacity > 0
        )

    @property
    def is_healthy(self) -> bool:
        """True when not in cooldown."""
        return time.monotonic() >= self.disabled_until

    @property
    def available_capacity(self) -> int:
        """Remaining concurrent slots (0 when saturated or unhealthy)."""
        if not self.is_healthy:
            return 0
        return max(0, self.max_concurrent - self.in_flight)


# ---------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------


class AccountPool:
    """Multi-account scheduler with health tracking and concurrency limits.

    Usage::

        pool = AccountPool(config)
        await pool.start()
        slot = await pool.acquire()      # blocks until a slot is free
        try:
            ... use slot.token, slot.session_id ...
        finally:
            await pool.release(slot, success=True)
        await pool.close()
    """

    @staticmethod
    def _slots_from_config(config: PoolConfig) -> list[AccountSlot]:
        return [
            AccountSlot(
                token=a.token,
                session_id=a.session_id or f"sub2api-pool-{i:03d}",
                max_concurrent=a.max_concurrent,
                pre_registered=bool(a.session_id),
                cookies=a.cookies,
            )
            for i, a in enumerate(config.accounts)
        ]

    def __init__(self, config: PoolConfig) -> None:
        if not config.accounts:
            raise ValueError("AccountPool requires at least one account")
        self._strategy = config.strategy
        self._max_failures = config.max_failures
        self._cooldown_seconds = config.cooldown_seconds
        self._slots = self._slots_from_config(config)
        # Created once and never replaced: swapping the semaphore while a
        # waiter is parked on the old object loses that waiter's permit
        # forever.  reconfigure adjusts the permit count instead.
        self._global_sem = asyncio.Semaphore(sum(s.max_concurrent for s in self._slots))
        self._sem_permits = sum(s.max_concurrent for s in self._slots)
        self._lock = asyncio.Lock()
        self._rr_index = 0
        self._waiting = 0
        # affinity key -> (session_id, bound_at monotonic)
        self._affinity: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._overflow_count = 0

    async def reconfigure(self, config: PoolConfig) -> bool:
        """Replace account definitions only when no request or waiter exists."""
        if not config.accounts:
            return False
        async with self._lock:
            if any(slot.in_flight for slot in self._slots) or self._waiting:
                return False
            previous = {s.session_id: s for s in self._slots}
            self._strategy = config.strategy
            self._max_failures = config.max_failures
            self._cooldown_seconds = config.cooldown_seconds
            self._slots = self._slots_from_config(config)
            # Carry over parked-account state when the account entry itself
            # didn't change (the registrar rewrites accounts.json with re-minted
            # tokens whose cookies are identical; the dead reason — e.g. a
            # banned Clerk user — still applies).  Fresh cookies mean the
            # account may have been fixed, so allow a re-probe.
            for slot in self._slots:
                old = previous.get(slot.session_id)
                if old is not None and old.dead_reason and old.cookies == slot.cookies:
                    slot.dead_reason = old.dead_reason
                    slot.disabled_until = old.disabled_until
                    slot.cooldown_strikes = old.cooldown_strikes
            # Resize the existing semaphore instead of replacing it.  Safe
            # without blocking: we only get here with no waiters and nothing
            # in flight, so every permit is free.
            total = sum(slot.max_concurrent for slot in self._slots)
            delta = total - self._sem_permits
            for _ in range(delta):
                self._global_sem.release()
            for _ in range(-delta):
                await self._global_sem.acquire()
            self._sem_permits = total
            # Account set changed: cached prefixes are no longer known to live
            # on any particular account.
            self._affinity.clear()
            self._rr_index = 0
            return True

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """No-op for now; reserved for future eager registration."""
        self._started = True

    async def close(self) -> None:
        self._started = False

    @property
    def size(self) -> int:
        return len(self._slots)

    @property
    def slots(self) -> list[AccountSlot]:
        return self._slots

    @property
    def is_idle(self) -> bool:
        """Whether no request is active or waiting for capacity."""
        return not self._waiting and all(slot.in_flight == 0 for slot in self._slots)

    async def close_if_idle(self) -> bool:
        """Stop the pool when no request is using or waiting for a slot."""
        async with self._lock:
            if self._waiting or any(slot.in_flight for slot in self._slots):
                return False
            self._started = False
            return True

    # -- core scheduling ---------------------------------------------------

    async def acquire(self, affinity_key: str | None = None) -> AccountSlot:
        """Return a schedulable slot, preferring the cache-affine account.

        ``affinity_key`` is a stable digest of the request's cacheable prefix
        (see :mod:`sub2api.compat.affinity`).  When it is already bound to a
        usable account that account is reused so the upstream prompt cache
        hits; otherwise scheduling falls through to :meth:`_pick` and the key
        is rebound to whichever account actually served the request.
        """
        async with self._lock:
            semaphore = self._global_sem
            self._waiting += 1
        waiting = True
        acquired = False
        try:
            await semaphore.acquire()
            acquired = True
            async with self._lock:
                self._waiting -= 1
                waiting = False
                slot = self._pick_affine(affinity_key) if affinity_key else self._pick()
                slot.in_flight += 1
                slot.total_served += 1
                if affinity_key:
                    self._bind_affinity(affinity_key, slot.session_id)
                return slot
        except BaseException:
            async with self._lock:
                if waiting:
                    self._waiting -= 1
            if acquired:
                semaphore.release()
            raise

    async def release(self, slot: AccountSlot, *, success: bool = True) -> None:
        """Return a slot to the pool and update health."""
        async with self._lock:
            slot.in_flight = max(0, slot.in_flight - 1)
            if success:
                slot.fail_count = 0
                slot.cooldown_strikes = 0
            else:
                slot.fail_count += 1
                if slot.fail_count >= self._max_failures:
                    # Exponential backoff: an account that keeps failing after
                    # its cooldown is probably broken, not flaky, and retrying
                    # it at the base interval forever burns every request that
                    # lands on it.  First arming uses the base cooldown.
                    slot.cooldown_strikes += 1
                    factor = 2 ** min(slot.cooldown_strikes - 1, MAX_COOLDOWN_DOUBLINGS)
                    slot.disabled_until = (
                        time.monotonic() + self._cooldown_seconds * factor
                    )
                    slot.fail_count = 0  # reset so cooldown end = clean slate
            self._global_sem.release()

    # -- affinity ----------------------------------------------------------

    @property
    def overflow_count(self) -> int:
        """Times an affinity binding existed but its account was unusable."""
        return self._overflow_count

    def _bind_affinity(self, key: str, session_id: str) -> None:
        self._affinity[key] = (session_id, time.monotonic())
        self._affinity.move_to_end(key)
        while len(self._affinity) > AFFINITY_MAX_ENTRIES:
            self._affinity.popitem(last=False)

    def _pick_affine(self, key: str) -> AccountSlot:
        """The slot bound to ``key`` when still usable, else :meth:`_pick`."""
        bound = self._affinity.get(key)
        if bound is not None:
            session_id, bound_at = bound
            if time.monotonic() - bound_at > AFFINITY_TTL_SECONDS:
                # Past the longest upstream cache TTL: the binding buys nothing.
                del self._affinity[key]
            else:
                for slot in self._slots:
                    if slot.session_id == session_id:
                        if slot.is_schedulable:
                            self._affinity.move_to_end(key)
                            return slot
                        break
                self._overflow_count += 1
        return self._pick()

    # -- selection ---------------------------------------------------------

    def _pick(self) -> AccountSlot:
        """Select the next slot according to strategy and health.

        Preference order: live accounts with credit and spare capacity, then
        live accounts out of credit with spare capacity, then whatever is
        left (fail-open, earliest recovery).  A ``dead_reason`` account is
        chosen only when every account is dead — routing traffic to a banned
        account is a guaranteed failure, so a merely-cooling account wins.
        An out-of-credit account is a last resort for the same reason,
        inverted: it authenticates fine but would be rejected for credit.
        """
        alive = [s for s in self._slots if not s.dead_reason]
        healthy = [s for s in alive if s.is_healthy]
        unlimited = [s for s in healthy if not s.usage_limited]
        candidates = (
            [s for s in unlimited if s.available_capacity > 0]
            or [s for s in healthy if s.available_capacity > 0]
            or unlimited
            or healthy
        )
        if not candidates:
            # Everything is cooling or dead: pick the earliest recovery among
            # live accounts, falling back to the full list only if all are dead.
            pool = alive or self._slots
            return min(pool, key=lambda s: s.disabled_until)

        if self._strategy == "least-connections":
            return min(candidates, key=lambda s: (s.in_flight, s.total_served))

        # round-robin (default)
        idx = self._rr_index % len(candidates)
        self._rr_index += 1
        # Skip saturated slots by advancing to the first with capacity.
        for _ in range(len(candidates)):
            slot = candidates[idx % len(candidates)]
            if slot.in_flight < slot.max_concurrent:
                return slot
            idx += 1
        # All saturated (shouldn't happen — semaphore guards) — pick first.
        return candidates[0]

    # -- introspection -----------------------------------------------------

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a JSON-serialisable view of every slot."""
        now = time.monotonic()
        return [
            {
                "session_id": s.session_id,
                "in_flight": s.in_flight,
                "max_concurrent": s.max_concurrent,
                "fail_count": s.fail_count,
                "healthy": s.is_healthy,
                "cooldown_remaining": max(0.0, s.disabled_until - now),
                "total_served": s.total_served,
                "dead_reason": s.dead_reason,
                "usage_limited": s.usage_limited,
                "cooldown_strikes": s.cooldown_strikes,
            }
            for s in self._slots
        ]
