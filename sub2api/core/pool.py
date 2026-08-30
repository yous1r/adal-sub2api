"""Account pool for multi-account scheduling with health tracking.

When multiple AdaL accounts are configured the pool distributes requests
across them using round-robin or least-connections scheduling.  Each slot
tracks in-flight requests, consecutive failures, and a cooldown window so
flaky accounts are temporarily skipped without manual intervention.

The pool is async-native: ``acquire`` blocks when every slot is saturated and
resumes as soon as one becomes available.  A global ``asyncio.Semaphore``
caps total concurrency at the sum of per-slot limits.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

POOL_CONFIG_PATH = Path.home() / ".adal" / "accounts.json"

DEFAULT_MAX_CONCURRENT = 4
DEFAULT_MAX_FAILURES = 3
DEFAULT_COOLDOWN_SECONDS = 60.0


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
) -> PoolConfig | None:
    """Load multi-account config from env JSON or a file.

    Resolution order:
      1. ``SUB2API_ACCOUNTS`` env var containing a JSON object
         (``{"strategy": ..., "accounts": [{"token": "..."}]}``).
      2. ``~/.adal/accounts.json`` file with the same shape.
      3. ``None`` when no config exists (single-account fallback).

    Raises ``ValueError`` on malformed JSON.
    """
    import os

    raw = os.environ.get(env_var)
    if raw:
        return PoolConfig.from_dict(json.loads(raw))

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
        self._global_sem = asyncio.Semaphore(sum(s.max_concurrent for s in self._slots))
        self._lock = asyncio.Lock()
        self._rr_index = 0
        self._waiting = 0

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
                if (
                    old is not None
                    and old.dead_reason
                    and old.cookies == slot.cookies
                ):
                    slot.dead_reason = old.dead_reason
                    slot.disabled_until = old.disabled_until
            self._global_sem = asyncio.Semaphore(
                sum(slot.max_concurrent for slot in self._slots)
            )
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

    async def acquire(self) -> AccountSlot:
        """Return a healthy, non-saturated slot."""
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
                slot = self._pick()
                slot.in_flight += 1
                slot.total_served += 1
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
            else:
                slot.fail_count += 1
                if slot.fail_count >= self._max_failures:
                    slot.disabled_until = time.monotonic() + self._cooldown_seconds
                    slot.fail_count = 0  # reset so cooldown end = clean slate
            self._global_sem.release()

    # -- selection ---------------------------------------------------------

    def _pick(self) -> AccountSlot:
        """Select the next slot according to strategy and health."""
        healthy = [s for s in self._slots if s.is_healthy]
        if healthy:
            candidates = healthy
        else:
            # Fail-open: all in cooldown — pick earliest recovery.
            candidates = self._slots
            return min(candidates, key=lambda s: s.disabled_until)

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
            }
            for s in self._slots
        ]
