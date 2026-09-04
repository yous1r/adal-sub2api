"""Tests for the multi-account pool: scheduling, health, and concurrency."""

from __future__ import annotations

import asyncio
import time

import pytest

import sub2api.core.pool as pool_mod
from sub2api.core.pool import (
    AccountConfig,
    AccountPool,
    PoolConfig,
    load_pool_config,
)
from sub2api.core.store import Store


# -- helpers -----------------------------------------------------------------


def _pool(
    n: int = 3,
    *,
    strategy: str = "round-robin",
    max_concurrent: int = 2,
    max_failures: int = 3,
    cooldown: float = 60.0,
) -> AccountPool:
    cfg = PoolConfig(
        strategy=strategy,
        max_failures=max_failures,
        cooldown_seconds=cooldown,
        accounts=[
            AccountConfig(
                token=f"tok-{i}", session_id=f"sess-{i}", max_concurrent=max_concurrent
            )
            for i in range(n)
        ],
    )
    return AccountPool(cfg)


# -- config loading ----------------------------------------------------------


def test_load_pool_config_from_env(monkeypatch):
    raw = '{"strategy":"least-connections","accounts":[{"token":"a"},{"token":"b"}]}'
    monkeypatch.setenv("SUB2API_ACCOUNTS", raw)
    cfg = load_pool_config()
    assert cfg is not None
    assert cfg.strategy == "least-connections"
    assert len(cfg.accounts) == 2
    assert cfg.accounts[0].token == "a"
    assert cfg.accounts[1].token == "b"


def test_load_pool_config_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("SUB2API_ACCOUNTS", raising=False)
    cfg = load_pool_config(path=tmp_path / "nonexistent.json")
    assert cfg is None


@pytest.mark.anyio
async def test_reconfigure_replaces_accounts_when_idle():
    pool = _pool(1)
    await pool.start()
    changed = PoolConfig(accounts=[AccountConfig(token="new", session_id="new-sid")])

    assert await pool.reconfigure(changed)
    assert pool.size == 1
    assert pool.slots[0].token == "new"
    assert pool.slots[0].session_id == "new-sid"
    await pool.close()


@pytest.mark.anyio
async def test_reconfigure_defers_while_request_is_in_flight():
    pool = _pool(1)
    await pool.start()
    slot = await pool.acquire()
    changed = PoolConfig(accounts=[AccountConfig(token="new")])

    assert not await pool.reconfigure(changed)
    assert pool.slots[0] is slot
    await pool.release(slot)
    await pool.close()


def test_pool_requires_at_least_one_account():
    with pytest.raises(ValueError, match="at least one account"):
        AccountPool(PoolConfig(accounts=[]))


# -- round-robin scheduling --------------------------------------------------


@pytest.mark.anyio
async def test_round_robin_distributes_evenly():
    pool = _pool(3, max_concurrent=10)
    await pool.start()
    picked = []
    for _ in range(6):
        slot = await pool.acquire()
        picked.append(slot.session_id)
        await pool.release(slot)
    assert picked == ["sess-0", "sess-1", "sess-2", "sess-0", "sess-1", "sess-2"]
    await pool.close()


@pytest.mark.anyio
async def test_least_connections_picks_idle_slot():
    pool = _pool(3, strategy="least-connections", max_concurrent=5)
    await pool.start()
    # Acquire two slots without releasing — they're in-flight.
    s0 = await pool.acquire()
    s1 = await pool.acquire()
    # The third slot should be the least-connections pick.
    s2 = await pool.acquire()
    assert s2 is not s0 and s2 is not s1
    assert s2.in_flight == 1
    assert s0.in_flight == 1 and s1.in_flight == 1
    await pool.close()


# -- health tracking ---------------------------------------------------------


@pytest.mark.anyio
async def test_failures_trigger_cooldown():
    pool = _pool(1, max_concurrent=5, max_failures=2, cooldown=1.0)
    await pool.start()
    slot = pool.slots[0]
    # Two failures → cooldown.
    await pool.release(slot, success=False)
    assert slot.fail_count == 1
    await pool.release(slot, success=False)
    assert slot.fail_count == 0  # reset after cooldown trigger
    assert not slot.is_healthy
    assert slot.disabled_until > time.monotonic()
    await pool.close()


@pytest.mark.anyio
async def test_success_resets_fail_count():
    pool = _pool(1, max_concurrent=5, max_failures=3, cooldown=60.0)
    await pool.start()
    slot = pool.slots[0]
    await pool.release(slot, success=False)
    await pool.release(slot, success=False)
    assert slot.fail_count == 2
    await pool.release(slot, success=True)
    assert slot.fail_count == 0
    await pool.close()


@pytest.mark.anyio
async def test_cooldown_recovery_after_timeout():
    pool = _pool(1, max_concurrent=5, max_failures=1, cooldown=0.05)
    await pool.start()
    slot = pool.slots[0]
    await pool.release(slot, success=False)
    assert not slot.is_healthy
    await asyncio.sleep(0.06)
    assert slot.is_healthy
    await pool.close()


@pytest.mark.anyio
async def test_all_in_cooldown_fail_open():
    pool = _pool(2, max_concurrent=1, max_failures=1, cooldown=60.0)
    await pool.start()
    # Force both into cooldown.
    for s in pool.slots:
        s.disabled_until = time.monotonic() + 60.0
    # Acquire should still return the earliest-recovery slot.
    slot = await pool.acquire()
    assert slot is not None
    await pool.release(slot)
    await pool.close()


# -- concurrency -------------------------------------------------------------


@pytest.mark.anyio
async def test_acquire_blocks_at_capacity():
    pool = _pool(1, max_concurrent=2)
    await pool.start()
    # Saturate the single slot.
    s1 = await pool.acquire()
    s2 = await pool.acquire()
    # Third acquire should block.
    started = asyncio.Event()
    done = asyncio.Event()

    async def try_acquire():
        started.set()
        s3 = await pool.acquire()
        done.set()
        await pool.release(s3)

    task = asyncio.create_task(try_acquire())
    await started.wait()
    assert not done.is_set()  # blocked
    # Release one → task should complete.
    await pool.release(s1)
    await asyncio.wait_for(task, timeout=2.0)
    assert done.is_set()
    await pool.release(s2)
    await pool.close()


@pytest.mark.anyio
async def test_concurrent_acquires_distribute_across_slots():
    pool = _pool(4, max_concurrent=1)
    await pool.start()

    async def acquire_release():
        s = await pool.acquire()
        await asyncio.sleep(0.01)
        await pool.release(s)
        return s.session_id

    results = await asyncio.gather(*[acquire_release() for _ in range(8)])
    # Each of the 4 slots should have served exactly 2 requests.
    counts: dict[str, int] = {}
    for sid in results:
        counts[sid] = counts.get(sid, 0) + 1
    assert len(counts) == 4
    assert all(c == 2 for c in counts.values())
    await pool.close()


# -- snapshot ----------------------------------------------------------------


@pytest.mark.anyio
async def test_snapshot_reflects_state():
    pool = _pool(2, max_concurrent=3)
    await pool.start()
    s = await pool.acquire()
    snap = pool.snapshot()
    assert len(snap) == 2
    assert snap[0]["in_flight"] == 1
    assert snap[0]["healthy"] is True
    assert snap[0]["max_concurrent"] == 3
    await pool.release(s)
    await pool.close()


@pytest.mark.anyio
async def test_snapshot_includes_dead_reason():
    pool = _pool(1)
    await pool.start()
    pool.slots[0].dead_reason = "user_banned"
    snap = pool.snapshot()
    assert snap[0]["dead_reason"] == "user_banned"
    await pool.close()


@pytest.mark.anyio
async def test_reconfigure_preserves_dead_state_for_unchanged_accounts():
    """Re-minted tokens (same cookies) keep the parked state; fresh cookies
    (account possibly fixed) allow a re-probe."""
    pool = AccountPool(
        PoolConfig(
            accounts=[
                AccountConfig(
                    token="tok-a",
                    session_id="sess-a",
                    cookies=[{"name": "__client", "value": "c1"}],
                ),
                AccountConfig(
                    token="tok-b",
                    session_id="sess-b",
                    cookies=[{"name": "__client", "value": "c2"}],
                ),
            ]
        )
    )
    await pool.start()
    pool.slots[0].dead_reason = "user_banned"
    pool.slots[0].disabled_until = time.monotonic() + 3600
    pool.slots[1].dead_reason = "signed_out"
    pool.slots[1].disabled_until = time.monotonic() + 3600

    ok = await pool.reconfigure(
        PoolConfig(
            accounts=[
                # same cookies as before -> dead state carried over
                AccountConfig(
                    token="tok-a2",
                    session_id="sess-a",
                    cookies=[{"name": "__client", "value": "c1"}],
                ),
                # registrar attached fresh cookies -> allow a re-probe
                AccountConfig(
                    token="tok-b2",
                    session_id="sess-b",
                    cookies=[{"name": "__client", "value": "c2-new"}],
                ),
            ]
        )
    )
    assert ok is True
    a, b = pool.slots
    assert a.dead_reason == "user_banned"
    assert a.disabled_until > time.monotonic()
    assert b.dead_reason == ""
    assert b.disabled_until == 0.0
    await pool.close()


# -- cache affinity ----------------------------------------------------------


@pytest.mark.anyio
async def test_affinity_reuses_the_same_slot_for_the_same_key():
    """Upstream prompt caching is per account, so a known prefix must go back
    to the account that already holds it."""
    pool = _pool(3, max_concurrent=2)
    await pool.start()
    first = await pool.acquire("key-a")
    await pool.release(first)
    for _ in range(5):
        again = await pool.acquire("key-a")
        assert again is first
        await pool.release(again)
    assert pool.overflow_count == 0
    await pool.close()


@pytest.mark.anyio
async def test_affinity_keys_spread_across_accounts():
    pool = _pool(3, max_concurrent=2)
    await pool.start()
    a = await pool.acquire("key-a")
    b = await pool.acquire("key-b")
    assert a is not b
    await pool.release(a)
    await pool.release(b)
    assert await pool.acquire("key-b") is b
    await pool.close()


@pytest.mark.anyio
async def test_affinity_overflows_when_bound_slot_is_saturated():
    pool = _pool(2, max_concurrent=1)
    await pool.start()
    bound = await pool.acquire("key-a")  # binds and saturates it
    other = await pool.acquire("key-a")  # same key, no capacity left
    assert other is not bound
    assert pool.overflow_count == 1
    await pool.release(bound)
    await pool.release(other)
    await pool.close()


@pytest.mark.anyio
async def test_affinity_overflows_away_from_a_dead_account():
    pool = _pool(2, max_concurrent=2)
    await pool.start()
    bound = await pool.acquire("key-a")
    await pool.release(bound)
    bound.dead_reason = "user_banned"
    moved = await pool.acquire("key-a")
    assert moved is not bound
    assert pool.overflow_count == 1
    # The key is rebound to the account that actually served it.
    await pool.release(moved)
    assert await pool.acquire("key-a") is moved
    assert pool.overflow_count == 1
    await pool.close()


@pytest.mark.anyio
async def test_affinity_overflows_away_from_a_usage_limited_account():
    pool = _pool(2, max_concurrent=2)
    await pool.start()
    bound = await pool.acquire("key-a")
    await pool.release(bound)
    bound.usage_limited = True
    moved = await pool.acquire("key-a")
    assert moved is not bound
    assert pool.overflow_count == 1
    await pool.release(moved)
    await pool.close()


@pytest.mark.anyio
async def test_affinity_binding_expires_after_the_cache_ttl():
    pool = _pool(2, max_concurrent=2)
    await pool.start()
    bound = await pool.acquire("key-a")
    await pool.release(bound)
    # Age the binding past the longest upstream cache TTL.
    session_id, bound_at = pool._affinity["key-a"]
    pool._affinity["key-a"] = (session_id, bound_at - pool_mod.AFFINITY_TTL_SECONDS - 1)
    await pool.release(await pool.acquire("key-a"))
    # Expiry is not an overflow: nothing was wrong with the account.
    assert pool.overflow_count == 0
    await pool.close()


@pytest.mark.anyio
async def test_affinity_map_is_lru_capped():
    pool = _pool(1, max_concurrent=1)
    await pool.start()
    for i in range(pool_mod.AFFINITY_MAX_ENTRIES + 10):
        await pool.release(await pool.acquire(f"key-{i}"))
    assert len(pool._affinity) == pool_mod.AFFINITY_MAX_ENTRIES
    assert "key-0" not in pool._affinity
    await pool.close()


@pytest.mark.anyio
async def test_reconfigure_clears_affinity():
    pool = _pool(2)
    await pool.start()
    await pool.release(await pool.acquire("key-a"))
    assert await pool.reconfigure(
        PoolConfig(accounts=[AccountConfig(token="new", session_id="new-sid")])
    )
    assert pool._affinity == {}
    await pool.close()


# -- selection safety --------------------------------------------------------


@pytest.mark.anyio
async def test_dead_account_is_never_preferred_over_a_cooling_one():
    """Routing to a banned account is a guaranteed failure; a cooling account
    at least might work."""
    pool = _pool(2, max_concurrent=1)
    await pool.start()
    dead, cooling = pool.slots
    dead.dead_reason = "user_banned"
    cooling.disabled_until = time.monotonic() + 60.0
    slot = await pool.acquire()
    assert slot is cooling
    await pool.release(slot)
    await pool.close()


@pytest.mark.anyio
async def test_usage_limited_slot_is_last_resort():
    pool = _pool(2, max_concurrent=4)
    await pool.start()
    limited, ok = pool.slots
    limited.usage_limited = True
    for _ in range(4):
        assert await pool.acquire() is ok
    # Only the out-of-credit account has capacity left now.
    assert await pool.acquire() is limited
    await pool.close()


@pytest.mark.anyio
async def test_all_dead_still_fails_open():
    pool = _pool(2, max_concurrent=1)
    await pool.start()
    for i, s in enumerate(pool.slots):
        s.dead_reason = "user_banned"
        s.disabled_until = time.monotonic() + 60.0 * (2 - i)
    slot = await pool.acquire()
    assert slot is pool.slots[1]  # earliest recovery
    await pool.release(slot)
    await pool.close()


# -- cooldown escalation -----------------------------------------------------


@pytest.mark.anyio
async def test_cooldown_doubles_on_each_consecutive_arming():
    pool = _pool(1, max_concurrent=5, max_failures=1, cooldown=10.0)
    await pool.start()
    slot = pool.slots[0]

    await pool.release(slot, success=False)
    first = slot.disabled_until - time.monotonic()
    assert slot.cooldown_strikes == 1
    assert 9.0 < first <= 10.0  # first arming is the base cooldown

    await pool.release(slot, success=False)
    second = slot.disabled_until - time.monotonic()
    assert slot.cooldown_strikes == 2
    assert 19.0 < second <= 20.0

    await pool.release(slot, success=False)
    third = slot.disabled_until - time.monotonic()
    assert slot.cooldown_strikes == 3
    assert 39.0 < third <= 40.0
    await pool.close()


@pytest.mark.anyio
async def test_cooldown_backoff_is_capped():
    pool = _pool(1, max_concurrent=5, max_failures=1, cooldown=1.0)
    await pool.start()
    slot = pool.slots[0]
    for _ in range(8):
        await pool.release(slot, success=False)
    assert slot.cooldown_strikes == 8
    capped = slot.disabled_until - time.monotonic()
    assert 15.0 < capped <= 16.0  # 2 ** MAX_COOLDOWN_DOUBLINGS
    await pool.close()


@pytest.mark.anyio
async def test_success_resets_cooldown_escalation():
    pool = _pool(1, max_concurrent=5, max_failures=1, cooldown=10.0)
    await pool.start()
    slot = pool.slots[0]
    await pool.release(slot, success=False)
    await pool.release(slot, success=False)
    assert slot.cooldown_strikes == 2
    await pool.release(slot, success=True)
    assert slot.cooldown_strikes == 0
    slot.disabled_until = 0.0
    await pool.release(slot, success=False)
    assert 9.0 < slot.disabled_until - time.monotonic() <= 10.0
    await pool.close()


# -- semaphore resizing ------------------------------------------------------


@pytest.mark.anyio
async def test_reconfigure_resizes_capacity_without_replacing_the_semaphore():
    """Replacing the semaphore object would orphan any waiter parked on the
    old one; resizing keeps a single object for the pool's lifetime."""
    pool = _pool(1, max_concurrent=1)
    await pool.start()
    original = pool._global_sem

    assert await pool.reconfigure(
        PoolConfig(
            accounts=[AccountConfig(token="t", session_id="s", max_concurrent=3)]
        )
    )
    assert pool._global_sem is original
    assert pool._sem_permits == 3
    slots = [await pool.acquire() for _ in range(3)]
    assert all(s is pool.slots[0] for s in slots)
    for s in slots:
        await pool.release(s)

    assert await pool.reconfigure(
        PoolConfig(
            accounts=[AccountConfig(token="t", session_id="s", max_concurrent=1)]
        )
    )
    assert pool._global_sem is original
    assert pool._sem_permits == 1
    held = await pool.acquire()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(pool.acquire(), timeout=0.05)
    await pool.release(held)
    await pool.close()


# -- DB-backed config --------------------------------------------------------


@pytest.mark.anyio
async def test_load_pool_config_prefers_db_over_accounts_json(monkeypatch, tmp_path):
    monkeypatch.delenv("SUB2API_ACCOUNTS", raising=False)
    db = tmp_path / "sub2api.sqlite3"
    store = Store(db)
    await store.import_accounts(
        {
            "strategy": "least-connections",
            "max_failures": 7,
            "cooldown_seconds": 12.5,
            "accounts": [
                {"session_id": "db-a", "token": "tok-db-a", "max_concurrent": 2},
                {"session_id": "db-b", "token": "tok-db-b", "max_concurrent": 3},
            ],
        }
    )
    await store.close()

    json_file = tmp_path / "accounts.json"
    json_file.write_text(
        '{"accounts":[{"token":"tok-file","session_id":"file-a"}]}', encoding="utf-8"
    )

    cfg = load_pool_config(path=json_file, db_path=db)
    assert cfg is not None
    assert [a.session_id for a in cfg.accounts] == ["db-a", "db-b"]
    assert cfg.strategy == "least-connections"
    assert cfg.max_failures == 7
    assert cfg.cooldown_seconds == 12.5


@pytest.mark.anyio
async def test_load_pool_config_skips_dead_db_rows(monkeypatch, tmp_path):
    monkeypatch.delenv("SUB2API_ACCOUNTS", raising=False)
    db = tmp_path / "sub2api.sqlite3"
    store = Store(db)
    await store.upsert_account("live", "tok-live", status="ok")
    await store.upsert_account("gone", "tok-gone", status="dead")
    await store.close()

    cfg = load_pool_config(path=tmp_path / "missing.json", db_path=db)
    assert cfg is not None
    assert [a.session_id for a in cfg.accounts] == ["live"]


def test_load_pool_config_env_still_wins_over_db(monkeypatch, tmp_path):
    db = tmp_path / "sub2api.sqlite3"
    Store(db)
    monkeypatch.setenv("SUB2API_ACCOUNTS", '{"accounts":[{"token":"env-tok"}]}')
    cfg = load_pool_config(db_path=db)
    assert cfg is not None
    assert cfg.accounts[0].token == "env-tok"


def test_load_pool_config_falls_back_to_file_when_db_is_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("SUB2API_ACCOUNTS", raising=False)
    db = tmp_path / "sub2api.sqlite3"
    Store(db)  # schema only, no accounts
    json_file = tmp_path / "accounts.json"
    json_file.write_text(
        '{"accounts":[{"token":"tok-file","session_id":"file-a"}]}', encoding="utf-8"
    )
    cfg = load_pool_config(path=json_file, db_path=db)
    assert cfg is not None
    assert cfg.accounts[0].token == "tok-file"
