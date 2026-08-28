"""Tests for the multi-account pool: scheduling, health, and concurrency."""

from __future__ import annotations

import asyncio
import time

import pytest

from sub2api.core.pool import (
    AccountConfig,
    AccountPool,
    PoolConfig,
    load_pool_config,
)


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
