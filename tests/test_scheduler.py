"""Tests for the background task scheduler (``sub2api.core.scheduler``)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sub2api.core.scheduler import (
    CATALOG_REFRESH_INTERVAL,
    CREDIT_SYNC_INTERVAL,
    DEFAULT_TASKS,
    PeriodicTask,
    Scheduler,
)


class FakeChannel:
    """Records hook calls; exposes only the hooks its flags claim."""

    def __init__(self, *, with_catalog: bool = True, with_credits: bool = True):
        if with_catalog:
            self.refresh_catalog_from_upstream = self._catalog
        if with_credits:
            self.refresh_credits = self._credits
        self.catalog_calls: list[Any] = []
        self.credit_calls: list[Any] = []
        self.fail_next = False

    async def _catalog(self, store):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("upstream down")
        self.catalog_calls.append(store)

    async def _credits(self, store):
        self.credit_calls.append(store)


@pytest.mark.anyio
async def test_intervals_are_the_documented_cadence():
    assert CATALOG_REFRESH_INTERVAL == 600.0
    assert CREDIT_SYNC_INTERVAL == 600.0
    assert {t.name for t in DEFAULT_TASKS} == {"catalog_refresh", "credit_sync"}


@pytest.mark.anyio
async def test_start_runs_applicable_tasks_immediately():
    channel = FakeChannel()
    store = object()
    s = Scheduler(channel, store)
    s.start()
    try:
        await asyncio.sleep(0.05)
        assert s.running == ("catalog_refresh", "credit_sync")
        assert channel.catalog_calls == [store]
        assert channel.credit_calls == [store]
    finally:
        await s.stop()
    assert s.running == ()


@pytest.mark.anyio
async def test_skips_tasks_whose_hook_the_channel_lacks():
    channel = FakeChannel(with_catalog=False)
    s = Scheduler(channel)
    s.start()
    try:
        await asyncio.sleep(0.05)
        assert s.running == ("credit_sync",)
        assert channel.credit_calls
    finally:
        await s.stop()


@pytest.mark.anyio
async def test_failure_is_swallowed_and_next_tick_retries(monkeypatch):
    channel = FakeChannel()
    channel.fail_next = True
    tick = PeriodicTask("catalog_refresh", 0.01, DEFAULT_TASKS[0].run)
    credit = PeriodicTask("credit_sync", 3600.0, DEFAULT_TASKS[1].run)
    s = Scheduler(channel)
    s.start(tasks=(tick, credit))
    try:
        await asyncio.sleep(0.1)
        # First call raised; the second tick succeeded.
        assert len(channel.catalog_calls) >= 2
    finally:
        await s.stop()


@pytest.mark.anyio
async def test_stop_is_idempotent():
    s = Scheduler(FakeChannel())
    s.start()
    await s.stop()
    await s.stop()  # must not raise
    assert s.running == ()


@pytest.mark.anyio
async def test_start_is_idempotent():
    channel = FakeChannel()
    s = Scheduler(channel)
    s.start()
    s.start()  # second call must not double-spawn
    try:
        await asyncio.sleep(0.05)
        assert len(channel.catalog_calls) == 1
    finally:
        await s.stop()
