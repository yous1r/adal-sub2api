"""Background task scheduler: the single home for periodic maintenance.

Every periodic task — model-catalog refresh, per-account credit and
subscription-status sync, and anything that follows — is registered here as
a :class:`PeriodicTask` and driven by one asyncio loop per task, started
from the app lifespan.  Central ownership buys three invariants:

1. **One owner per concern.**  The request path never performs network
   maintenance (catalog reads come from the SQLite cache); the scheduler is
   the sole writer of cached upstream state.
2. **Uniform failure semantics.**  A task failure is logged and swallowed —
   the last good state stays authoritative until the next tick.  A
   maintenance task must never take the gateway down.
3. **Uniform lifecycle.**  Tasks are cancelled and awaited in the lifespan
   teardown; nothing leaks past server shutdown.

Channels opt in per task through :data:`_HOOK_FOR_TASK`: a task whose hook
the channel lacks is skipped, so channels without a catalog or subscription
surface degrade cleanly instead of erroring every tick.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

_log = logging.getLogger(__name__)

#: Model-catalog refresh cadence (seconds).
CATALOG_REFRESH_INTERVAL = 600.0
#: Per-account credit / subscription-state sync cadence (seconds).
CREDIT_SYNC_INTERVAL = 600.0


@dataclass(frozen=True, slots=True)
class PeriodicTask:
    """One periodic maintenance job: name, cadence, and async body.

    The body receives ``(channel, store)`` — ``store`` may be ``None`` when
    the metering database is unavailable — and must tolerate being cancelled
    at any await point.
    """

    name: str
    interval: float
    run: Callable[[Any, Any], Awaitable[Any]]


async def _refresh_catalog(channel: Any, store: Any) -> None:
    """Re-fetch the upstream model catalog and cache it in the store.

    The request path reads the cached catalog only, so a slow or dead
    upstream never delays a request; this task is the sole cache writer.
    """
    await channel.refresh_catalog_from_upstream(store)


async def _sync_credits(channel: Any, store: Any) -> None:
    """Probe every account's credits and subscription state.

    ``refresh_credits`` snapshots the live balances into the store (driving
    the admin UI) and parks usage-limited slots; subscription ``tier`` /
    ``status`` ride along in the same payload.
    """
    await channel.refresh_credits(store)


#: The standard task set, in registration order.
#:
#: Intervals mirror the historical ad-hoc loops (``server.app``): long enough
#: to stay well under upstream rate limits, short enough that a newly
#: published model or a flipped subscription status shows up without a
#: restart.
DEFAULT_TASKS: tuple[PeriodicTask, ...] = (
    PeriodicTask("catalog_refresh", CATALOG_REFRESH_INTERVAL, _refresh_catalog),
    PeriodicTask("credit_sync", CREDIT_SYNC_INTERVAL, _sync_credits),
)

#: Channel hook each built-in task drives.  A task whose hook the channel
#: lacks is skipped at start.
_HOOK_FOR_TASK = {
    "catalog_refresh": "refresh_catalog_from_upstream",
    "credit_sync": "refresh_credits",
}


async def _run_task(task: PeriodicTask, channel: Any, store: Any) -> None:
    """Run *task* forever, starting immediately, swallowing failures."""
    while True:
        try:
            await task.run(channel, store)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a maintenance failure is not fatal
            _log.exception("periodic task %s failed; retrying next tick", task.name)
        await asyncio.sleep(task.interval)


class Scheduler:
    """Owns the lifespan-scoped periodic tasks for one channel."""

    def __init__(self, channel: Any, store: Any = None) -> None:
        self._channel = channel
        self._store = store
        self._tasks: dict[str, asyncio.Task] = {}

    def start(self, tasks: tuple[PeriodicTask, ...] | None = None) -> None:
        """Spawn every applicable task; idempotent (running tasks are kept).

        Tasks whose channel hook is missing are skipped, so the same task
        set serves every channel.  Must be called from a running loop.
        """
        if tasks is None:
            tasks = DEFAULT_TASKS
        for task in tasks:
            if task.name in self._tasks:
                continue
            hook = _HOOK_FOR_TASK.get(task.name)
            if hook is not None and not hasattr(self._channel, hook):
                continue
            self._tasks[task.name] = asyncio.create_task(
                _run_task(task, self._channel, self._store),
                name=f"periodic:{task.name}",
            )

    async def stop(self) -> None:
        """Cancel and await every task; idempotent."""
        for name, t in self._tasks.items():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - teardown must not raise
                _log.exception("periodic task %s raised during stop", name)
        self._tasks.clear()

    @property
    def running(self) -> tuple[str, ...]:
        """Names of the tasks currently running."""
        return tuple(sorted(self._tasks))
