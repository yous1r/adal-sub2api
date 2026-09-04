"""HTTP gateway: the single consumer of the normalized event stream.

The server knows nothing about any specific agent backend — it only
speaks ChatRequest/Event. All channel-specific behavior lives behind
the registry.

Assembly only: the channel, the session store, the lifespan-owned metering
store and credit-sync task live here, while every route lives in
``server/routes/*`` behind a ``router(ctx)`` factory.  ``create_channel`` stays
a module global of *this* module because tests rebind it by attribute.
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI

from .. import __version__, channels  # noqa: F401  (channels = registration)
from ..core.config import AppSettings
from ..core.registry import create_channel
from ..core.sessions import SessionStore
from ..core.store import Store
from .deps import AppContext
from .routes import admin, anthropic, chat, models, openai_compat, responses, usage

CREDIT_SYNC_INTERVAL = 600.0


async def credit_sync_loop(
    channel: Any, usage_store: Store, interval: float = CREDIT_SYNC_INTERVAL
) -> None:
    """Poll every account's credit balance forever, starting immediately.

    Runs as a lifespan task so ``usage_limited`` is known before the first
    request instead of being discovered by burning one.  Every failure is
    swallowed: a credit probe must never take the gateway down, and the next
    tick retries anyway.
    """
    while True:
        try:
            await channel.refresh_credits(usage_store)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a probe failure is not fatal
            pass
        await asyncio.sleep(interval)


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or AppSettings.from_env()
    channel = create_channel(settings.channel, settings.channel_config)
    store = SessionStore()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # The metering store is opened at startup, never in create_app:
        # constructing an app object (tests, --help) must not create a
        # database file.  A store that cannot be opened degrades to no
        # metering rather than refusing to serve.
        usage_store: Store | None = None
        credit_task: asyncio.Task | None = None
        try:
            usage_store = Store(settings.db or None)
        except (sqlite3.Error, OSError):
            usage_store = None
        app.state.usage_store = usage_store
        await channel.start()
        if usage_store is not None and hasattr(channel, "refresh_credits"):
            credit_task = asyncio.create_task(credit_sync_loop(channel, usage_store))
            app.state.credit_task = credit_task
        try:
            yield
        finally:
            if credit_task is not None:
                credit_task.cancel()
                with suppress(asyncio.CancelledError):
                    await credit_task
            await channel.close()
            if usage_store is not None:
                await usage_store.close()

    app = FastAPI(
        title="sub2api",
        description="Subscription-to-API gateway with pluggable agent channels.",
        version=__version__,
        lifespan=lifespan,
    )
    # Exposed for tests and introspection.
    app.state.settings = settings
    app.state.channel = channel
    app.state.store = store
    app.state.usage_store = None
    app.state.credit_task = None

    ctx = AppContext(settings=settings, channel=channel, store=store)
    # Registration order mirrors the pre-split app; each alias travels with the
    # handler it delegates to.  No parameterized path shadows a literal one.
    app.include_router(chat.router(ctx))
    app.include_router(openai_compat.router(ctx))
    app.include_router(models.router(ctx))
    app.include_router(anthropic.router(ctx))
    app.include_router(responses.router(ctx))
    app.include_router(usage.router(ctx))
    app.include_router(admin.router(ctx))

    if settings.web:
        # Imported lazily so a build without --web never pays for the admin
        # module, and never exposes an /admin route by accident.
        from .routes import web as web_routes

        app.include_router(web_routes.router())

    return app
