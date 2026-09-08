"""HTTP gateway: the single consumer of the normalized event stream.

The server knows nothing about any specific agent backend — it only
speaks ChatRequest/Event. All channel-specific behavior lives behind
the registry.

Assembly only: the channel, the session store, the lifespan-owned metering
store and the background :class:`~sub2api.core.scheduler.Scheduler` (periodic
model-catalog refresh, credit/subscription sync) live here, while every route
lives in ``server/routes/*`` behind a ``router(ctx)`` factory.
``create_channel`` stays a module global of *this* module because tests
rebind it by attribute.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .. import __version__, channels  # noqa: F401  (channels = registration)
from ..core.config import AppSettings
from ..core.registry import create_channel
from ..core.scheduler import Scheduler
from ..core.sessions import SessionStore
from ..core.store import Store
from .deps import AppContext
from .routes import admin, anthropic, chat, models, openai_compat, responses, usage


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
        try:
            usage_store = Store(settings.db or None)
        except (sqlite3.Error, OSError):
            usage_store = None
        app.state.usage_store = usage_store
        # The channel reads the catalog cache (and the scheduler writes it)
        # through this store; channels without catalog support ignore it.
        if usage_store is not None and hasattr(channel, "_store"):
            channel._store = usage_store
        await channel.start()
        scheduler = Scheduler(channel, usage_store)
        scheduler.start()
        app.state.scheduler = scheduler
        try:
            yield
        finally:
            await scheduler.stop()
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
    app.state.scheduler = None

    ctx = AppContext(settings=settings, channel=channel, store=store)
    # Registration order mirrors the pre-split app; each alias travels with the
    # handler it delegates to.  No parameterized path shadows a literal one.
    # Responses-only keeps Chat compatibility through the Responses bridge;
    # normalized Chat and Anthropic routes remain unregistered.
    if not settings.responses_only:
        app.include_router(chat.router(ctx))
    app.include_router(openai_compat.router(ctx))
    app.include_router(models.router(ctx))
    if not settings.responses_only:
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
