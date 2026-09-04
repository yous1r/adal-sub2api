"""Operational surface: channel listing, session lookup, liveness, health.

``/healthz`` is the one route in the whole app with no auth guard, so it
reports nothing but liveness; everything that could name an account or a
session sits behind :func:`sub2api.server.auth.unauthorized`.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ... import __version__
from ...core.errors import SessionNotFoundError
from ...core.registry import available_channels, get_channel_class
from ..auth import unauthorized
from ..deps import AppContext, error_response


def router(ctx: AppContext) -> APIRouter:
    api = APIRouter()
    channel = ctx.channel
    store = ctx.store

    @api.get("/v1/channels")
    async def channels(request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        configured = channel.name
        listing = []
        for name in available_channels():
            cls = get_channel_class(name)
            listing.append(
                {
                    "name": name,
                    "display_name": cls.display_name or name,
                    "models": list(cls.models),
                    "configured": name == configured,
                }
            )
        return {"channels": listing, "active": await channel.health()}

    @api.get("/v1/sessions/{session_id}")
    async def session_info(session_id: str, request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        try:
            session = store.get(session_id)
        except SessionNotFoundError as exc:
            return error_response(404, exc.code, str(exc))
        return session.to_dict()

    @api.get("/healthz")
    async def healthz():
        """Unauthenticated liveness only.

        ``channel.health()`` carries pool session ids, per-account
        ``dead_reason``s and token presence — never behind no auth.  The full
        payload lives at the authenticated ``GET /v1/health``.
        """
        return {"status": "ok", "version": __version__}

    @api.get("/v1/health")
    async def health_detail(request: Request):
        """Full channel health, including pool internals (authenticated)."""
        denial = unauthorized(request)
        if denial is not None:
            return denial
        return {
            "status": "ok",
            "version": __version__,
            "channel": await channel.health(),
        }

    return api
