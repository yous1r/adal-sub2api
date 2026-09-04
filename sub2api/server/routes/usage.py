"""Subscription usage: the channel's quota payload plus locally metered totals.

Flat, extractor-friendly quota payload for cc-switch's "用量查询"
custom-script hook: the top-level keys are exactly the fields its
extractor reads (isValid / planName / used / total / remaining / unit /
extra), with sub2api's own per-account detail nested under `sub2api`.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import openai as oai
from ..auth import unauthorized
from ..deps import AppContext

USAGE_WINDOW_SECONDS = 86400.0


async def local_usage_detail(request: Request) -> dict[str, Any]:
    """Locally metered totals for the trailing 24 h, or ``{}``.

    Empty when no store is open (metering off) so the cc-switch payload
    never grows keys that would read as "0 spent" when the truth is
    "not measured".  ``by_session`` repeats the same window per pool
    account (keyed by ``session_id``) so the admin UI can bill each
    account for its own traffic; it is absent with the rest when
    metering is off.
    """
    usage_store = request.app.state.usage_store
    if usage_store is None:
        return {}
    totals = await usage_store.usage_totals(time.time() - USAGE_WINDOW_SECONDS)
    by_session = await usage_store.usage_by_session(time.time() - USAGE_WINDOW_SECONDS)
    return {
        "requests": totals["requests"],
        "tokens": totals["tokens"],
        "cost_usd": totals["cost_usd"],
        "window": "24h",
        "rate_source": "calibrated",
        "by_session": by_session,
    }


def router(ctx: AppContext) -> APIRouter:
    api = APIRouter()
    channel = ctx.channel

    @api.get("/v1/usage")
    async def usage(request: Request, refresh: bool = False):
        """Live subscription credits for the active channel.

        ``?refresh=1`` bypasses the channel's short-lived cache.  Channels
        with no subscription to report answer 501 ``channel_not_supported``.
        """
        denial = unauthorized(request)
        if denial is not None:
            return denial
        fetch = getattr(channel, "usage", None)
        if fetch is None:
            return JSONResponse(
                status_code=501,
                content=oai.openai_error(
                    f"channel `{channel.name}` reports no subscription usage",
                    err_type="api_error",
                    code="channel_not_supported",
                ),
            )
        payload = await fetch(refresh=refresh)
        local = await local_usage_detail(request)
        if not local:
            return payload
        # Copy before merging: the channel caches this payload for
        # USAGE_CACHE_TTL and must not accumulate per-request detail.
        merged = dict(payload)
        detail = dict(merged.get("sub2api") or {})
        detail.update(local)
        merged["sub2api"] = detail
        return merged

    @api.get("/v1/v1/usage")
    async def usage_v1v1(request: Request, refresh: bool = False):
        """Compat alias for clients whose base-url includes /v1 twice."""
        return await usage(request, refresh)

    @api.get("/usage")
    async def usage_compat(request: Request, refresh: bool = False):
        """Compat alias for GET /usage (without /v1 prefix)."""
        return await usage(request, refresh)

    return api
