"""Model listing, in OpenAI's ``/v1/models`` shape, plus the /v1-less alias.

``channel.models`` is read on every call and never captured: the catalog is
refreshed from upstream and tests replace it after the app is built.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import openai as oai
from ..auth import unauthorized
from ..deps import AppContext


def router(ctx: AppContext) -> APIRouter:
    api = APIRouter()
    channel = ctx.channel

    @api.get("/v1/models")
    async def list_models(request: Request):
        denial = unauthorized(request)
        if denial is not None:
            return denial
        await channel.refresh()
        created = oai.now_epoch()
        data = [
            {"id": m, "object": "model", "created": created, "owned_by": channel.name}
            for m in channel.models
        ]
        return {"object": "list", "data": data}

    @api.get("/models")
    async def models_compat(request: Request):
        """Compat alias for GET /models (without /v1 prefix)."""
        return await list_models(request)

    return api
