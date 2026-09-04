"""Live model catalog and tier lookups for the AdaL cloud channel.

The catalog is the single source of truth for model keys, their upstream
``model_id`` and their provider, so it sits in its own module: the channel,
the route layer and ``/v1/models`` all read it, and the tests stub
``fetch_catalog`` on the package to run offline.  ``urlopen`` is resolved
through the package namespace (``_pkg``) for the same reason — the fetch is
stubbed by rebinding ``sub2api.channels.adal_cloud.urlopen``.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request

import httpx

from .auth import ADAL_APP_URL
from .routing import PROVIDER_ROUTES, UNLISTED_PROVIDERS

_pkg = sys.modules[__package__]

ADAL_PROXY_URL = "https://api.adal.sylph.ai"


def fetch_catalog(
    proxy_url: str = ADAL_PROXY_URL, timeout: float = 20.0
) -> dict[str, Any]:
    """Fetch the live model catalog (no auth required)."""
    req = Request(f"{proxy_url}/proxy/models/catalog")
    try:
        with _pkg.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (HTTPError, OSError, ValueError):
        return {}


def catalog_models(catalog: dict[str, Any]) -> tuple[str, ...]:
    """Extract model keys from a catalog response, in catalog order."""
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(models, list):
        return ()
    keys = tuple(m["key"] for m in models if isinstance(m, dict) and m.get("key"))
    return keys


def reachable_models(catalog: dict[str, Any]) -> tuple[str, ...]:
    """Upstream model ids sub2api can actually reach through the proxy.

    Returns the upstream ``model_id`` — ``claude-sonnet-5``, not
    ``anthropic-claude-sonnet-5`` — so a client configured for the official
    vendor API needs no rewriting.  Catalog keys keep working on the way in
    (:func:`upstream_model_id` accepts both), they are simply not advertised.

    An entry is advertised only when its provider has a verified route (see
    :data:`PROVIDER_ROUTES`), it is not a local-runtime model, and its
    provider is not in :data:`UNLISTED_PROVIDERS`.  That keeps ``/v1/models``
    honest — every id resolves to a host the proxy answers 200 for — and
    keeps it unambiguous, since ``chatgpt_web`` re-exports the ``openai``
    model ids under a second key.
    """
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(models, list):
        return ()
    ids: list[str] = []
    seen: set[str] = set()
    for m in models:
        if not isinstance(m, dict) or m.get("is_local_model"):
            continue
        provider = m.get("provider")
        if provider not in PROVIDER_ROUTES or provider in UNLISTED_PROVIDERS:
            continue
        model_id = m.get("model_id") or m.get("key")
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        ids.append(model_id)
    return tuple(ids)


def provider_for_model(catalog: dict[str, Any], model: str) -> str | None:
    """Look up the provider for a model key/id in the catalog.

    A catalog ``key`` is explicit and always wins, so ``chatgpt_web-…`` still
    routes to ``chatgpt_web``.  A bare upstream ``model_id`` resolves to the
    listed provider: the ids are shared with :data:`UNLISTED_PROVIDERS`, and
    an unqualified request must land on the advertised route.
    """
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(models, list):
        return None
    fallback: str | None = None
    for m in models:
        if not isinstance(m, dict):
            continue
        if m.get("key") == model:
            return m.get("provider")
        if m.get("model_id") == model:
            provider = m.get("provider")
            if provider not in UNLISTED_PROVIDERS:
                return provider
            fallback = fallback or provider
    return fallback


def upstream_model_id(catalog: dict[str, Any], model: str) -> str:
    """Resolve a request model (key or id) to the upstream model_id."""
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if isinstance(models, list):
        for m in models:
            if isinstance(m, dict) and (
                m.get("key") == model or m.get("model_id") == model
            ):
                return m.get("model_id") or model
    return model


async def fetch_tiers(
    client: httpx.AsyncClient,
    *,
    app_url: str = ADAL_APP_URL,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Fetch the tier catalog (``{tier_id: {...}}``); ``{}`` on any failure."""
    try:
        resp = await client.get(
            f"{app_url}/api/subscription/tiers",
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        tiers = resp.json().get("tiers") if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError, AttributeError):
        return {}
    return tiers if isinstance(tiers, dict) else {}


def tier_display_name(tiers: dict[str, Any], tier: str) -> str:
    """Human label for a tier id, falling back to the id itself."""
    entry = tiers.get(tier) if isinstance(tiers, dict) else None
    if isinstance(entry, dict):
        name = entry.get("display_name")
        if isinstance(name, str) and name:
            return name
    return tier
