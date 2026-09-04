"""Upstream route table for the AdaL cloud proxy.

Kept apart from the catalog and the channel because these tables are pure
measured facts about the proxy — which sub-path each provider answers and at
which ``X-Target-URL`` — with no I/O and no channel state.  Only
:func:`route_for_request` and :func:`target_for_request` consult the live
catalog, and they reach it through the package namespace so a test that
rebinds ``sub2api.channels.adal_cloud.provider_for_model`` (or any other
catalog helper) is honoured by the code that calls it; that also keeps this
module free of a module-level import cycle with :mod:`.catalog`, which needs
:data:`PROVIDER_ROUTES`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

_pkg = sys.modules[__package__]


# -- upstream routing --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderRoute:
    """One provider's reachable surface on the AdaL proxy."""

    target_url: str
    paths: frozenset[str]
    """Proxy sub-paths this provider answers; anything else is not served."""
    native_path: str
    """Sub-path used by the normalized ``/v1/chat`` pipeline."""


# provider (catalog id) -> reachable proxy routes.  Every row below was
# verified HTTP 200 with real content against ``api.adal.sylph.ai``: the proxy
# is a thin, path-bound SDK wrapper that performs *no* cross-protocol
# translation, so a provider is reachable only on the sub-paths listed here
# and only at that exact ``X-Target-URL`` (bare host or vendor-specific
# prefix; the proxy appends the ``/v1/...`` suffix from the inbound path).
#
# ``google`` is deliberately absent: its OpenAI-compat shim answers 403
# ``model_not_allowed`` for every permutation and only the native
# ``generativelanguage.googleapis.com`` paths work, which needs a translation
# layer sub2api does not have yet.
PROVIDER_ROUTES: dict[str, ProviderRoute] = {
    "anthropic": ProviderRoute(
        "https://api.anthropic.com",
        frozenset({"/v1/messages"}),
        "/v1/messages",
    ),
    "openai": ProviderRoute(
        "https://api.openai.com",
        frozenset({"/v1/chat/completions", "/v1/responses"}),
        "/v1/chat/completions",
    ),
    "chatgpt_web": ProviderRoute(
        "https://api.openai.com",
        frozenset({"/v1/chat/completions", "/v1/responses"}),
        "/v1/chat/completions",
    ),
    "zai": ProviderRoute(
        "https://api.z.ai/api/anthropic",
        frozenset({"/v1/messages"}),
        "/v1/messages",
    ),
    "deepseek": ProviderRoute(
        "https://api.deepseek.com",
        frozenset({"/v1/chat/completions", "/v1/responses"}),
        "/v1/chat/completions",
    ),
    "kimi": ProviderRoute(
        "https://api.moonshot.ai",
        frozenset({"/v1/chat/completions", "/v1/responses"}),
        "/v1/chat/completions",
    ),
    "minimax": ProviderRoute(
        "https://api.minimax.io/anthropic",
        frozenset({"/v1/messages"}),
        "/v1/messages",
    ),
    "xai": ProviderRoute(
        "https://api.x.ai",
        frozenset({"/v1/messages", "/v1/chat/completions", "/v1/responses"}),
        "/v1/chat/completions",
    ),
    "qwen": ProviderRoute(
        "https://dashscope-intl.aliyuncs.com/compatible-mode",
        frozenset({"/v1/chat/completions", "/v1/responses"}),
        "/v1/chat/completions",
    ),
    "meta": ProviderRoute(
        "https://api.meta.ai",
        frozenset({"/v1/chat/completions", "/v1/responses"}),
        "/v1/chat/completions",
    ),
}

# Providers whose catalog entries are never advertised on ``/v1/models``.
# ``chatgpt_web`` re-exports the three ``openai`` models under a second key
# (measured: ``chatgpt_web-gpt-5.6-terra`` and ``openai-gpt-5.6-terra`` share
# ``model_id: gpt-5.6-terra``).  Publishing upstream ids means those pairs
# would collide, and the ChatGPT-web route is the worse half: same model, same
# host, but billed through a scraped web session.  The route row stays so an
# explicit ``chatgpt_web-`` key still works for anyone who wants it.
UNLISTED_PROVIDERS: frozenset[str] = frozenset({"chatgpt_web"})

# provider (catalog id) -> upstream target base URL, for passthrough routing.
PROVIDER_BASE_URLS: dict[str, str] = {
    provider: route.target_url for provider, route in PROVIDER_ROUTES.items()
}


def provider_from_key(model: str) -> str | None:
    """Longest provider prefix of a catalog model *key*, or ``None``.

    Used only when the catalog has no entry for the requested model (stale
    cache, offline start).  Matching whole ``{provider}-`` prefixes against
    :data:`PROVIDER_ROUTES` — rather than a naive ``model.split("-", 1)[0]`` —
    is what makes ``None`` reachable: now that ``/v1/models`` advertises bare
    upstream ids, a splitting version would turn ``gpt-5.6-luna`` into
    ``"gpt"``, a provider that exists in no table, instead of admitting it
    cannot tell.  ``None`` lets :func:`route_for_request` fall back to the
    endpoint's native provider, which is right for a bare id.  Longest-match
    keeps that sound if a future provider id ever prefixes another.
    """
    best: str | None = None
    for provider in PROVIDER_ROUTES:
        if model.startswith(f"{provider}-") and (
            best is None or len(provider) > len(best)
        ):
            best = provider
    return best


def proxy_path_for(provider: str, requested_path: str) -> str | None:
    """``requested_path`` when *provider* serves it there, else ``None``.

    ``None`` means "this model is not served by this endpoint" — the route
    layer answers a native-shaped 404, exactly like ``api.anthropic.com``
    does for a model it does not host.  No cross-protocol bridging.
    """
    route = PROVIDER_ROUTES.get(provider)
    if route is None:
        return None
    return requested_path if requested_path in route.paths else None


def protocol_for(provider: str, path: str) -> str:
    """Wire protocol (``anthropic`` / ``openai_chat`` / ``responses``).

    Resolved from the provider's own route so a request that names a path the
    provider does not serve is still sanitized for the shape it will actually
    be sent in.
    """
    route = PROVIDER_ROUTES.get(provider)
    if route is not None and path not in route.paths:
        path = route.native_path
    if path.endswith("/messages"):
        return "anthropic"
    if path.endswith("/responses"):
        return "responses"
    return "openai_chat"


@dataclass(frozen=True, slots=True)
class RequestRoute:
    """Everything the forwarding layer needs for one passthrough request."""

    provider: str
    protocol: str
    """``anthropic`` | ``openai_chat`` | ``responses`` — the wire shape sent."""
    target_url: str
    """``X-Target-URL``: the upstream host the proxy will call."""
    proxy_path: str
    """Sub-path appended to ``{proxy_url}/proxy``."""
    model: str
    """Upstream ``model_id`` (catalog keys are resolved away)."""


def route_for_request(
    path: str, body: dict[str, Any], catalog: dict[str, Any]
) -> RequestRoute | None:
    """Resolve a client request to a reachable upstream route.

    ``None`` means "this model is not served here": either the catalog knows
    the model and its provider does not answer ``path``, or the catalog knows
    the model list and this model is not in it.  The route layer turns that
    into a native-shaped 404, which is what ``api.anthropic.com`` does for a
    model it does not host.

    When the catalog is empty (offline start, fetch failure) routing falls
    back to the path's native provider rather than refusing every request —
    an unreachable catalog must not brick the gateway.
    """
    model = str(body.get("model") or "")
    provider = _pkg.provider_for_model(catalog, model) or _pkg.provider_from_key(model)
    known_catalog = bool(_pkg.reachable_models(catalog))
    if provider is None:
        if known_catalog and model:
            return None  # catalog is live and does not serve this model
        provider = "anthropic" if path.endswith("/messages") else "openai"
    proxy_path = _pkg.proxy_path_for(provider, path)
    if proxy_path is None:
        return None
    return RequestRoute(
        provider=provider,
        protocol=_pkg.protocol_for(provider, path),
        target_url=PROVIDER_ROUTES[provider].target_url,
        proxy_path=proxy_path,
        model=_pkg.upstream_model_id(catalog, model),
    )


def target_for_request(path: str, body: dict[str, Any], catalog: dict[str, Any]) -> str:
    """Resolve the upstream ``X-Target-URL`` for a passthrough request.

    Routing by path + model:

    - ``/v1/messages`` is Anthropic-native → always targets the Anthropic host.
    - ``/v1/chat/completions`` and ``/v1/responses`` are OpenAI-native; the
      provider is inferred from the request's ``model`` field via the
      catalog, defaulting to OpenAI.  ``/v1/responses`` is the modern
      stateful Responses endpoint (reasoning models, background mode,
      polling) and routes to the same upstream hosts as chat/completions.
    - Any other path defaults to Anthropic.
    """
    if path.endswith("/messages"):
        return PROVIDER_BASE_URLS["anthropic"]
    model = str(body.get("model") or "")
    provider = _pkg.provider_for_model(catalog, model)
    if provider is None and model:
        provider = _pkg.provider_from_key(model)
    return PROVIDER_BASE_URLS.get(provider or "", PROVIDER_BASE_URLS["openai"])
