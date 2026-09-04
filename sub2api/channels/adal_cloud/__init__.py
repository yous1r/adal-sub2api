"""AdaL channel backed by the cloud proxy API (no local runtime).

This channel talks **directly** to AdaL's hosted proxy at
``api.adal.sylph.ai`` — there is no ``adal`` CLI, no ``adal --sdk-runtime``
subprocess, and no local ``adal-backend`` binary involved.  The user's
subscription quota is consumed through the same proxy the official clients
use; sub2api merely re-implements the thin client side of that protocol.

Protocol (reverse-engineered from the bundled ``adal-backend``):

  1. authenticate — read a Clerk JWT from ``~/.adal/adal_oauth_creds.json``
     (or ``SUB2API_AUTH_TOKEN``); if absent/expired, run the OAuth **device
     code** flow against ``adal.sylph.ai`` and persist the fresh token.
  2. register session — ``POST {ADAL_APP_URL}/api/client-sessions/start``
     with ``{session_id, source, ...}`` and ``Authorization: Bearer <jwt>``.
     The remote stores the ``session_id`` keyed to the user; one registration
     per session_id is enough and is cached for the channel's lifetime.
  3. chat — forward the request to ``{ADAL_PROXY_URL}/proxy/<path>`` with
     ``X-Session-ID`` and ``X-Target-URL`` headers.  The proxy enforces
     subscription billing and relays to the upstream provider (Anthropic via
     AWS Bedrock, OpenAI, Google, Z.AI, …).  Two shapes are supported:

       * Anthropic native: ``/proxy/v1/messages`` → target ``api.anthropic.com``
       * OpenAI native:    ``/proxy/v1/chat/completions`` → target ``api.openai.com``

  4. model catalog — ``GET {ADAL_PROXY_URL}/proxy/models/catalog`` (no auth)
     yields the live model list keyed by ``provider`` → ``model_client``.

The provider→target base-URL map mirrors ``adal_backend``'s
``_CLIENT_REGISTRY`` so every catalog model routes to its real upstream.

Docs: https://docs.sylph.ai/cloud-agents/overview

The implementation is split by concern — :mod:`.auth` (Clerk identity),
:mod:`.catalog` (live model list), :mod:`.routing` (measured proxy route
table), :mod:`.quota` (subscription credits), :mod:`.sse` (wire formats) and
:mod:`.channel` (stateful channel) — but this package is the *public* surface:
every name below is re-exported here, and the submodules resolve their
collaborators through this namespace so rebinding a name here (which the test
suite does to stay offline) is honoured by the code that calls it.
"""

from __future__ import annotations

# Re-exported so ``sub2api.channels.adal_cloud.asyncio.to_thread``,
# ``...time.sleep`` and ``...httpx.post`` remain patchable module attributes,
# as they were when this package was one module.
import asyncio
import time
from urllib.request import urlopen

import httpx

from ...core.pool import (
    AccountConfig,
    AccountPool,
    AccountSlot,
    PoolConfig,
    load_pool_config,
)
from .auth import (
    ADAL_APP_URL,
    CLERK_BASE,
    CLERK_MINT_HEADERS,
    CREDS_PATH,
    DEAD_COOLDOWN_S,
    DEAD_MINT_ERRORS,
    SESSION_PATH,
    TOKEN_REFRESH_SKEW,
    TRANSIENT_COOLDOWN_S,
    _jwt_claims,
    _jwt_exp,
    clerk_id_from_token,
    device_flow_login,
    initiate_device_flow,
    load_cached_session,
    mint_token_with_cookies,
    poll_device_flow,
    read_token,
    refresh_token_with_cookies,
    register_session,
    save_cached_session,
    token_needs_refresh,
)
from .catalog import (
    ADAL_PROXY_URL,
    catalog_models,
    fetch_catalog,
    fetch_tiers,
    provider_for_model,
    reachable_models,
    tier_display_name,
    upstream_model_id,
)
from .channel import AdalCloudChannel
from .quota import (
    TIERS_CACHE_TTL,
    USAGE_CACHE_TTL,
    USAGE_FETCH_CONCURRENCY,
    USAGE_UNIT,
    _as_float,
    _invalid_message,
    aggregate_quota,
    fetch_account_quota,
    fetch_credits_balance,
    fetch_credits_raw,
    parse_credits_balance,
)
from .routing import (
    PROVIDER_BASE_URLS,
    PROVIDER_ROUTES,
    ProviderRoute,
    RequestRoute,
    protocol_for,
    provider_from_key,
    proxy_path_for,
    route_for_request,
    target_for_request,
)
from .sse import (
    _has_cache_control,
    _inject_cache_control,
    anthropic_request,
    openai_request,
    parse_anthropic_sse,
    parse_openai_sse,
)

__all__ = [
    "ADAL_APP_URL",
    "ADAL_PROXY_URL",
    "AccountConfig",
    "AccountPool",
    "AccountSlot",
    "AdalCloudChannel",
    "CLERK_BASE",
    "CLERK_MINT_HEADERS",
    "CREDS_PATH",
    "DEAD_COOLDOWN_S",
    "DEAD_MINT_ERRORS",
    "PROVIDER_BASE_URLS",
    "PROVIDER_ROUTES",
    "PoolConfig",
    "ProviderRoute",
    "RequestRoute",
    "SESSION_PATH",
    "TIERS_CACHE_TTL",
    "TOKEN_REFRESH_SKEW",
    "TRANSIENT_COOLDOWN_S",
    "USAGE_CACHE_TTL",
    "USAGE_FETCH_CONCURRENCY",
    "USAGE_UNIT",
    "_as_float",
    "_has_cache_control",
    "_inject_cache_control",
    "_invalid_message",
    "_jwt_claims",
    "_jwt_exp",
    "aggregate_quota",
    "anthropic_request",
    "asyncio",
    "catalog_models",
    "clerk_id_from_token",
    "device_flow_login",
    "fetch_account_quota",
    "fetch_catalog",
    "fetch_credits_balance",
    "fetch_credits_raw",
    "fetch_tiers",
    "httpx",
    "initiate_device_flow",
    "load_cached_session",
    "load_pool_config",
    "mint_token_with_cookies",
    "openai_request",
    "parse_anthropic_sse",
    "parse_credits_balance",
    "parse_openai_sse",
    "poll_device_flow",
    "protocol_for",
    "provider_for_model",
    "provider_from_key",
    "proxy_path_for",
    "reachable_models",
    "read_token",
    "refresh_token_with_cookies",
    "register_session",
    "route_for_request",
    "save_cached_session",
    "target_for_request",
    "tier_display_name",
    "time",
    "token_needs_refresh",
    "upstream_model_id",
    "urlopen",
]
