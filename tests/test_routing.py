"""Tests for provider routing: the measured proxy route table.

Every ``X-Target-URL`` asserted here was verified HTTP 200 with real content
against ``api.adal.sylph.ai`` this session.  The proxy performs no
cross-protocol translation, so a provider is reachable *only* on the
sub-paths its row lists, and only at that exact target host.
"""

from __future__ import annotations

from sub2api.channels.adal_cloud import (
    PROVIDER_BASE_URLS,
    PROVIDER_ROUTES,
    UNLISTED_PROVIDERS,
    protocol_for,
    provider_from_key,
    proxy_path_for,
    reachable_models,
    route_for_request,
)

# provider -> measured X-Target-URL (the ten reachable rows).
MEASURED_TARGETS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "chatgpt_web": "https://api.openai.com",
    "zai": "https://api.z.ai/api/anthropic",
    "deepseek": "https://api.deepseek.com",
    "kimi": "https://api.moonshot.ai",
    "minimax": "https://api.minimax.io/anthropic",
    "xai": "https://api.x.ai",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode",
    "meta": "https://api.meta.ai",
}


# -- route table --------------------------------------------------------------


def test_every_measured_provider_resolves_to_its_target():
    assert PROVIDER_BASE_URLS == MEASURED_TARGETS
    for provider, target in MEASURED_TARGETS.items():
        assert PROVIDER_ROUTES[provider].target_url == target


def test_google_is_absent_until_a_native_gemini_bridge_exists():
    # The OpenAI-compat shim answers 403 model_not_allowed for every
    # permutation; only the native generativelanguage.googleapis.com paths
    # work, which needs a translation layer sub2api does not have.
    assert "google" not in PROVIDER_ROUTES
    assert "google" not in PROVIDER_BASE_URLS


def test_native_path_matches_the_provider_protocol_family():
    for provider, route in PROVIDER_ROUTES.items():
        assert route.native_path in route.paths, provider


# -- provider_from_key --------------------------------------------------------


def test_provider_from_key_longest_prefix_wins():
    # Whole-prefix matching, not model.split("-", 1)[0]: the split version
    # turns the now-advertised bare id gpt-5.6-luna into "gpt" and routes it
    # by a provider that exists in no table.
    assert provider_from_key("chatgpt_web-gpt-5.6-luna") == "chatgpt_web"
    assert provider_from_key("openai-gpt-5.6-terra") == "openai"
    assert provider_from_key("anthropic-claude-sonnet-4-6") == "anthropic"


def test_provider_from_key_none_for_bare_model_ids():
    assert provider_from_key("gpt-5.6-luna") is None
    assert provider_from_key("claude-sonnet-5") is None
    assert provider_from_key("") is None


def test_provider_from_key_rejects_unreachable_provider():
    assert provider_from_key("google-gemini-3.7-flash") is None


# -- proxy_path_for -----------------------------------------------------------


def test_proxy_path_for_anthropic_only_provider():
    assert proxy_path_for("zai", "/v1/messages") == "/v1/messages"
    assert proxy_path_for("zai", "/v1/chat/completions") is None
    assert proxy_path_for("zai", "/v1/responses") is None


def test_proxy_path_for_openai_only_provider():
    assert proxy_path_for("kimi", "/v1/chat/completions") == "/v1/chat/completions"
    assert proxy_path_for("kimi", "/v1/responses") == "/v1/responses"
    assert proxy_path_for("kimi", "/v1/messages") is None


def test_proxy_path_for_dual_protocol_provider():
    for path in ("/v1/messages", "/v1/chat/completions", "/v1/responses"):
        assert proxy_path_for("xai", path) == path


def test_proxy_path_for_unknown_provider_is_none():
    assert proxy_path_for("google", "/v1/chat/completions") is None
    assert proxy_path_for("", "/v1/messages") is None


# -- protocol_for -------------------------------------------------------------


def test_protocol_for_uses_the_requested_path_when_served():
    assert protocol_for("anthropic", "/v1/messages") == "anthropic"
    assert protocol_for("openai", "/v1/chat/completions") == "openai_chat"
    assert protocol_for("openai", "/v1/responses") == "responses"


def test_protocol_for_falls_back_to_the_native_path():
    # zai does not serve /v1/chat/completions; the body will go out
    # Anthropic-shaped, so it must be sanitized that way.
    assert protocol_for("zai", "/v1/chat/completions") == "anthropic"
    assert protocol_for("kimi", "/v1/messages") == "openai_chat"


def test_protocol_for_unknown_provider_trusts_the_path():
    assert protocol_for("", "/v1/messages") == "anthropic"
    assert protocol_for("google", "/v1/responses") == "responses"


# -- reachable_models ---------------------------------------------------------


CATALOG = {
    "models": [
        {
            "key": "anthropic-claude-sonnet-4-6",
            "model_id": "claude-sonnet-4-6",
            "provider": "anthropic",
        },
        {
            "key": "google-gemini-3.7-flash",
            "model_id": "gemini-3.7-flash",
            "provider": "google",
        },
        {
            "key": "chatgpt_web-gpt-5.6-luna",
            "model_id": "gpt-5.6-luna",
            "provider": "chatgpt_web",
        },
        {
            "key": "local-thing",
            "model_id": "thing",
            "provider": "openai",
            "is_local_model": True,
        },
        {"key": "zai-glm-5.2", "model_id": "glm-5.2", "provider": "zai"},
        {"provider": "anthropic"},
    ]
}


ASTRA_CATALOG = {
    "models": [
        {
            "key": "openai-gpt-6-astra",
            "model_id": "gpt-6-astra",
            "provider": "openai",
            "is_local_model": False,
        }
    ]
}


def test_astra_catalog_model_routes_through_openai_protocols():
    assert reachable_models(ASTRA_CATALOG) == ("gpt-6-astra",)
    for path in ("/v1/chat/completions", "/v1/responses"):
        route = route_for_request(path, {"model": "gpt-6-astra"}, ASTRA_CATALOG)
        assert route is not None
        assert route.provider == "openai"
        assert route.target_url == PROVIDER_ROUTES["openai"].target_url
        assert route.proxy_path == path
        assert route.model == "gpt-6-astra"


def test_reachable_models_advertises_bare_upstream_ids():
    # The catalog key is {provider}-{model_id} for all 36 measured entries;
    # clients configured for the official vendor API send the bare id, so
    # that is what /v1/models must list.
    assert reachable_models(CATALOG) == ("claude-sonnet-4-6", "glm-5.2")


def test_reachable_models_drops_google_and_local_models():
    ids = reachable_models(CATALOG)
    assert "gemini-3.7-flash" not in ids  # no verified route
    assert "thing" not in ids  # is_local_model


def test_reachable_models_drops_unlisted_providers():
    # chatgpt_web re-exports the openai model ids, so advertising it would
    # publish gpt-5.6-luna twice pointing at two different routes.
    assert "chatgpt_web" in UNLISTED_PROVIDERS
    assert "chatgpt_web" in PROVIDER_ROUTES  # still routable by explicit key
    assert "gpt-5.6-luna" not in reachable_models(CATALOG)


def test_reachable_models_falls_back_to_the_key_without_a_model_id():
    catalog = {"models": [{"key": "xai-grok-4.6", "provider": "xai"}]}
    assert reachable_models(catalog) == ("xai-grok-4.6",)


def test_reachable_models_deduplicates_repeated_ids():
    catalog = {
        "models": [
            {
                "key": "openai-gpt-5.6-sol",
                "model_id": "gpt-5.6-sol",
                "provider": "openai",
            },
            {"key": "kimi-gpt-5.6-sol", "model_id": "gpt-5.6-sol", "provider": "kimi"},
        ]
    }
    assert reachable_models(catalog) == ("gpt-5.6-sol",)


def test_reachable_models_tolerates_garbage():
    assert reachable_models({}) == ()
    assert reachable_models({"models": "bogus"}) == ()
    assert reachable_models(None) == ()


# -- bare upstream ids route identically to catalog keys ----------------------


def test_bare_upstream_id_routes_like_its_catalog_key():
    # The contract behind advertising bare ids: whatever /v1/models lists must
    # resolve, and the old prefixed key must keep resolving the same way.
    for model in ("claude-sonnet-4-6", "anthropic-claude-sonnet-4-6"):
        route = route_for_request("/v1/messages", {"model": model}, CATALOG)
        assert route is not None, model
        assert route.provider == "anthropic"
        assert route.target_url == "https://api.anthropic.com"
        assert route.proxy_path == "/v1/messages"
        assert route.model == "claude-sonnet-4-6"  # always sent bare upstream


def test_every_advertised_id_resolves_to_a_route():
    for model in reachable_models(CATALOG):
        provider = PROVIDER_ROUTES[
            next(m["provider"] for m in CATALOG["models"] if m.get("model_id") == model)
        ]
        route = route_for_request(provider.native_path, {"model": model}, CATALOG)
        assert route is not None, model
        assert route.model == model


def test_unlisted_provider_still_routes_by_explicit_key():
    # chatgpt_web is hidden from /v1/models, not disabled: the explicit key
    # keeps working so existing configs do not break.
    route = route_for_request(
        "/v1/chat/completions", {"model": "chatgpt_web-gpt-5.6-luna"}, CATALOG
    )
    assert route is not None
    assert route.provider == "chatgpt_web"
    assert route.model == "gpt-5.6-luna"


def test_bare_id_shared_with_an_unlisted_provider_prefers_the_listed_one():
    # gpt-5.6-luna exists under both openai and chatgpt_web in the live
    # catalog; an unqualified request must land on the advertised route.
    catalog = {
        "models": [
            {
                "key": "chatgpt_web-gpt-5.6-luna",
                "model_id": "gpt-5.6-luna",
                "provider": "chatgpt_web",
            },
            {
                "key": "openai-gpt-5.6-luna",
                "model_id": "gpt-5.6-luna",
                "provider": "openai",
            },
        ]
    }
    route = route_for_request(
        "/v1/chat/completions", {"model": "gpt-5.6-luna"}, catalog
    )
    assert route is not None
    assert route.provider == "openai"


def test_unknown_model_is_refused_when_the_catalog_is_live():
    assert route_for_request("/v1/messages", {"model": "nope-9"}, CATALOG) is None


def test_bare_id_falls_back_to_the_path_provider_when_the_catalog_is_empty():
    # Offline start: no catalog to look the bare id up in, so the endpoint's
    # native provider wins rather than refusing every request.
    route = route_for_request("/v1/messages", {"model": "claude-sonnet-5"}, {})
    assert route is not None
    assert route.provider == "anthropic"
    assert route.model == "claude-sonnet-5"
