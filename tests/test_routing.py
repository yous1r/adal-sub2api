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
    protocol_for,
    provider_from_key,
    proxy_path_for,
    reachable_models,
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
    # The old model.split("-", 1)[0] returned "chatgpt", missed every table,
    # and silently routed an OpenAI request to the Anthropic host.
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
        {"key": "anthropic-claude-sonnet-4-6", "provider": "anthropic"},
        {"key": "google-gemini-3.7-flash", "provider": "google"},
        {"key": "chatgpt_web-gpt-5.6-luna", "provider": "chatgpt_web"},
        {"key": "local-thing", "provider": "openai", "is_local_model": True},
        {"key": "zai-glm-5.2", "provider": "zai"},
        {"provider": "anthropic"},
    ]
}


def test_reachable_models_drops_google_and_local_models():
    assert reachable_models(CATALOG) == (
        "anthropic-claude-sonnet-4-6",
        "chatgpt_web-gpt-5.6-luna",
        "zai-glm-5.2",
    )


def test_reachable_models_tolerates_garbage():
    assert reachable_models({}) == ()
    assert reachable_models({"models": "bogus"}) == ()
    assert reachable_models(None) == ()
