"""Tests for the request-body sanitizer against measured AdaL cloud-proxy rejections.

Each test pins a behaviour traceable to a live upstream error: a field the proxy
rejects must be removed (and recorded in `dropped`), a field the proxy accepts
natively must survive untouched, and the caller's dict must never be mutated.
"""

from __future__ import annotations

import pytest

from sub2api.compat.sanitize import sanitize


def test_anthropic_drops_measured_unknown_kwargs() -> None:
    body = {
        "model": "claude-sonnet-4-6",
        "context_management": {"edits": []},
        "mcp_servers": [],
        "betas": ["interleaved-thinking-2025-05-14"],
        "stream_options": {"include_usage": True},
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    dropped: list[str] = []
    out = sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-4-6", dropped=dropped
    )
    assert "context_management" not in out
    assert "mcp_servers" not in out
    assert "betas" not in out
    assert "stream_options" not in out
    assert out["max_tokens"] == 64
    assert {"context_management", "mcp_servers", "betas", "stream_options"} <= set(
        dropped
    )


def test_anthropic_metadata_pruned_to_user_id() -> None:
    body = {
        "metadata": {"user_id": "u", "session_id": "s"},
        "max_tokens": 64,
        "messages": [],
    }
    dropped: list[str] = []
    out = sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-4-6", dropped=dropped
    )
    assert out["metadata"] == {"user_id": "u"}
    assert "metadata.session_id" in dropped


def test_anthropic_metadata_removed_when_empty_after_prune() -> None:
    body = {"metadata": {"session_id": "s"}, "max_tokens": 64, "messages": []}
    dropped: list[str] = []
    out = sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-4-6", dropped=dropped
    )
    assert "metadata" not in out
    assert "metadata" in dropped


def test_anthropic_no_sampling_model_drops_sampling_and_rewrites_thinking() -> None:
    body = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "thinking": {"type": "enabled", "budget_tokens": 8000},
        "max_tokens": 64,
        "messages": [],
    }
    dropped: list[str] = []
    out = sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-5", dropped=dropped
    )
    assert "temperature" not in out
    assert "top_p" not in out
    assert "top_k" not in out
    assert out["thinking"] == {"type": "adaptive"}
    assert out["output_config"] == {"effort": "high"}
    assert {"temperature", "top_p", "top_k", "thinking.type", "output_config"} <= set(
        dropped
    )


def test_anthropic_client_output_config_survives_thinking_rewrite() -> None:
    body = {
        "thinking": {"type": "enabled", "budget_tokens": 8000},
        "output_config": {"effort": "low"},
        "max_tokens": 64,
        "messages": [],
    }
    dropped: list[str] = []
    out = sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-5", dropped=dropped
    )
    assert out["output_config"] == {"effort": "low"}
    assert out["thinking"] == {"type": "adaptive"}


def test_anthropic_temperature_forced_to_one_when_thinking_on_4_6_family() -> None:
    body = {
        "temperature": 0.7,
        "thinking": {"type": "adaptive", "budget_tokens": 2000},
        "max_tokens": 64,
        "messages": [],
    }
    dropped: list[str] = []
    out = sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-4-6", dropped=dropped
    )
    assert out["temperature"] == 1
    assert "temperature" in dropped


def test_anthropic_temperature_untouched_without_thinking() -> None:
    body = {"temperature": 0.7, "max_tokens": 64, "messages": []}
    dropped: list[str] = []
    out = sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-4-6", dropped=dropped
    )
    assert out["temperature"] == 0.7
    assert "temperature" not in dropped


def test_anthropic_bash_typed_tool_survives_untouched() -> None:
    tool = {"type": "bash_20250124", "name": "bash"}
    body = {"tools": [tool], "max_tokens": 64, "messages": []}
    dropped: list[str] = []
    out = sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-4-6", dropped=dropped
    )
    assert out["tools"] == [tool]
    assert dropped == []


def test_anthropic_unknown_tool_type_demoted_to_custom_with_schema() -> None:
    body = {
        "tools": [{"type": "text_editor_20250429", "name": "str_replace_editor"}],
        "max_tokens": 64,
        "messages": [],
    }
    dropped: list[str] = []
    out = sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-4-6", dropped=dropped
    )
    tool = out["tools"][0]
    assert tool["type"] == "custom"
    assert tool["input_schema"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }
    assert tool["name"] == "str_replace_editor"
    assert "tools[0].type" in dropped


def test_anthropic_untyped_tool_with_input_schema_untouched() -> None:
    tool = {
        "name": "Bash",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
    body = {"tools": [tool], "max_tokens": 64, "messages": []}
    dropped: list[str] = []
    out = sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-4-6", dropped=dropped
    )
    assert out["tools"] == [tool]
    assert dropped == []


def test_anthropic_no_sampling_temperature_not_forced_on_4_6() -> None:
    # The no-sampling drop must not fire for the 4-6 family: temperature stays
    # subject only to the thinking-mode == 1 rule.
    body = {
        "temperature": 1,
        "thinking": {"type": "enabled"},
        "max_tokens": 64,
        "messages": [],
    }
    dropped: list[str] = []
    out = sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-4-6", dropped=dropped
    )
    assert out["temperature"] == 1
    assert out["thinking"] == {"type": "enabled"}
    assert dropped == []


def test_openai_chat_max_tokens_renamed() -> None:
    body = {"max_tokens": 100, "messages": []}
    dropped: list[str] = []
    out = sanitize(
        body, protocol="openai_chat", upstream_model="gpt-5.6-luna", dropped=dropped
    )
    assert "max_tokens" not in out
    assert out["max_completion_tokens"] == 100
    assert "max_tokens" in dropped


def test_openai_chat_max_tokens_dropped_when_completion_present() -> None:
    body = {"max_tokens": 100, "max_completion_tokens": 200, "messages": []}
    dropped: list[str] = []
    out = sanitize(
        body, protocol="openai_chat", upstream_model="gpt-5.6-luna", dropped=dropped
    )
    assert "max_tokens" not in out
    assert out["max_completion_tokens"] == 200


def test_openai_chat_drops_measured_penalty_keys() -> None:
    body = {
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "stop": ["x"],
        "messages": [],
    }
    dropped: list[str] = []
    out = sanitize(
        body, protocol="openai_chat", upstream_model="gpt-5.6-luna", dropped=dropped
    )
    assert not {"top_p", "frequency_penalty", "presence_penalty", "stop"} & set(out)
    assert {"top_p", "frequency_penalty", "presence_penalty", "stop"} <= set(dropped)


def test_openai_chat_temperature_dropped_unless_one() -> None:
    dropped: list[str] = []
    out = sanitize(
        {"temperature": 0.7, "messages": []},
        protocol="openai_chat",
        upstream_model="m",
        dropped=dropped,
    )
    assert "temperature" not in out
    dropped = []
    out = sanitize(
        {"temperature": 1, "messages": []},
        protocol="openai_chat",
        upstream_model="m",
        dropped=dropped,
    )
    assert out["temperature"] == 1
    assert "temperature" not in dropped


def test_openai_chat_metadata_dropped_unless_store() -> None:
    dropped: list[str] = []
    out = sanitize(
        {"metadata": {"a": 1}, "messages": []},
        protocol="openai_chat",
        upstream_model="m",
        dropped=dropped,
    )
    assert "metadata" not in out
    dropped = []
    out = sanitize(
        {"metadata": {"a": 1}, "store": True, "messages": []},
        protocol="openai_chat",
        upstream_model="m",
        dropped=dropped,
    )
    assert out["metadata"] == {"a": 1}
    assert "metadata" not in dropped


def test_openai_chat_json_object_response_format_dropped_without_json_mention() -> None:
    body = {
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    dropped: list[str] = []
    out = sanitize(body, protocol="openai_chat", upstream_model="m", dropped=dropped)
    assert "response_format" not in out
    assert "response_format" in dropped


def test_openai_chat_json_object_response_format_kept_when_messages_mention_json() -> (
    None
):
    body = {
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": "reply in JSON"}],
    }
    dropped: list[str] = []
    out = sanitize(body, protocol="openai_chat", upstream_model="m", dropped=dropped)
    assert out["response_format"] == {"type": "json_object"}
    assert "response_format" not in dropped


def test_openai_chat_reasoning_effort_forced_none_with_tools() -> None:
    body = {
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "reasoning_effort": "high",
        "messages": [],
    }
    dropped: list[str] = []
    out = sanitize(
        body, protocol="openai_chat", upstream_model="gpt-5.6-luna", dropped=dropped
    )
    assert out["reasoning_effort"] == "none"
    assert "reasoning_effort" in dropped


def test_openai_chat_reasoning_effort_untouched_without_tools() -> None:
    body = {"reasoning_effort": "high", "messages": []}
    dropped: list[str] = []
    out = sanitize(body, protocol="openai_chat", upstream_model="m", dropped=dropped)
    assert out["reasoning_effort"] == "high"
    assert "reasoning_effort" not in dropped


def test_openai_chat_stream_options_injected_on_stream() -> None:
    body = {"stream": True, "messages": []}
    dropped: list[str] = []
    out = sanitize(body, protocol="openai_chat", upstream_model="m", dropped=dropped)
    assert out["stream_options"] == {"include_usage": True}
    assert "stream_options" in dropped


def test_openai_chat_stream_options_merged_not_clobbered() -> None:
    body = {"stream": True, "stream_options": {"include_usage": False}, "messages": []}
    dropped: list[str] = []
    out = sanitize(body, protocol="openai_chat", upstream_model="m", dropped=dropped)
    assert out["stream_options"]["include_usage"] is True


def test_openai_chat_no_stream_options_without_stream() -> None:
    body = {"messages": []}
    dropped: list[str] = []
    out = sanitize(body, protocol="openai_chat", upstream_model="m", dropped=dropped)
    assert "stream_options" not in out


def test_responses_max_tokens_renamed_to_max_output_tokens() -> None:
    body = {"max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
    dropped: list[str] = []
    out = sanitize(
        body, protocol="responses", upstream_model="gpt-5.6-terra", dropped=dropped
    )
    assert "max_tokens" not in out
    assert out["max_output_tokens"] == 100
    assert out["input"] == [{"role": "user", "content": "hi"}]
    assert {"max_tokens", "messages"} <= set(dropped)


def test_responses_max_completion_tokens_renamed_when_no_max_tokens() -> None:
    body = {"max_completion_tokens": 100, "messages": []}
    dropped: list[str] = []
    out = sanitize(body, protocol="responses", upstream_model="m", dropped=dropped)
    assert "max_completion_tokens" not in out
    assert out["max_output_tokens"] == 100


def test_responses_existing_max_output_tokens_wins() -> None:
    body = {"max_tokens": 100, "max_output_tokens": 200, "messages": []}
    dropped: list[str] = []
    out = sanitize(body, protocol="responses", upstream_model="m", dropped=dropped)
    assert out["max_output_tokens"] == 200
    assert "max_tokens" not in out


def test_responses_drops_measured_sampling_keys() -> None:
    body = {"temperature": 0.5, "top_p": 0.9, "stop": ["x"], "messages": []}
    dropped: list[str] = []
    out = sanitize(body, protocol="responses", upstream_model="m", dropped=dropped)
    assert not {"temperature", "top_p", "stop"} & set(out)
    assert {"temperature", "top_p", "stop"} <= set(dropped)


def test_unknown_protocol_returns_copy_unchanged() -> None:
    body = {"weird": {"a": 1}, "max_tokens": 10}
    dropped: list[str] = []
    out = sanitize(body, protocol="grpc", upstream_model="m", dropped=dropped)
    assert out == body
    assert out is not body
    assert dropped == []


def test_caller_dict_never_mutated() -> None:
    import copy

    body = {
        "context_management": {},
        "metadata": {"user_id": "u", "session_id": "s"},
        "temperature": 0.7,
        "thinking": {"type": "enabled", "budget_tokens": 8000},
        "tools": [{"type": "text_editor_20250429", "name": "str_replace_editor"}],
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    original = copy.deepcopy(body)
    dropped: list[str] = []
    sanitize(
        body, protocol="anthropic", upstream_model="claude-sonnet-5", dropped=dropped
    )
    assert body == original
    # openai_chat path too
    body2 = {"max_tokens": 100, "temperature": 0.7, "stream": True, "messages": []}
    original2 = copy.deepcopy(body2)
    sanitize(body2, protocol="openai_chat", upstream_model="m", dropped=[])
    assert body2 == original2


def test_sanitize_returns_new_top_level_dict() -> None:
    body = {"max_tokens": 10, "messages": []}
    out = sanitize(body, protocol="anthropic", upstream_model="m", dropped=[])
    assert out is not body


@pytest.mark.parametrize(
    ("body", "protocol", "upstream_model", "expect"),
    [
        (
            {"betas": ["x"]},
            "anthropic",
            "claude-sonnet-4-6",
            lambda out, d: "betas" not in out and "betas" in d,
        ),
        (
            {"tools": [{"type": "memory_20250818", "name": "memory"}]},
            "anthropic",
            "claude-sonnet-4-6",
            lambda out, d: (
                out["tools"] == [{"type": "memory_20250818", "name": "memory"}]
            ),
        ),
        (
            {"tools": [{"type": "bash_20250124", "name": "custom_bash_name"}]},
            "anthropic",
            "claude-sonnet-4-6",
            lambda out, d: out["tools"][0]["type"] == "custom",
        ),
        (
            {"service_tier": "auto"},
            "anthropic",
            "m",
            lambda out, d: "service_tier" not in out,
        ),
        (
            {"cache_control": {"type": "ephemeral"}},
            "anthropic",
            "m",
            lambda out, d: "cache_control" not in out,
        ),
        (
            {"logprobs": True, "top_logprobs": 3},
            "openai_chat",
            "m",
            lambda out, d: "logprobs" not in out and "top_logprobs" not in out,
        ),
        (
            {"metadata": {}, "messages": []},
            "openai_chat",
            "m",
            lambda out, d: "metadata" not in out,
        ),
    ],
)
def test_table_driven_rejections(
    body: dict, protocol: str, upstream_model: str, expect: object
) -> None:
    dropped: list[str] = []
    out = sanitize(
        body, protocol=protocol, upstream_model=upstream_model, dropped=dropped
    )
    assert expect(out, dropped)
