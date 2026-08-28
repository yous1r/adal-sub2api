"""Tests for prompt-cache auto-injection in rewrite_body."""

from __future__ import annotations

import json

from sub2api.channels.adal_cloud import (
    AdalCloudChannel,
    _has_cache_control,
    _inject_cache_control,
)
from sub2api.core.channel import ChannelConfig


SAMPLE_CATALOG = {
    "models": [
        {
            "key": "anthropic-claude-sonnet-5",
            "model_id": "claude-sonnet-5",
            "provider": "anthropic",
        },
        {"key": "openai-gpt-5.6-sol", "model_id": "gpt-5.6-sol", "provider": "openai"},
    ]
}


def _channel() -> AdalCloudChannel:
    ch = AdalCloudChannel(ChannelConfig())
    ch._catalog = SAMPLE_CATALOG
    return ch


# -- _has_cache_control / _inject_cache_control -------------------------------


def test_has_cache_control_string_returns_false():
    assert not _has_cache_control("system prompt")


def test_has_cache_control_dict_without():
    assert not _has_cache_control({"type": "text", "text": "hi"})


def test_has_cache_control_dict_with():
    assert _has_cache_control(
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
    )


def test_has_cache_control_list_without():
    assert not _has_cache_control([{"type": "text", "text": "hi"}])


def test_has_cache_control_list_with():
    assert _has_cache_control(
        [{"type": "text", "text": "hi"}, {"cache_control": {"type": "ephemeral"}}]
    )


def test_inject_cache_control_on_dict():
    d = {"type": "text", "text": "hi"}
    _inject_cache_control(d)
    assert d["cache_control"] == {"type": "ephemeral"}


def test_inject_cache_control_on_list():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    _inject_cache_control(blocks)
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[0]


def test_inject_cache_control_on_string_is_noop():
    s = "system prompt"
    _inject_cache_control(s)  # should not raise


# -- rewrite_body: Anthropic cache_control injection --------------------------


def test_anthropic_string_system_gets_cache_control():
    ch = _channel()
    body = json.dumps(
        {
            "model": "anthropic-claude-sonnet-5",
            "max_tokens": 64,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode()
    result = json.loads(ch.rewrite_body(body))
    system = result["system"]
    assert isinstance(system, list)
    assert system[-1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_list_system_gets_cache_control_on_last_block():
    ch = _channel()
    body = json.dumps(
        {
            "model": "anthropic-claude-sonnet-5",
            "max_tokens": 64,
            "system": [
                {"type": "text", "text": "block 1"},
                {"type": "text", "text": "block 2"},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode()
    result = json.loads(ch.rewrite_body(body))
    assert result["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in result["system"][0]


def test_anthropic_with_existing_cache_control_not_modified():
    ch = _channel()
    body = json.dumps(
        {
            "model": "anthropic-claude-sonnet-5",
            "max_tokens": 64,
            "system": [
                {
                    "type": "text",
                    "text": "block 1",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode()
    result = json.loads(ch.rewrite_body(body))
    # Already has cache_control → no new injection
    assert result["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_no_system_not_crash():
    ch = _channel()
    body = json.dumps(
        {
            "model": "anthropic-claude-sonnet-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode()
    result = json.loads(ch.rewrite_body(body))
    assert "system" not in result


# -- rewrite_body: OpenAI prompt_cache_key injection --------------------------


def test_openai_responses_gets_prompt_cache_key():
    ch = _channel()
    body = json.dumps(
        {
            "model": "openai-gpt-5.6-sol",
            "input": [{"role": "user", "content": "hi"}],
        }
    ).encode()
    result = json.loads(ch.rewrite_body(body))
    assert result["prompt_cache_key"] == "sub2api"


def test_openai_responses_with_existing_key_not_overwritten():
    ch = _channel()
    body = json.dumps(
        {
            "model": "openai-gpt-5.6-sol",
            "input": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "custom-key",
        }
    ).encode()
    result = json.loads(ch.rewrite_body(body))
    assert result["prompt_cache_key"] == "custom-key"


def test_openai_chat_completions_no_input_field_no_cache_key():
    ch = _channel()
    body = json.dumps(
        {
            "model": "openai-gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode()
    result = json.loads(ch.rewrite_body(body))
    # chat/completions uses "messages" not "input" — no prompt_cache_key injected
    assert "prompt_cache_key" not in result
