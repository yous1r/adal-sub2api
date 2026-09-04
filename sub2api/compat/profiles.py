"""Frozen capability profiles for the AdaL cloud proxy.

Every entry below is grounded in a live rejection measured against the real proxy:
sending a listed field produced the exact upstream error named in the comment, so
these tables exist to keep sub2api from forwarding inputs the proxy demonstrably
cannot accept.
"""

from __future__ import annotations

# Upstream raises: Messages.stream() got an unexpected keyword argument '<key>'
ANTHROPIC_UNKNOWN_KWARGS = frozenset(
    {
        "context_management",
        "mcp_servers",
        "stream_options",
        "n",
        "top_logprobs",
        "betas",
        "system_prompt",
        "logit_bias",
        "seed",
        "user",
        "response_format",
        "max_completion_tokens",
        "parallel_tool_calls",
    }
)

# Upstream pydantic raises: Extra inputs are not permitted
ANTHROPIC_FORBIDDEN_EXTRA = frozenset({"service_tier", "container", "cache_control"})

# metadata.session_id triggers: Extra inputs are not permitted
ANTHROPIC_METADATA_KEEP = frozenset({"user_id"})

# Upstream raises: `temperature` is deprecated for this model
ANTHROPIC_NO_SAMPLING = frozenset(
    {"claude-sonnet-5", "claude-opus-5", "claude-fable-5-1"}
)

# Per-model tool-type allow-set; unknown types were rejected with a 400 on claude-sonnet-4-6
ANTHROPIC_TOOL_TYPES = frozenset(
    {
        "custom",
        "bash_20250124",
        "memory_20250818",
        "text_editor_20250728",
        "tool_search_tool_bm25_20251119",
        "tool_search_tool_regex_20251119",
    }
)

# Canonical server-side tool names; a mismatching custom name was rejected upstream
ANTHROPIC_TOOL_NAMES = {
    "bash_20250124": "bash",
    "memory_20250818": "memory",
    "text_editor_20250728": "str_replace_based_edit_tool",
}

# Measured: each key here provoked an upstream 400 on /v1/chat/completions
OPENAI_CHAT_DROP = frozenset(
    {
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "logprobs",
        "top_logprobs",
    }
)

# Measured: each key here provoked an upstream 400 on /v1/responses
RESPONSES_DROP = frozenset({"temperature", "top_p", "stop"})
