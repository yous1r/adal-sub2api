"""Request-body sanitizer matching the measured capabilities of the AdaL cloud proxy.

Upstream is a thin SDK wrapper that neither translates across protocols nor accepts
everything the official APIs do. Each rewrite below removes or adjusts exactly the
fields measured to trigger an upstream rejection, so clients never see a 500 that
sub2api could have prevented. The function never mutates the caller's dict: it works
on a shallow copy and deep-copies only the nested containers it actually rewrites.
Every removal or rewrite appends the affected field name (dotted for nested keys)
to `dropped` so the caller can log what was changed at DEBUG.
"""

from __future__ import annotations

import json

from sub2api.compat.profiles import (
    ANTHROPIC_FORBIDDEN_EXTRA,
    ANTHROPIC_METADATA_KEEP,
    ANTHROPIC_NO_SAMPLING,
    ANTHROPIC_TOOL_FORBIDDEN_EXTRA,
    ANTHROPIC_TOOL_NAMES,
    ANTHROPIC_TOOL_TYPES,
    ANTHROPIC_UNKNOWN_KWARGS,
    OPENAI_CHAT_DROP,
    RESPONSES_DROP,
)

_THINKING_SAMPLING_MODES = frozenset({"enabled", "adaptive"})
_SYNTHETIC_INPUT_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}


def sanitize(
    body: dict, *, protocol: str, upstream_model: str, dropped: list[str]
) -> dict:
    """Return a sanitized shallow copy of `body`; append removed/rewritten field names to `dropped`."""
    out = dict(body)
    if protocol == "anthropic":
        _sanitize_anthropic(out, upstream_model, dropped)
    elif protocol == "openai_chat":
        _sanitize_openai_chat(out, dropped)
    elif protocol == "responses":
        _sanitize_responses(out, dropped)
    return out


def _sanitize_anthropic(out: dict, upstream_model: str, dropped: list[str]) -> None:
    for key in ANTHROPIC_UNKNOWN_KWARGS | ANTHROPIC_FORBIDDEN_EXTRA:
        if key in out:
            del out[key]
            dropped.append(key)

    if isinstance(out.get("metadata"), dict):
        pruned = {
            k: v for k, v in out["metadata"].items() if k in ANTHROPIC_METADATA_KEEP
        }
        for key in sorted(set(out["metadata"]) - set(pruned)):
            dropped.append(f"metadata.{key}")
        if pruned:
            out["metadata"] = pruned
        else:
            del out["metadata"]
            dropped.append("metadata")

    thinking = out.get("thinking") if isinstance(out.get("thinking"), dict) else None
    if upstream_model in ANTHROPIC_NO_SAMPLING:
        for key in ("temperature", "top_p", "top_k"):
            if key in out:
                del out[key]
                dropped.append(key)
        if thinking is not None and thinking.get("type") == "enabled":
            out["thinking"] = {"type": "adaptive"}
            dropped.append("thinking.type")
            if "output_config" not in out:
                out["output_config"] = {"effort": "high"}
                dropped.append("output_config")
    else:
        if (
            thinking is not None
            and thinking.get("type") in _THINKING_SAMPLING_MODES
            and "temperature" in out
            and out["temperature"] != 1
        ):
            out["temperature"] = 1
            dropped.append("temperature")

    tools = out.get("tools")
    if isinstance(tools, list):
        out["tools"] = [_fix_tool(tool, i, dropped) for i, tool in enumerate(tools)]


def _fix_tool(tool: object, index: int, dropped: list[str]) -> object:
    if not isinstance(tool, dict):
        return tool
    # Two independent rewrites: a forbidden per-tool extra is removed, and an
    # unknown or misnamed `type` is demoted to a plain custom tool.  A tool
    # with a natively supported type can still carry a forbidden extra, so
    # neither check may short-circuit the other.
    extras = tool.keys() & ANTHROPIC_TOOL_FORBIDDEN_EXTRA
    demote = "type" in tool and (
        tool["type"] not in ANTHROPIC_TOOL_TYPES
        or ANTHROPIC_TOOL_NAMES.get(tool["type"], tool.get("name")) != tool.get("name")
    )
    if not extras and not demote:
        return tool
    fixed = {k: v for k, v in tool.items() if k not in extras}
    for key in sorted(extras):
        dropped.append(f"tools[{index}].{key}")
    if demote:
        fixed["type"] = "custom"
        if "input_schema" not in fixed:
            fixed["input_schema"] = dict(_SYNTHETIC_INPUT_SCHEMA)
        dropped.append(f"tools[{index}].type")
    return fixed


def _sanitize_openai_chat(out: dict, dropped: list[str]) -> None:
    if "max_tokens" in out:
        value = out.pop("max_tokens")
        dropped.append("max_tokens")
        if "max_completion_tokens" not in out:
            out["max_completion_tokens"] = value
    for key in OPENAI_CHAT_DROP:
        if key in out:
            del out[key]
            dropped.append(key)
    if out.get("temperature") is not None and out["temperature"] != 1:
        del out["temperature"]
        dropped.append("temperature")
    if "metadata" in out and "store" not in out:
        del out["metadata"]
        dropped.append("metadata")
    if out.get("response_format") == {"type": "json_object"}:
        serialized = json.dumps(out.get("messages"))
        if serialized is not None and "json" not in serialized.lower():
            del out["response_format"]
            dropped.append("response_format")
    if isinstance(out.get("tools"), list) and out["tools"]:
        out["reasoning_effort"] = "none"
        dropped.append("reasoning_effort")
    if out.get("stream"):
        stream_options = (
            dict(out["stream_options"])
            if isinstance(out.get("stream_options"), dict)
            else {}
        )
        if not stream_options.get("include_usage"):
            stream_options["include_usage"] = True
            out["stream_options"] = stream_options
            dropped.append("stream_options")


def _sanitize_responses(out: dict, dropped: list[str]) -> None:
    if "max_output_tokens" not in out:
        for key in ("max_tokens", "max_completion_tokens"):
            if key in out:
                value = out.pop(key)
                dropped.append(key)
                out["max_output_tokens"] = value
                break
    else:
        for key in ("max_tokens", "max_completion_tokens"):
            if key in out:
                del out[key]
                dropped.append(key)
    if "messages" in out and "input" not in out:
        out["input"] = out.pop("messages")
        dropped.append("messages")
    for key in RESPONSES_DROP:
        if key in out:
            del out[key]
            dropped.append(key)
