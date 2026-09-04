"""Wire-format helpers: request builders, SSE parsers, prompt-cache injection.

These are the only pieces that know the *shape* of an Anthropic or OpenAI
payload, so they live apart from the channel (which owns auth, pooling and
transport) and from the route table (which owns hosts and paths).  All pure
functions over strings/dicts: no I/O, no channel state, trivially testable.
"""

from __future__ import annotations

import json
from typing import Any

from ...core.errors import UpstreamError
from ...core.types import (
    ChatRequest,
    Event,
    TextDelta,
    ThoughtDelta,
)


def anthropic_request(model: str, request: ChatRequest) -> dict[str, Any]:
    """Build an Anthropic ``/v1/messages`` body for one sub2api turn."""
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": True,
    }
    if request.thinking_effort:
        # "adaptive" lets the model/proxy decide thinking depth from effort;
        # the proxy maps effort to provider-native thinking config server-side.
        body["thinking"] = {"type": "adaptive"}
    return body


def openai_request(model: str, request: ChatRequest) -> dict[str, Any]:
    """Build an OpenAI ``/v1/chat/completions`` body for one sub2api turn."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": True,
    }
    if request.thinking_effort:
        body["reasoning_effort"] = request.thinking_effort
    return body


def parse_anthropic_sse(raw: str) -> list[Event]:
    """Translate one Anthropic SSE event line into normalized events.

    Anthropic streams ``event: <type>\\ndata: <json>`` pairs.  This parser is
    driven by the ``data:`` payload's ``type`` and delta shape; the ``event:``
    line is informational.
    """
    if not raw or not raw.startswith("data:"):
        return []
    payload = raw[5:].strip()
    if not payload:
        return []
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return []
    typ = obj.get("type")
    if typ == "content_block_delta":
        delta = obj.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta":
            return [TextDelta(text=delta.get("text", ""))]
        if dtype in ("thinking_delta", "signature_delta"):
            return [ThoughtDelta(text=delta.get("thinking", delta.get("text", "")))]
        return []
    if typ == "message_stop":
        return []  # terminal handled by caller via the non-stream envelope
    if typ == "error":
        err = obj.get("error") or {}
        raise UpstreamError(err.get("message", "anthropic upstream error"))
    return []


def parse_openai_sse(raw: str) -> list[Event]:
    """Translate one OpenAI chat-completion SSE chunk into normalized events."""
    if not raw or not raw.startswith("data:"):
        return []
    payload = raw[5:].strip()
    if not payload:
        return []
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if obj.get("error"):
        raise UpstreamError(str(obj["error"].get("message", "openai upstream error")))
    choices = obj.get("choices") or []
    if not choices:
        return []
    delta = choices[0].get("delta") or {}
    events: list[Event] = []
    if delta.get("reasoning_content"):
        events.append(ThoughtDelta(text=delta["reasoning_content"]))
    if delta.get("content"):
        events.append(TextDelta(text=delta["content"]))
    return events


# -- prompt cache helpers ----------------------------------------------------


def _has_cache_control(system: Any) -> bool:
    """True when *system* already carries ``cache_control`` somewhere."""
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                return True
    elif isinstance(system, dict):
        return "cache_control" in system
    elif isinstance(system, str):
        # Plain string system — no cache_control possible.
        pass
    return False


def _inject_cache_control(system: Any) -> None:
    """Inject ``cache_control: {type:"ephemeral"}`` on the last block of
    *system* in place.  Handles string, dict, and list-of-dicts formats.
    """
    ephemeral: dict[str, str] = {"type": "ephemeral"}
    if isinstance(system, str):
        # Can't mutate a string; caller should have converted it.
        return
    if isinstance(system, dict):
        system["cache_control"] = ephemeral
        return
    if isinstance(system, list) and system:
        last = system[-1]
        if isinstance(last, dict):
            last["cache_control"] = ephemeral
