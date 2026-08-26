"""OpenAI Chat Completions compatibility: pure mapping helpers.

Translates between the OpenAI wire format and sub2api's normalized
contracts. No I/O here — everything is unit-testable without a server.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

ROLE_TAGS = {
    "system": "SYSTEM",
    "developer": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
    "tool": "TOOL",
}


def content_to_text(content: Any) -> str:
    """Flatten OpenAI message content (string or typed parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten a chat history into a single prompt for one-shot channels.

    A lone user message maps to its raw text; longer histories become a
    tagged transcript so the agent sees speaker boundaries.
    """
    if len(messages) == 1 and messages[0].get("role", "user") == "user":
        return content_to_text(messages[0].get("content"))
    sections = []
    for message in messages:
        role = ROLE_TAGS.get(message.get("role", ""), "USER")
        text = content_to_text(message.get("content"))
        sections.append(f"[{role}]\n{text}")
    return "\n\n".join(sections)


def completion_id() -> str:
    return f"chatcmpl-{uuid4().hex[:12]}"


def build_completion(
    *, id: str, created: int, model: str, content: str, finish_reason: str = "stop"
) -> dict[str, Any]:
    return {
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }


def build_chunk(
    *,
    id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# Sub2ApiError code -> (HTTP status, OpenAI error type)
_ERROR_MAPPING = {
    "unknown_channel": (404, "invalid_request_error"),
    "session_not_found": (404, "invalid_request_error"),
    "channel_mismatch": (409, "invalid_request_error"),
    "runtime_missing": (503, "api_error"),
    "auth_failed": (502, "api_error"),
    "channel_not_ready": (503, "api_error"),
    "upstream_failed": (502, "api_error"),
    "internal_error": (500, "internal_error"),
}


def error_status_and_type(code: str) -> tuple[int, str]:
    return _ERROR_MAPPING.get(code, (502, "api_error"))


def openai_error(message: str, *, err_type: str = "api_error", code: str | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"message": message, "type": err_type}
    if code:
        error["code"] = code
    return {"error": error}


def now_epoch() -> int:
    return int(time.time())
