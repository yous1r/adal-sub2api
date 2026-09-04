"""Stable prompt-cache affinity keys for account-pinned pool routing.

Upstream prompt caching on the AdaL proxy is per ACCOUNT, not per session:
replaying the same cacheable prefix on the same account hits
``cache_read_input_tokens`` (~10x cheaper) even across sessions. The pool
therefore pins each request's cacheable prefix to the account that served
it. That needs a key which is identical for identical prefixes across
processes, so it is a sha256 over canonical JSON — never ``hash()``, which
is salted per process, and never dict-iteration order, which ``sort_keys``
neutralises.
"""

from __future__ import annotations

import hashlib
import json

# Body keys whose mere presence means there is something cacheable to pin.
_AFFINITY_BODY_KEYS = ("system", "tools", "messages", "input")


def affinity_key(body: dict, protocol: str) -> str | None:
    """Return a deterministic cache-affinity key for ``body``, or ``None``.

    ``metadata.user_id`` wins when present (Claude Code sends a stable JSON
    blob containing its ``session_id`` there): everything after it in the
    prompt belongs to that session, so the account binding should survive
    any later prompt change. Otherwise the key hashes the cacheable prefix —
    ``system``, the sorted tool names, and the first message — so requests
    differing only in later messages stay pinned to one account. ``None``
    when the body carries nothing cacheable.
    """
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        user_id = metadata.get("user_id")
        if isinstance(user_id, str) and user_id:
            return "u:" + hashlib.sha256(user_id.encode()).hexdigest()[:32]

    if not any(key in body for key in _AFFINITY_BODY_KEYS):
        return None

    tools = body.get("tools")
    tool_names = (
        sorted(
            tool.get("name")
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        )
        if isinstance(tools, list)
        else []
    )

    digest = hashlib.sha256(
        json.dumps(
            [body.get("system"), tool_names, _first_message(body, protocol)],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    return digest[:32]


def _first_message(body: dict, protocol: str) -> object:
    """First message of the cacheable prefix.

    ``body["messages"][0]`` for anthropic/openai_chat; ``body["input"][0]``
    (list input) or ``body["input"]`` itself (bare-string input) for
    responses; ``None`` when neither exists.
    """
    if protocol == "responses":
        inp = body.get("input")
        if isinstance(inp, list) and inp:
            return inp[0]
        return inp
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        return messages[0]
    return None
