"""Contract tests for sub2api.compat.errors and sub2api.compat.affinity.

Every assertion below defends behaviour measured live against the AdaL cloud
proxy this session: the proxy answers rejected Anthropic requests with
HTTP 200 plus an ``event: error`` SSE frame (e.g. the literal
``Messages.stream() got an unexpected keyword argument 'context_management'``
with ``error.type == "invalid_request_error"``), so sub2api must extract,
classify, and re-emit that as a real failure. Affinity keys must be stable
across processes because upstream prompt caching is per ACCOUNT and the pool
routes on these keys.
"""

from __future__ import annotations

import hashlib

import pytest

from sub2api.compat.affinity import affinity_key
from sub2api.compat.errors import (
    anthropic_error,
    parse_error_frame,
    status_for_anthropic_error,
)

# The literal measured upstream failure, exactly as it arrived in the
# SSE error frame.
CONTEXT_MANAGEMENT_MESSAGE = (
    "Messages.stream() got an unexpected keyword argument 'context_management'"
)


def _error_frame(
    message: str = CONTEXT_MANAGEMENT_MESSAGE, err_type: str = "invalid_request_error"
) -> dict:
    return {"type": "error", "error": {"type": err_type, "message": message}}


# --------------------------------------------------------------------------
# parse_error_frame
# --------------------------------------------------------------------------


def test_parse_error_frame_extracts_measured_context_management_error():
    frames = [
        {
            "type": "message_start",
            "message": {"id": "msg_1", "usage": {"input_tokens": 10}},
        },
        _error_frame(),
    ]
    err = parse_error_frame(frames)
    assert err == {
        "type": "invalid_request_error",
        "message": CONTEXT_MANAGEMENT_MESSAGE,
    }


def test_parse_error_frame_returns_none_when_no_error_frame():
    frames = [
        {"type": "message_start", "message": {"id": "msg_1"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "message_stop"},
    ]
    assert parse_error_frame(frames) is None


def test_parse_error_frame_returns_none_for_empty_frame_list():
    assert parse_error_frame([]) is None


def test_parse_error_frame_malformed_frame_missing_error_returns_safe_default():
    assert parse_error_frame([{"type": "error"}]) == {
        "type": "api_error",
        "message": "upstream error",
    }


@pytest.mark.parametrize("bad_error", [None, "boom", 42, {"type": 1}, []])
def test_parse_error_frame_non_dict_or_empty_error_returns_safe_default(bad_error):
    frame = {"type": "error", "error": bad_error}
    assert parse_error_frame([frame]) == {
        "type": "api_error",
        "message": "upstream error",
    }


def test_parse_error_frame_tolerates_non_dict_frames():
    frames = ["not a frame", {"type": "message_start"}, _error_frame()]
    err = parse_error_frame(frames)  # type: ignore[arg-type]
    assert err is not None
    assert err["message"] == CONTEXT_MANAGEMENT_MESSAGE


def test_parse_error_frame_returns_first_error_frame():
    first = _error_frame(message="first failure", err_type="rate_limit_error")
    second = _error_frame(message="second failure")
    err = parse_error_frame([first, second])
    assert err == {"type": "rate_limit_error", "message": "first failure"}


@pytest.mark.parametrize(
    "frame",
    [
        {
            "type": "response.failed",
            "response": {
                "error": {"code": "rate_limit_exceeded", "message": "quota exhausted"}
            },
        },
        {"type": "error", "code": "rate_limit_exceeded", "message": "quota exhausted"},
    ],
)
def test_responses_error_shapes_keep_their_rate_limit_classification(frame):
    error = parse_error_frame([frame])
    assert error["code"] == "rate_limit_exceeded"
    assert error["message"] == "quota exhausted"
    assert status_for_anthropic_error(error["type"]) == 429


# --------------------------------------------------------------------------
# status_for_anthropic_error
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("err_type", "status"),
    [
        ("invalid_request_error", 400),
        ("authentication_error", 401),
        ("permission_error", 403),
        ("not_found_error", 404),
        ("rate_limit_error", 429),
        ("overloaded_error", 529),
    ],
)
def test_status_table_rows(err_type: str, status: int):
    assert status_for_anthropic_error(err_type) == status


@pytest.mark.parametrize(
    "err_type", [None, "", "internal_error", "api_error", "totally_unknown"]
)
def test_status_missing_empty_unknown_maps_to_502(err_type):
    assert status_for_anthropic_error(err_type) == 502


def test_measured_error_frame_maps_to_400():
    err = parse_error_frame([_error_frame()])
    assert err is not None
    assert status_for_anthropic_error(err["type"]) == 400


# --------------------------------------------------------------------------
# anthropic_error
# --------------------------------------------------------------------------


def test_anthropic_error_envelope_shape():
    assert anthropic_error("missing key") == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "missing key"},
    }


def test_anthropic_error_default_type_is_invalid_request():
    built = anthropic_error("missing key")
    assert built["error"]["type"] == "invalid_request_error"
    assert status_for_anthropic_error(built["error"]["type"]) == 400


def test_anthropic_error_with_explicit_type():
    built = anthropic_error("overloaded", err_type="overloaded_error")
    assert built == {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "overloaded"},
    }
    assert status_for_anthropic_error(built["error"]["type"]) == 529


def test_anthropic_error_round_trips_through_parse_error_frame():
    # The envelope sub2api emits downstream must itself be parseable as an
    # error frame upstream-style, closing the loop with parse_error_frame.
    frames = [anthropic_error("bad input", err_type="not_found_error")]
    parsed = parse_error_frame(frames)
    assert parsed == {"type": "not_found_error", "message": "bad input"}
    assert status_for_anthropic_error(parsed["type"]) == 404


# --------------------------------------------------------------------------
# affinity_key
# --------------------------------------------------------------------------


def _system_tools_body(
    system: str = "You are a coding agent.", tool_names: list[str] | None = None
) -> dict:
    tools = [
        {"name": name, "input_schema": {"type": "object"}}
        for name in (tool_names or ["Bash", "Read"])
    ]
    return {
        "system": system,
        "tools": tools,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ],
    }


def test_affinity_key_prefix_stable_when_only_last_message_differs():
    a = _system_tools_body()
    b = _system_tools_body()
    b["messages"] = b["messages"] + [
        {"role": "user", "content": [{"type": "text", "text": "one more turn"}]}
    ]
    # Same prefix (system, tools, first message), different tail -> same key.
    assert affinity_key(a, "anthropic") == affinity_key(b, "anthropic")


def test_affinity_key_differs_when_system_differs():
    a = _system_tools_body(system="system A")
    b = _system_tools_body(system="system B")
    assert affinity_key(a, "anthropic") != affinity_key(b, "anthropic")


def test_affinity_key_differs_when_tool_names_differ():
    a = _system_tools_body(tool_names=["Bash", "Read"])
    b = _system_tools_body(tool_names=["Bash", "Grep"])
    assert affinity_key(a, "anthropic") != affinity_key(b, "anthropic")


def test_affinity_key_tool_order_is_normalized():
    # Tool iteration order must not matter; the names are sorted before hashing.
    a = _system_tools_body(tool_names=["Bash", "Read", "Write"])
    b = _system_tools_body(tool_names=["Write", "Read", "Bash"])
    assert affinity_key(a, "anthropic") == affinity_key(b, "anthropic")


def test_affinity_key_metadata_user_id_takes_precedence():
    body = _system_tools_body()
    body["metadata"] = {"user_id": "cc-session-abc123"}
    with_meta = affinity_key(body, "anthropic")

    body_without = _system_tools_body()
    body_without["metadata"] = {
        "session_id": "other"
    }  # non-user_id metadata is ignored
    without_meta = affinity_key(body_without, "anthropic")

    assert with_meta.startswith("u:")
    assert with_meta != without_meta
    # Precedence: changing the whole cacheable prefix does not move the key.
    other = _system_tools_body(system="completely different")
    other["metadata"] = {"user_id": "cc-session-abc123"}
    assert affinity_key(other, "anthropic") == with_meta


def test_affinity_key_user_id_key_is_deterministic_sha256():
    body = _system_tools_body()
    body["metadata"] = {"user_id": "cc-session-abc123"}
    expected = "u:" + hashlib.sha256(b"cc-session-abc123").hexdigest()[:32]
    assert affinity_key(body, "anthropic") == expected


def test_affinity_key_none_for_empty_body():
    assert affinity_key({}, "anthropic") is None


def test_affinity_key_none_when_nothing_cacheable():
    assert affinity_key({"stream": True, "max_tokens": 64}, "anthropic") is None
    # metadata alone (no user_id, no cacheable keys) is still None.
    assert affinity_key({"metadata": {"session_id": "s1"}}, "anthropic") is None


def test_affinity_key_protocol_responses_uses_input():
    a = {
        "input": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ],
    }
    b = {
        "input": a["input"]
        + [{"role": "user", "content": [{"type": "text", "text": "tail"}]}],
    }
    assert affinity_key(a, "responses") == affinity_key(b, "responses")

    c = {
        "input": [{"role": "user", "content": [{"type": "text", "text": "different"}]}]
    }
    assert affinity_key(a, "responses") != affinity_key(c, "responses")


def test_affinity_key_protocol_responses_bare_string_input():
    a = {"input": "hello", "system": "sys"}
    b = {"input": "hello", "system": "sys"}
    assert affinity_key(a, "responses") == affinity_key(b, "responses")
    c = {"input": "goodbye", "system": "sys"}
    assert affinity_key(a, "responses") != affinity_key(c, "responses")


def test_affinity_key_first_message_only_for_prefix():
    # A differing first message changes the prefix -> different key.
    a = _system_tools_body()
    a["messages"][0] = {
        "role": "user",
        "content": [{"type": "text", "text": "changed"}],
    }
    b = _system_tools_body()
    assert affinity_key(a, "anthropic") != affinity_key(b, "anthropic")


def test_affinity_key_deterministic_across_reordered_dict_literals():
    import json

    body_a = {
        "max_tokens": 64,
        "model": "claude-sonnet-4-6",
        "system": "You are terse.",
        "tools": [
            {"name": "Bash", "input_schema": {"type": "object", "properties": {}}}
        ],
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    body_b = {
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"input_schema": {"properties": {}, "type": "object"}, "name": "Bash"}
        ],
        "system": "You are terse.",
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
    }
    ka = affinity_key(body_a, "anthropic")
    kb = affinity_key(body_b, "anthropic")
    assert ka == kb
    # And it is a 32-hex-char sha256 prefix — reproducible in any process.
    expected = json.dumps(
        ["You are terse.", ["Bash"], {"content": "hi", "role": "user"}],
        sort_keys=True,
        default=str,
    )
    assert ka == hashlib.sha256(expected.encode()).hexdigest()[:32]


def test_affinity_key_openai_chat_protocol_uses_messages():
    a = {"system": "s", "messages": [{"role": "user", "content": "hello"}]}
    b = {
        "system": "s",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "again"},
        ],
    }
    assert affinity_key(a, "openai_chat") == affinity_key(b, "openai_chat")


def test_affinity_key_hash_independent_of_message_dict_ordering():
    # sort_keys=True in json.dumps must neutralise inner-dict ordering too.
    a = {"system": "s", "messages": [{"content": "hi", "role": "user"}]}
    b = {"system": "s", "messages": [{"role": "user", "content": "hi"}]}
    assert affinity_key(a, "anthropic") == affinity_key(b, "anthropic")


def test_affinity_key_two_independent_calls_same_value():
    body = _system_tools_body()
    first = affinity_key(body, "anthropic")
    # Rebuild the same body from scratch to simulate a second process.
    rebuilt = _system_tools_body()
    assert affinity_key(rebuilt, "anthropic") == first
