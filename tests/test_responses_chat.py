"""Consumer-visible regressions for the Responses-to-Chat bridge."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from sub2api.server.responses_chat import (
    ResponsesBridgeError,
    chat_chunks_from_responses,
    chat_completion_from_response,
)


def _event(frame: dict[str, Any]) -> bytes:
    return (
        b"data: "
        + json.dumps(frame, ensure_ascii=False, separators=(",", ":")).encode()
        + b"\r\n\r\n"
    )


async def _stream(parts: list[bytes]) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


async def _collect(parts: list[bytes], *, include_usage: bool = False) -> list[bytes]:
    return [
        frame
        async for frame in chat_chunks_from_responses(
            _stream(parts), model="gpt-6-astra", include_usage=include_usage
        )
    ]


def _chat_frames(raw_frames: list[bytes]) -> tuple[list[dict[str, Any]], int]:
    frames: list[dict[str, Any]] = []
    done = 0
    for raw in raw_frames:
        assert raw.startswith(b"data: ") and raw.endswith(b"\n\n")
        data = raw[6:-2]
        if data == b"[DONE]":
            done += 1
        else:
            frames.append(json.loads(data))
    return frames, done


def _consume_chunks(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Model the Chat client's reconstruction, rather than bridge internals."""
    result: dict[str, Any] = {
        "content": "",
        "refusal": "",
        "reasoning_content": "",
        "tool_calls": {},
        "finish_reason": None,
        "usage": None,
    }
    for frame in frames:
        if frame.get("choices") == []:
            result["usage"] = frame.get("usage")
            continue
        for choice in frame["choices"]:
            delta = choice["delta"]
            result["content"] += delta.get("content", "")
            result["refusal"] += delta.get("refusal", "")
            result["reasoning_content"] += delta.get("reasoning_content", "")
            for tool_delta in delta.get("tool_calls", []):
                index = tool_delta["index"]
                tool = result["tool_calls"].setdefault(
                    index,
                    {"id": None, "type": None, "name": None, "arguments": ""},
                )
                if "id" in tool_delta:
                    tool["id"] = tool_delta["id"]
                if "type" in tool_delta:
                    tool["type"] = tool_delta["type"]
                function = tool_delta.get("function", {})
                if "name" in function:
                    tool["name"] = function["name"]
                tool["arguments"] += function.get("arguments", "")
            if choice["finish_reason"] is not None:
                result["finish_reason"] = choice["finish_reason"]
    return result


def test_non_stream_response_preserves_tool_call_ids_reasoning_and_usage():
    payload = {
        "id": "resp_source",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "check facts"}],
            },
            {
                "id": "fc_item_not_for_chat",
                "type": "function_call",
                "call_id": "call_real_id",
                "name": "lookup",
                "arguments": '{"query":"weather"}',
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "It is sunny."}],
            },
        ],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    }

    completion = chat_completion_from_response(payload, model="gpt-6-astra")

    choice = completion["choices"][0]
    message = choice["message"]
    assert completion["object"] == "chat.completion"
    assert completion["model"] == "gpt-6-astra"
    assert message["content"] == "It is sunny."
    assert message["reasoning_content"] == "check facts"
    assert message["tool_calls"] == [
        {
            "id": "call_real_id",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query":"weather"}'},
        }
    ]
    assert choice["finish_reason"] == "tool_calls"
    assert completion["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 3},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }


@pytest.mark.anyio
async def test_stream_reconstructs_interleaved_sparse_tools_after_reasoning():
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 4,
            "item": {"id": "rs_1", "type": "reasoning", "summary": []},
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 4,
            "summary_index": 0,
            "delta": "plan ",
        },
        {
            "type": "response.reasoning_summary_text.done",
            "output_index": 4,
            "summary_index": 0,
            "text": "plan δ",
        },
        {
            "type": "response.output_item.added",
            "output_index": 17,
            "item": {
                "id": "fc_item_alpha",
                "type": "function_call",
                "call_id": "call_alpha",
                "name": "weather",
                "arguments": "",
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 92,
            "item": {
                "id": "fc_item_beta",
                "type": "function_call",
                "call_id": "call_beta",
                "name": "units",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 17,
            "delta": '{"city":"',
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 92,
            "delta": '{"unit":"',
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 17,
            "delta": 'Paris"}',
        },
        {
            "type": "response.function_call_arguments.done",
            "output_index": 17,
            "arguments": '{"city":"Paris"}',
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 92,
            "delta": 'C"}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 92,
            "item": {
                "id": "fc_item_beta",
                "type": "function_call",
                "call_id": "call_beta",
                "name": "units",
                "arguments": '{"unit":"C"}',
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 120,
            "item": {"id": "msg_1", "type": "message", "content": []},
        },
        {
            "type": "response.output_text.delta",
            "output_index": 120,
            "content_index": 0,
            "delta": "café",
        },
        {
            "type": "response.output_text.done",
            "output_index": 120,
            "content_index": 0,
            "text": "café",
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "id": "rs_1",
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "plan δ"}],
                    },
                    {
                        "id": "fc_item_alpha",
                        "type": "function_call",
                        "call_id": "call_alpha",
                        "name": "weather",
                        "arguments": '{"city":"Paris"}',
                    },
                    {
                        "id": "fc_item_beta",
                        "type": "function_call",
                        "call_id": "call_beta",
                        "name": "units",
                        "arguments": '{"unit":"C"}',
                    },
                    {
                        "id": "msg_1",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "café"}],
                    },
                ],
            },
        },
    ]
    parts = [_event(event) for event in events]
    utf8 = "é".encode()
    text_event = parts[-2]
    split = text_event.index(utf8) + 1
    parts[-2:-1] = [text_event[:split], text_event[split:]]

    raw_frames = await _collect(parts)
    frames, done = _chat_frames(raw_frames)
    reconstructed = _consume_chunks(frames)

    assert done == 1
    assert reconstructed["reasoning_content"] == "plan δ"
    assert reconstructed["content"] == "café"
    assert reconstructed["tool_calls"] == {
        0: {
            "id": "call_alpha",
            "type": "function",
            "name": "weather",
            "arguments": '{"city":"Paris"}',
        },
        1: {
            "id": "call_beta",
            "type": "function",
            "name": "units",
            "arguments": '{"unit":"C"}',
        },
    }
    assert reconstructed["finish_reason"] == "tool_calls"
    assert len({frame["id"] for frame in frames}) == 1
    assert len({frame["created"] for frame in frames}) == 1
    assert {frame["model"] for frame in frames} == {"gpt-6-astra"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reason", "finish_reason"),
    [("max_output_tokens", "length"), ("content_filter", "content_filter")],
)
async def test_terminal_only_incomplete_response_preserves_content_refusal_and_reasoning(
    reason: str, finish_reason: str
):
    raw_frames = await _collect(
        [
            _event(
                {
                    "type": "response.incomplete",
                    "response": {
                        "status": "incomplete",
                        "incomplete_details": {"reason": reason},
                        "output": [
                            {
                                "type": "reasoning",
                                "summary": [
                                    {"type": "summary_text", "text": "safety review"}
                                ],
                            },
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": "partial"},
                                    {"type": "refusal", "refusal": "cannot continue"},
                                ],
                            },
                        ],
                    },
                }
            )
        ]
    )
    frames, done = _chat_frames(raw_frames)
    reconstructed = _consume_chunks(frames)

    assert done == 1
    assert reconstructed["content"] == "partial"
    assert reconstructed["refusal"] == "cannot continue"
    assert reconstructed["reasoning_content"] == "safety review"
    assert reconstructed["finish_reason"] == finish_reason


@pytest.mark.anyio
async def test_stream_emits_usage_only_when_requested_before_done():
    terminal = _event(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 2},
                    "output_tokens_details": {"reasoning_tokens": 1},
                },
            },
        }
    )

    without_usage, without_done = _chat_frames(await _collect([terminal]))
    with_raw = await _collect([terminal], include_usage=True)
    with_usage, with_done = _chat_frames(with_raw)

    assert without_done == 1
    assert all(frame["choices"] != [] for frame in without_usage)
    assert with_done == 1
    assert with_usage[-1]["choices"] == []
    assert with_usage[-1]["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
        "prompt_tokens_details": {"cached_tokens": 2},
        "completion_tokens_details": {"reasoning_tokens": 1},
    }
    assert with_raw[-2] != b"data: [DONE]\n\n"
    assert with_raw[-1] == b"data: [DONE]\n\n"


def test_non_stream_failed_and_in_progress_payloads_raise_openai_errors():
    with pytest.raises(ResponsesBridgeError) as failed:
        chat_completion_from_response(
            {
                "status": "failed",
                "error": {
                    "message": "upstream denied",
                    "type": "server_error",
                    "code": "denied",
                },
            },
            model="gpt-6-astra",
        )
    assert failed.value.error == {
        "message": "upstream denied",
        "type": "server_error",
        "code": "denied",
    }

    with pytest.raises(ResponsesBridgeError) as in_progress:
        chat_completion_from_response(
            {"status": "in_progress", "output": []}, model="gpt-6-astra"
        )
    assert (
        in_progress.value.error["message"]
        == "Responses upstream did not return a terminal response"
    )
    assert in_progress.value.error["type"] == "api_error"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("event", "expected_message", "expected_type", "expected_code"),
    [
        (
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {
                        "message": "provider unavailable",
                        "type": "server_error",
                        "code": "unavailable",
                    },
                },
            },
            "provider unavailable",
            "server_error",
            "unavailable",
        ),
        (
            {
                "type": "error",
                "error": {
                    "message": "connection reset",
                    "type": "api_error",
                    "code": "reset",
                },
            },
            "connection reset",
            "api_error",
            "reset",
        ),
    ],
)
async def test_stream_failures_emit_structured_error_without_successful_finish(
    event: dict[str, Any],
    expected_message: str,
    expected_type: str,
    expected_code: str,
):
    frames, _ = _chat_frames(await _collect([_event(event)]))

    assert frames == [
        {
            "error": {
                "message": expected_message,
                "type": expected_type,
                "code": expected_code,
            }
        }
    ]


@pytest.mark.anyio
async def test_truncated_stream_preserves_partial_output_without_successful_finish():
    frames, _ = _chat_frames(
        await _collect(
            [
                _event(
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "partial answer",
                    }
                ),
            ]
        )
    )
    choices = [choice for frame in frames for choice in frame.get("choices", [])]
    assert (
        "".join(choice["delta"].get("content", "") for choice in choices)
        == "partial answer"
    )
    assert all(choice["finish_reason"] is None for choice in choices)
    assert frames[-1]["error"]["type"] == "api_error"


class _CloseTrackingSource:
    def __init__(self) -> None:
        self._frames = [
            _event(
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "first",
                }
            ),
            _event(
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "first"}],
                            }
                        ],
                    },
                }
            ),
        ]
        self.closed = False

    def __aiter__(self) -> _CloseTrackingSource:
        return self

    async def __anext__(self) -> bytes:
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.pop(0)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_stream_closes_upstream_when_consumer_stops_early():
    source = _CloseTrackingSource()
    bridge = chat_chunks_from_responses(source, model="gpt-6-astra")

    await anext(bridge)
    await bridge.aclose()

    assert source.closed is True
