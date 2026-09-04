"""Tests for sub2api/compat/aggregate.py.

Each test defends an observable contract measured against real provider
streams: the reassembled envelope a client would receive, not internal
plumbing.  Frames below mirror what zai/minimax/xai (bare ``message_stop``),
api.anthropic.com (``message_stop.message`` fast path), OpenAI chat/responses
and DeepSeek actually emit.
"""

from __future__ import annotations

import json

from sub2api.compat.aggregate import (
    AnthropicAggregator,
    OpenAIChatAggregator,
    ResponsesAggregator,
    SSEDecoder,
)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_zai_style_stream_assembles_content_and_usage():
    # zai/minimax/xai style: bare message_stop, usage split across
    # message_start (input) and message_delta (output), ping frames present.
    agg = AnthropicAggregator()
    for frame in [
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "id": "msg_zai_1",
                "role": "assistant",
                "model": "glm-5.2",
                "usage": {"input_tokens": 120, "output_tokens": 1},
            },
        },
        {"type": "ping"},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hel"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "lo, "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]:
        agg.feed(frame)
    result = agg.result()
    assert result["id"] == "msg_zai_1"
    assert result["model"] == "glm-5.2"
    assert result["role"] == "assistant"
    assert result["type"] == "message"
    assert result["content"] == [{"type": "text", "text": "Hello, world"}]
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 120
    assert result["usage"]["output_tokens"] == 9
    usage = agg.usage()
    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] == 9


def test_anthropic_xai_style_output_tokens_only():
    # xai reports only output_tokens in message_delta.usage; input comes from
    # message_start.  The per-key merge must keep both.
    agg = AnthropicAggregator()
    agg.feed(
        {
            "type": "message_start",
            "message": {"id": "m", "usage": {"input_tokens": 55}},
        }
    )
    agg.feed(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 4},
        }
    )
    assert agg.usage() == {
        "input_tokens": 55,
        "output_tokens": 4,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }


def test_anthropic_tool_use_json_fragments():
    # input_json_delta arrives in three fragments; parsed at content_block_stop.
    agg = AnthropicAggregator()
    for frame in [
        {"type": "message_start", "message": {"id": "m", "usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "bash",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"comm'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 'and": "ls -la'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '"}'},
        },
        {"type": "content_block_stop", "index": 0},
    ]:
        agg.feed(frame)
    block = agg.result()["content"][0]
    assert block["name"] == "bash"
    assert block["input"] == {"command": "ls -la"}


def test_anthropic_thinking_and_signature_deltas():
    agg = AnthropicAggregator()
    for frame in [
        {
            "type": "message_start",
            "message": {"id": "m", "usage": {"input_tokens": 10}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "let me see"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sigABC"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    ]:
        agg.feed(frame)
    block = agg.result()["content"][0]
    assert block["thinking"] == "hmm let me see"
    assert block["signature"] == "sigABC"


def test_anthropic_message_stop_with_message_used_verbatim():
    # api.anthropic.com fast path: terminal frame carries the full message.
    final_message = {
        "type": "message",
        "id": "msg_full",
        "role": "assistant",
        "content": [{"type": "text", "text": "complete"}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    agg = AnthropicAggregator()
    agg.feed({"type": "message_start", "message": {"id": "msg_partial", "usage": {}}})
    agg.feed({"type": "message_stop", "message": final_message})
    assert agg.result() == final_message


def test_anthropic_usage_cache_token_name_mapping():
    agg = AnthropicAggregator()
    agg.feed(
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "usage": {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 210,
                    "cache_read_input_tokens": 320,
                },
            },
        }
    )
    assert agg.usage() == {
        "input_tokens": 100,
        "output_tokens": 0,
        "cache_read_tokens": 320,
        "cache_write_tokens": 210,
        "reasoning_tokens": 0,
    }


def test_anthropic_unknown_frames_ignored_and_empty_stream_safe():
    agg = AnthropicAggregator()
    agg.feed({"type": "ping"})
    agg.feed({"type": "totally_unknown", "payload": {"x": 1}})
    result = agg.result()
    assert result["type"] == "message"
    assert result["content"] == []
    assert agg.usage()["input_tokens"] == 0


def test_anthropic_invalid_json_buffer_becomes_empty_input():
    agg = AnthropicAggregator()
    agg.feed({"type": "message_start", "message": {"id": "m", "usage": {}}})
    agg.feed(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t", "name": "f", "input": {}},
        }
    )
    agg.feed(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"broken": '},
        }
    )
    agg.feed({"type": "content_block_stop", "index": 0})
    assert agg.result()["content"][0]["input"] == {}


# ---------------------------------------------------------------------------
# OpenAI chat completions
# ---------------------------------------------------------------------------


def test_openai_chat_trailing_usage_chunk_with_cached_tokens():
    agg = OpenAIChatAggregator()
    for frame in [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-5.6-luna",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hi "},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-5.6-luna",
            "choices": [
                {"index": 0, "delta": {"content": "there"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-5.6-luna",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-5.6-luna",
            "choices": [],
            "usage": {
                "prompt_tokens": 9009,
                "completion_tokens": 4,
                "total_tokens": 9013,
                "prompt_tokens_details": {"cached_tokens": 9006},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ]:
        agg.feed(frame)
    result = agg.result()
    assert result["id"] == "chatcmpl-1"
    assert result["object"] == "chat.completion"
    assert result["model"] == "gpt-5.6-luna"
    assert result["choices"][0]["message"]["content"] == "Hi there"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 9009
    usage = agg.usage()
    assert usage == {
        "input_tokens": 9009,
        "output_tokens": 4,
        "cache_read_tokens": 9006,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }


def test_openai_chat_tool_calls_concatenate_in_index_order():
    agg = OpenAIChatAggregator()
    for frame in [
        {
            "id": "c2",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5.6-terra",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "c2",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5.6-terra",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"city": "Ber'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "c2",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5.6-terra",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'lin"}'}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]:
        agg.feed(frame)
    result = agg.result()
    message = result["choices"][0]["message"]
    assert message["tool_calls"] == [
        {
            "id": "call_a",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Berlin"}'},
        }
    ]
    assert result["choices"][0]["finish_reason"] == "tool_calls"


def test_openai_chat_deepseek_cache_hit_tokens():
    # DeepSeek reports prompt_cache_hit_tokens / prompt_cache_miss_tokens at
    # the top level of usage instead of prompt_tokens_details.
    agg = OpenAIChatAggregator()
    agg.feed(
        {
            "id": "d1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-chat",
            "choices": [
                {"index": 0, "delta": {"content": "ok"}, "finish_reason": None}
            ],
        }
    )
    agg.feed(
        {
            "id": "d1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-chat",
            "choices": [],
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 10,
                "prompt_cache_hit_tokens": 400,
                "prompt_cache_miss_tokens": 100,
            },
        }
    )
    usage = agg.usage()
    assert usage["input_tokens"] == 500
    assert usage["output_tokens"] == 10
    assert usage["cache_read_tokens"] == 400
    assert usage["cache_write_tokens"] == 0


def test_openai_chat_reasoning_content_accumulates():
    agg = OpenAIChatAggregator()
    agg.feed(
        {
            "id": "r1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "think"},
                    "finish_reason": None,
                }
            ],
        }
    )
    agg.feed(
        {
            "id": "r1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "ing"},
                    "finish_reason": None,
                }
            ],
        }
    )
    result = agg.result()
    assert result["choices"][0]["message"]["reasoning_content"] == "thinking"


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def test_responses_completed_frame_returned_verbatim():
    response = {
        "id": "resp_1",
        "object": "response",
        "model": "gpt-5.6-luna",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "final answer"}],
            }
        ],
        "usage": {
            "input_tokens": 88,
            "output_tokens": 6,
            "input_tokens_details": {"cached_tokens": 40},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    }
    agg = ResponsesAggregator()
    agg.feed({"type": "response.created", "response": {"id": "resp_1"}})
    agg.feed({"type": "response.completed", "response": response})
    assert agg.result() is response
    usage = agg.usage()
    assert usage == {
        "input_tokens": 88,
        "output_tokens": 6,
        "cache_read_tokens": 40,
        "cache_write_tokens": 0,
        "reasoning_tokens": 2,
    }


def test_responses_assembled_from_incremental_events():
    agg = ResponsesAggregator()
    for frame in [
        {
            "type": "response.created",
            "response": {"id": "resp_2", "model": "gpt-5.6-sol"},
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "content": []},
        },
        {"type": "response.output_text.delta", "output_index": 0, "delta": "an"},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "swer"},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            },
        },
    ]:
        agg.feed(frame)
    result = agg.result()
    assert result["id"] == "resp_2"
    assert result["model"] == "gpt-5.6-sol"
    assert result["output"][0]["content"][0]["text"] == "answer"


# ---------------------------------------------------------------------------
# SSEDecoder
# ---------------------------------------------------------------------------


def test_sse_decoder_reassembles_json_split_mid_token():
    payload = json.dumps(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        }
    )
    split = len(payload) // 2
    decoder = SSEDecoder()
    assert decoder.feed(f"data: {payload[:split]}".encode()) == []
    frames = decoder.feed(f"{payload[split:]}\n\n".encode())
    assert len(frames) == 1
    assert frames[0]["delta"]["text"] == "hi"
    assert decoder.flush() == []


def test_sse_decoder_skips_done_comments_and_event_lines():
    raw = (
        b"event: message_start\n"
        b'data: {"type": "ping"}\n'
        b"\n"
        b": keep-alive comment\n"
        b"retry: 5000\n"
        b"id: 7\n"
        b"\n"
        b"data: [DONE]\n"
        b"\n"
        b'data: {"type": "message_stop"}\n'
    )
    frames = SSEDecoder().feed(raw)
    assert frames == [{"type": "ping"}, {"type": "message_stop"}]


def test_sse_decoder_skips_unparseable_json():
    raw = b'data: {not json}\ndata: {"ok": true}\n'
    frames = SSEDecoder().feed(raw)
    assert frames == [{"ok": True}]


def test_sse_decoder_flushes_trailing_partial_line():
    decoder = SSEDecoder()
    assert decoder.feed(b'data: {"a": 1}') == []
    frames = decoder.flush()
    assert frames == [{"a": 1}]


def test_sse_decoder_handles_crlf_and_split_headers():
    decoder = SSEDecoder()
    assert decoder.feed(b'event: error\r\ndata: {"er') == []
    frames = decoder.feed(b'ror": {"type": "x"}}\r\n\r\n')
    assert frames == [{"error": {"type": "x"}}]
