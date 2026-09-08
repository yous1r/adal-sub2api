"""Protocol-boundary tests for Chat-to-Responses request conversion."""

from __future__ import annotations

from copy import deepcopy

import pytest

from sub2api.compat.chat_responses import ChatRequestError, chat_request_to_responses


def test_function_history_keeps_order_and_chat_call_ids() -> None:
    translated = chat_request_to_responses(
        {
            "model": "gpt-6-astra",
            "messages": [
                {"role": "system", "content": "Follow the policy."},
                {"role": "developer", "content": "Use tools carefully."},
                {"role": "user", "content": "Inspect both files."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_first",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":"a"}'},
                        },
                        {
                            "id": "call_second",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":"b"}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_first", "content": "alpha"},
                {"role": "tool", "tool_call_id": "call_second", "content": "beta"},
                {"role": "assistant", "content": "Both files are present."},
                {"role": "user", "content": "Summarize them."},
            ],
        }
    )

    items = translated["input"]
    assert [item.get("role") or item["type"] for item in items] == [
        "system",
        "developer",
        "user",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
        "assistant",
        "user",
    ]
    assert [item["call_id"] for item in items[3:7]] == [
        "call_first",
        "call_second",
        "call_first",
        "call_second",
    ]
    assert items[3]["arguments"] == '{"path":"a"}'
    assert items[6]["output"] == "beta"


def test_assistant_history_uses_responses_output_content_parts() -> None:
    translated = chat_request_to_responses(
        {
            "model": "gpt-6-astra",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First answer."},
                        {"type": "input_text", "text": "Second answer."},
                        {"type": "output_text", "text": "Third answer."},
                    ],
                },
                {"role": "user", "content": "Continue."},
            ],
        }
    )

    assert translated["input"][0] == {
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "First answer."},
            {"type": "output_text", "text": "Second answer."},
            {"type": "output_text", "text": "Third answer."},
        ],
    }


@pytest.mark.parametrize(
    "assistant_content",
    [
        {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
        {"type": "file", "file": {"file_id": "file_report"}},
    ],
)
def test_assistant_history_rejects_input_only_content_parts(
    assistant_content: dict[str, object],
) -> None:
    with pytest.raises(ChatRequestError) as caught:
        chat_request_to_responses(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "assistant", "content": [assistant_content]},
                    {"role": "user", "content": "Continue."},
                ],
            }
        )

    assert caught.value.param == "messages[0].content[0].type"


def test_mixed_content_and_typed_tool_result_use_responses_parts() -> None:
    translated = chat_request_to_responses(
        {
            "model": "gpt-6-astra",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare this screenshot."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.test/image.png",
                                "detail": "low",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_render",
                            "type": "function",
                            "function": {"name": "render", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_render",
                    "content": [
                        {"type": "text", "text": "Rendered result"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AA==",
                        },
                        {
                            "type": "file",
                            "file": {
                                "file_id": "file_report",
                                "filename": "report.pdf",
                            },
                        },
                    ],
                },
            ],
        }
    )

    assert translated["input"][0]["content"] == [
        {"type": "input_text", "text": "Compare this screenshot."},
        {
            "type": "input_image",
            "image_url": "https://example.test/image.png",
            "detail": "low",
        },
    ]
    assert translated["input"][1] == {
        "type": "function_call",
        "call_id": "call_render",
        "name": "render",
        "arguments": "{}",
    }
    assert translated["input"][2]["output"] == [
        {"type": "input_text", "text": "Rendered result"},
        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
        {"type": "input_file", "file_id": "file_report", "filename": "report.pdf"},
    ]


def test_reasoning_tokens_and_shared_controls_preserve_precedence_without_mutation() -> (
    None
):
    body = {
        "model": "gpt-6-astra",
        "messages": [{"role": "user", "content": "solve it"}],
        "max_tokens": 128,
        "max_completion_tokens": 256,
        "max_output_tokens": 512,
        "reasoning": {"summary": "detailed"},
        "thinking_effort": "high",
        "store": False,
        "metadata": {"request": "r_123"},
        "service_tier": "flex",
        "safety_identifier": "user_hash",
        "prompt_cache_key": "stable-prefix",
    }
    original = deepcopy(body)

    translated = chat_request_to_responses(body)

    assert translated["max_output_tokens"] == 512
    assert translated["reasoning"] == {"summary": "detailed", "effort": "high"}
    assert translated["store"] is False
    assert translated["metadata"] == {"request": "r_123"}
    assert translated["service_tier"] == "flex"
    assert translated["safety_identifier"] == "user_hash"
    assert translated["prompt_cache_key"] == "stable-prefix"
    assert body == original


def test_function_schema_named_choice_and_structured_output_keep_chat_semantics() -> (
    None
):
    translated = chat_request_to_responses(
        {
            "model": "gpt-6-astra",
            "messages": [{"role": "user", "content": "Get the forecast."}],
            "parallel_tool_calls": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get the local forecast.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "weather"}},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "forecast_answer",
                    "description": "A forecast response.",
                    "schema": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                    },
                    "strict": True,
                },
            },
            "verbosity": "high",
        }
    )

    tool = translated["tools"][0]
    assert tool["type"] == "function"
    assert tool["name"] == "weather"
    assert tool["description"] == "Get the local forecast."
    assert tool["parameters"]["required"] == ["city"]
    assert tool["strict"] is False
    assert translated["parallel_tool_calls"] is True
    assert translated["tool_choice"] == {"type": "function", "name": "weather"}
    assert translated["text"] == {
        "format": {
            "type": "json_schema",
            "name": "forecast_answer",
            "description": "A forecast response.",
            "schema": {"type": "object", "properties": {"summary": {"type": "string"}}},
            "strict": True,
        },
        "verbosity": "high",
    }


@pytest.mark.parametrize(
    ("body", "param"),
    [
        (
            {
                "model": "gpt-6-astra",
                "n": 2,
                "messages": [{"role": "user", "content": "choose twice"}],
            },
            "n",
        ),
        (
            {
                "model": "gpt-6-astra",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": "not-an-image-object"}
                        ],
                    }
                ],
            },
            "messages[0].content[0].image_url",
        ),
    ],
)
def test_unrepresentable_or_malformed_requests_fail_at_the_affected_parameter(
    body: dict[str, object], param: str
) -> None:
    with pytest.raises(ChatRequestError) as caught:
        chat_request_to_responses(body)

    assert caught.value.param == param
    assert param in str(caught.value)


def test_refusal_history_uses_native_assistant_content_part() -> None:
    refusal = "I cannot provide that information."
    translated = chat_request_to_responses(
        {
            "model": "gpt-6-astra",
            "messages": [
                {"role": "assistant", "content": None, "refusal": refusal},
                {"role": "user", "content": "Then explain the safe alternative."},
            ],
        }
    )
    assert translated["input"][0] == {
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": refusal}],
    }
    assert translated["input"][1]["role"] == "user"


def test_stop_sequences_are_rejected_instead_of_lost_or_forwarded():
    with pytest.raises(ChatRequestError) as failure:
        chat_request_to_responses(
            {
                "model": "gpt-6-astra",
                "messages": [{"role": "user", "content": "Continue until the marker."}],
                "stop": ["<end>"],
            }
        )
    assert failure.value.param == "stop"
