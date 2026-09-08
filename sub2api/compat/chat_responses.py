"""Loss-aware conversion from Chat Completions requests to Responses input.

The two APIs share most generation controls but model conversation state
separately.  This module converts only wire shapes with an exact Responses
counterpart and rejects the rest at the Chat boundary instead of dropping a
caller-controlled part of a conversation.
"""

from __future__ import annotations

from typing import Any


class ChatRequestError(ValueError):
    """A Chat request cannot be represented faithfully by Responses."""

    def __init__(self, message: str, param: str) -> None:
        super().__init__(message)
        self.param = param


_SUPPORTED_FIELDS = frozenset(
    {
        "messages",
        "model",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "stop",
        "metadata",
        "moderation",
        "store",
        "service_tier",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "user",
        "parallel_tool_calls",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "response_format",
        "verbosity",
        "reasoning",
        "reasoning_effort",
        "thinking_effort",
        "max_output_tokens",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "audio",
        "modalities",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "seed",
        "prediction",
        "web_search_options",
    }
)

_PASSTHROUGH_FIELDS = (
    "model",
    "stream",
    "temperature",
    "top_p",
    "metadata",
    "moderation",
    "store",
    "service_tier",
    "safety_identifier",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "user",
    "parallel_tool_calls",
)

_UNREPRESENTABLE_CONTROLS = frozenset(
    {
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "seed",
        "prediction",
        "web_search_options",
    }
)


def chat_request_to_responses(body: dict[str, Any]) -> dict[str, Any]:
    """Return a Responses request for a semantically equivalent Chat request.

    The returned dict is newly assembled.  Nested schemas and untouched
    pass-through values are reused rather than copied, but they are never
    modified.  ``stream_options.include_usage`` is intentionally consumed by
    the Chat response bridge, where it controls the trailing Chat usage chunk;
    Responses has no equivalent request option.
    """
    _validate_top_level(body)

    model = body.get("model")
    if not isinstance(model, str) or not model:
        _raise("model", "must be a non-empty string")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        _raise("messages", "must be a non-empty array")

    _validate_choice_count(body)
    _validate_modalities(body)
    _reject_unrepresentable_controls(body)

    out: dict[str, Any] = {"model": model}
    for field in _PASSTHROUGH_FIELDS:
        if field != "model" and field in body:
            out[field] = body[field]

    out["input"] = _translate_messages(messages)
    _translate_max_tokens(body, out)
    _translate_reasoning(body, out)
    _translate_tools(body, out)
    _translate_text_config(body, out)
    _translate_stream_options(body, out)
    return out


def _validate_top_level(body: dict[str, Any]) -> None:
    for field in body:
        if field not in _SUPPORTED_FIELDS:
            _raise(
                field, "is not supported when bridging Chat Completions to Responses"
            )


def _validate_choice_count(body: dict[str, Any]) -> None:
    if "n" not in body or body["n"] is None:
        return
    count = body["n"]
    if isinstance(count, int) and not isinstance(count, bool) and count == 1:
        return
    _raise("n", "must be 1 because Responses produces one completion")


def _validate_modalities(body: dict[str, Any]) -> None:
    if body.get("audio") is not None:
        _raise("audio", "audio output cannot be represented by Chat response bridging")
    if "modalities" not in body or body["modalities"] is None:
        return
    modalities = body["modalities"]
    if modalities == ["text"]:
        return
    _raise(
        "modalities", "only text output can be represented by Chat response bridging"
    )


def _reject_unrepresentable_controls(body: dict[str, Any]) -> None:
    for field in _UNREPRESENTABLE_CONTROLS:
        if body.get(field) is not None:
            _raise(field, "has no equivalent in the Responses API")


def _translate_messages(messages: list[Any]) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    pending_call_ids: set[str] = set()

    for index, raw_message in enumerate(messages):
        param = f"messages[{index}]"
        if not isinstance(raw_message, dict):
            _raise(param, "must be an object")
        role = raw_message.get("role")
        if role in {"system", "developer", "user"}:
            _reject_message_fields(raw_message, param, {"role", "content", "name"})
            _reject_name(raw_message, param)
            translated.append(
                {
                    "role": role,
                    "content": _translate_content(
                        raw_message.get("content"), f"{param}.content"
                    ),
                }
            )
            continue
        if role == "assistant":
            _reject_message_fields(
                raw_message,
                param,
                {
                    "role",
                    "content",
                    "name",
                    "tool_calls",
                    "function_call",
                    "audio",
                    "refusal",
                    "reasoning_content",
                    "reasoning",
                },
            )
            _reject_name(raw_message, param)
            # Chat summaries are readable metadata, not replayable Responses
            # reasoning items. Accept our own output without forging item IDs
            # or encrypted state; effort remains a separate request setting.
            for field in ("reasoning_content", "reasoning"):
                if raw_message.get(field) is not None:
                    _required_string(
                        raw_message[field], f"{param}.{field}", allow_empty=True
                    )
            if raw_message.get("audio") is not None:
                _raise(
                    f"{param}.audio",
                    "assistant audio history cannot be represented by Responses",
                )
            if raw_message.get("function_call") is not None:
                _raise(
                    f"{param}.function_call",
                    "legacy assistant function calls lack the call ID required by Responses",
                )
            calls = _translate_tool_calls(
                raw_message.get("tool_calls"),
                f"{param}.tool_calls",
                seen_call_ids,
                pending_call_ids,
            )
            content = raw_message.get("content")
            refusal = raw_message.get("refusal")
            converted_content: list[dict[str, Any]] = []
            if content is not None:
                if content != [] or refusal is None:
                    converted_content = _translate_assistant_content(
                        content, f"{param}.content"
                    )
            if refusal is not None:
                converted_content.append(
                    {
                        "type": "refusal",
                        "refusal": _required_string(
                            refusal, f"{param}.refusal", allow_empty=True
                        ),
                    }
                )
            if converted_content:
                translated.append({"role": "assistant", "content": converted_content})
            elif not calls:
                _raise(
                    f"{param}.content",
                    "is required when assistant message has no tool calls",
                )
            translated.extend(calls)
            continue
        if role == "tool":
            _reject_message_fields(
                raw_message, param, {"role", "content", "tool_call_id"}
            )
            call_id = _required_string(
                raw_message.get("tool_call_id"), f"{param}.tool_call_id"
            )
            if call_id not in pending_call_ids:
                _raise(
                    f"{param}.tool_call_id",
                    "must reference an earlier, unanswered assistant tool call",
                )
            pending_call_ids.remove(call_id)
            translated.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _translate_content(
                        raw_message.get("content"), f"{param}.content"
                    ),
                }
            )
            continue
        _raise(f"{param}.role", "must be system, developer, user, assistant, or tool")
    return translated


def _reject_message_fields(
    message: dict[str, Any], param: str, allowed: set[str]
) -> None:
    for field, value in message.items():
        if field not in allowed and value is not None:
            _raise(
                f"{param}.{field}",
                "is not supported when bridging Chat history to Responses",
            )


def _reject_name(message: dict[str, Any], param: str) -> None:
    if message.get("name") is not None:
        _raise(f"{param}.name", "has no equivalent on a Responses input message")


def _translate_tool_calls(
    raw_calls: Any,
    param: str,
    seen_call_ids: set[str],
    pending_call_ids: set[str],
) -> list[dict[str, Any]]:
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, list):
        _raise(param, "must be an array")

    translated: list[dict[str, Any]] = []
    for index, raw_call in enumerate(raw_calls):
        call_param = f"{param}[{index}]"
        if not isinstance(raw_call, dict):
            _raise(call_param, "must be an object")
        _reject_unexpected_fields(raw_call, call_param, {"id", "type", "function"})
        if raw_call.get("type") != "function":
            _raise(
                f"{call_param}.type",
                "only function tool calls can be represented by Responses",
            )
        call_id = _required_string(raw_call.get("id"), f"{call_param}.id")
        if call_id in seen_call_ids:
            _raise(f"{call_param}.id", "must be unique across the Chat history")
        function = raw_call.get("function")
        if not isinstance(function, dict):
            _raise(f"{call_param}.function", "must be an object")
        _reject_unexpected_fields(
            function, f"{call_param}.function", {"name", "arguments"}
        )
        translated.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": _required_string(
                    function.get("name"), f"{call_param}.function.name"
                ),
                "arguments": _required_string(
                    function.get("arguments"),
                    f"{call_param}.function.arguments",
                    allow_empty=True,
                ),
            }
        )
        seen_call_ids.add(call_id)
        pending_call_ids.add(call_id)
    return translated


def _translate_assistant_content(content: Any, param: str) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "output_text", "text": content}]
    if not isinstance(content, list):
        _raise(param, "must be a string or an array of supported content parts")
    if not content:
        _raise(param, "must contain at least one supported content part")

    translated: list[dict[str, Any]] = []
    for index, raw_part in enumerate(content):
        part_param = f"{param}[{index}]"
        if not isinstance(raw_part, dict):
            _raise(part_param, "must be an object")
        part_type = raw_part.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            _reject_unexpected_fields(
                raw_part, part_param, {"type", "text", "prompt_cache_breakpoint"}
            )
            translated.append(_output_text_part(raw_part, part_param))
        elif part_type == "refusal":
            _reject_unexpected_fields(raw_part, part_param, {"type", "refusal"})
            translated.append(
                {
                    "type": "refusal",
                    "refusal": _required_string(
                        raw_part.get("refusal"),
                        f"{part_param}.refusal",
                        allow_empty=True,
                    ),
                }
            )
        else:
            _raise(
                part_param + ".type",
                "is not a supported assistant text or refusal content type",
            )
    return translated


def _output_text_part(part: dict[str, Any], param: str) -> dict[str, Any]:
    translated = {
        "type": "output_text",
        "text": _required_string(part.get("text"), f"{param}.text", allow_empty=True),
    }
    _copy_cache_breakpoint(part, translated, param)
    return translated


def _translate_content(content: Any, param: str) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        _raise(param, "must be a string or an array of supported content parts")
    if not content:
        _raise(param, "must contain at least one supported content part")

    translated: list[dict[str, Any]] = []
    for index, raw_part in enumerate(content):
        part_param = f"{param}[{index}]"
        if not isinstance(raw_part, dict):
            _raise(part_param, "must be an object")
        part_type = raw_part.get("type")
        if part_type in {"text", "input_text"}:
            _reject_unexpected_fields(
                raw_part, part_param, {"type", "text", "prompt_cache_breakpoint"}
            )
            translated.append(_input_text_part(raw_part, part_param))
        elif part_type == "image_url":
            _reject_unexpected_fields(
                raw_part, part_param, {"type", "image_url", "prompt_cache_breakpoint"}
            )
            translated.append(_chat_image_part(raw_part, part_param))
        elif part_type == "input_image":
            _reject_unexpected_fields(
                raw_part,
                part_param,
                {"type", "image_url", "file_id", "detail", "prompt_cache_breakpoint"},
            )
            translated.append(_input_image_part(raw_part, part_param))
        elif part_type == "file":
            _reject_unexpected_fields(raw_part, part_param, {"type", "file"})
            file = raw_part.get("file")
            if not isinstance(file, dict):
                _raise(f"{part_param}.file", "must be an object")
            _reject_unexpected_fields(
                file, f"{part_param}.file", {"file_data", "file_id", "filename"}
            )
            translated.append(_input_file_part(file, f"{part_param}.file"))
        elif part_type == "refusal":
            _reject_unexpected_fields(raw_part, part_param, {"type", "refusal"})
            translated.append(
                {
                    "type": "input_text",
                    "text": _required_string(
                        raw_part.get("refusal"),
                        f"{part_param}.refusal",
                        allow_empty=True,
                    ),
                }
            )
        elif part_type == "input_file":
            _reject_unexpected_fields(
                raw_part,
                part_param,
                {
                    "type",
                    "file_data",
                    "file_id",
                    "file_url",
                    "filename",
                    "detail",
                    "prompt_cache_breakpoint",
                },
            )
            translated.append(_input_file_part(raw_part, part_param))
        else:
            _raise(
                part_param + ".type",
                "is not a supported text, image, or file content type",
            )
    return translated


def _input_text_part(part: dict[str, Any], param: str) -> dict[str, Any]:
    translated = {
        "type": "input_text",
        "text": _required_string(part.get("text"), f"{param}.text", allow_empty=True),
    }
    _copy_cache_breakpoint(part, translated, param)
    return translated


def _chat_image_part(part: dict[str, Any], param: str) -> dict[str, Any]:
    image_url = part.get("image_url")
    if not isinstance(image_url, dict):
        _raise(f"{param}.image_url", "must be an object with a URL")
    _reject_unexpected_fields(image_url, f"{param}.image_url", {"url", "detail"})
    translated = {
        "type": "input_image",
        "image_url": _required_string(image_url.get("url"), f"{param}.image_url.url"),
    }
    if "detail" in image_url:
        translated["detail"] = image_url["detail"]
    _copy_cache_breakpoint(part, translated, param)
    return translated


def _input_image_part(part: dict[str, Any], param: str) -> dict[str, Any]:
    image_url = part.get("image_url")
    file_id = part.get("file_id")
    if not (isinstance(image_url, str) and image_url) and not (
        isinstance(file_id, str) and file_id
    ):
        _raise(f"{param}.image_url", "or file_id must identify the input image")
    translated: dict[str, Any] = {"type": "input_image"}
    for field in ("image_url", "file_id", "detail"):
        if field in part:
            translated[field] = part[field]
    _copy_cache_breakpoint(part, translated, param)
    return translated


def _input_file_part(part: dict[str, Any], param: str) -> dict[str, Any]:
    if not any(
        isinstance(part.get(field), str) and part[field]
        for field in ("file_data", "file_id", "file_url")
    ):
        _raise(f"{param}.file_id", "file_data, file_id, or file_url is required")
    translated: dict[str, Any] = {"type": "input_file"}
    for field in ("file_data", "file_id", "file_url", "filename", "detail"):
        if field in part:
            translated[field] = part[field]
    _copy_cache_breakpoint(part, translated, param)
    return translated


def _copy_cache_breakpoint(
    source: dict[str, Any], target: dict[str, Any], param: str
) -> None:
    if "prompt_cache_breakpoint" not in source:
        return
    breakpoint = source["prompt_cache_breakpoint"]
    if not isinstance(breakpoint, dict):
        _raise(f"{param}.prompt_cache_breakpoint", "must be an object")
    target["prompt_cache_breakpoint"] = breakpoint


def _translate_max_tokens(body: dict[str, Any], out: dict[str, Any]) -> None:
    for field in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        if body.get(field) is not None:
            out["max_output_tokens"] = body[field]
            return


def _translate_reasoning(body: dict[str, Any], out: dict[str, Any]) -> None:
    raw_reasoning = body.get("reasoning")
    if raw_reasoning is not None and not isinstance(raw_reasoning, dict):
        _raise("reasoning", "must be an object or null")

    effort = _flat_reasoning_effort(body)
    if raw_reasoning is None:
        if effort is not None:
            out["reasoning"] = {"effort": effort}
        elif "reasoning" in body:
            out["reasoning"] = None
        return

    reasoning = dict(raw_reasoning)
    if "effort" not in reasoning and effort is not None:
        reasoning["effort"] = effort
    out["reasoning"] = reasoning


def _flat_reasoning_effort(body: dict[str, Any]) -> Any:
    efforts = [
        (field, body[field])
        for field in ("reasoning_effort", "thinking_effort")
        if body.get(field) is not None
    ]
    if not efforts:
        return None
    first_field, first_effort = efforts[0]
    if not isinstance(first_effort, str):
        _raise(first_field, "must be a reasoning effort string")
    for field, effort in efforts[1:]:
        if not isinstance(effort, str) or effort != first_effort:
            _raise(
                field, "must match reasoning_effort when both effort aliases are set"
            )
    return first_effort


def _translate_tools(body: dict[str, Any], out: dict[str, Any]) -> None:
    raw_tools = body.get("tools")
    raw_functions = body.get("functions")
    if raw_tools is not None and raw_functions is not None:
        _raise("functions", "cannot be combined with tools")

    if raw_tools is not None:
        if not isinstance(raw_tools, list):
            _raise("tools", "must be an array")
        out["tools"] = [
            _translate_function_tool(tool, f"tools[{index}]")
            for index, tool in enumerate(raw_tools)
        ]
    elif raw_functions is not None:
        if not isinstance(raw_functions, list):
            _raise("functions", "must be an array")
        out["tools"] = [
            _translate_function_definition(function, f"functions[{index}]")
            for index, function in enumerate(raw_functions)
        ]

    raw_tool_choice = body.get("tool_choice")
    raw_function_choice = body.get("function_call")
    if raw_tool_choice is not None and raw_function_choice is not None:
        _raise("function_call", "cannot be combined with tool_choice")
    if raw_tool_choice is not None:
        out["tool_choice"] = _translate_tool_choice(raw_tool_choice, "tool_choice")
    elif raw_function_choice is not None:
        out["tool_choice"] = _translate_function_choice(
            raw_function_choice, "function_call"
        )


def _translate_function_tool(raw_tool: Any, param: str) -> dict[str, Any]:
    if not isinstance(raw_tool, dict):
        _raise(param, "must be an object")
    _reject_unexpected_fields(raw_tool, param, {"type", "function"})
    if raw_tool.get("type") != "function":
        _raise(f"{param}.type", "only function tools can be represented by Responses")
    return _translate_function_definition(raw_tool.get("function"), f"{param}.function")


def _translate_function_definition(raw_function: Any, param: str) -> dict[str, Any]:
    if not isinstance(raw_function, dict):
        _raise(param, "must be an object")
    _reject_unexpected_fields(
        raw_function, param, {"name", "description", "parameters", "strict"}
    )
    translated: dict[str, Any] = {
        "type": "function",
        "name": _required_string(raw_function.get("name"), f"{param}.name"),
        "parameters": raw_function.get("parameters") or {},
        "strict": raw_function.get("strict")
        if raw_function.get("strict") is not None
        else False,
    }
    if (
        "parameters" in raw_function
        and raw_function["parameters"] is not None
        and not isinstance(raw_function["parameters"], dict)
    ):
        _raise(f"{param}.parameters", "must be an object or null")
    if "description" in raw_function:
        description = raw_function["description"]
        if description is not None and not isinstance(description, str):
            _raise(f"{param}.description", "must be a string or null")
        if description is not None:
            translated["description"] = description
    if (
        "strict" in raw_function
        and raw_function["strict"] is not None
        and not isinstance(raw_function["strict"], bool)
    ):
        _raise(f"{param}.strict", "must be a boolean or null")
    return translated


def _translate_tool_choice(choice: Any, param: str) -> dict[str, Any] | str:
    if isinstance(choice, str):
        if choice in {"none", "auto", "required"}:
            return choice
        _raise(param, "must be none, auto, required, or a named function choice")
    if not isinstance(choice, dict):
        _raise(param, "must be a string or an object")

    choice_type = choice.get("type")
    if choice_type == "function":
        _reject_unexpected_fields(choice, param, {"type", "function"})
        function = choice.get("function")
        if not isinstance(function, dict):
            _raise(f"{param}.function", "must be an object")
        _reject_unexpected_fields(function, f"{param}.function", {"name"})
        return {
            "type": "function",
            "name": _required_string(function.get("name"), f"{param}.function.name"),
        }
    if choice_type == "allowed_tools":
        _reject_unexpected_fields(choice, param, {"type", "allowed_tools"})
        allowed = choice.get("allowed_tools")
        if not isinstance(allowed, dict):
            _raise(f"{param}.allowed_tools", "must be an object")
        _reject_unexpected_fields(allowed, f"{param}.allowed_tools", {"mode", "tools"})
        mode = allowed.get("mode")
        if mode not in {"auto", "required"}:
            _raise(f"{param}.allowed_tools.mode", "must be auto or required")
        raw_tools = allowed.get("tools")
        if not isinstance(raw_tools, list):
            _raise(f"{param}.allowed_tools.tools", "must be an array")
        return {
            "type": "allowed_tools",
            "mode": mode,
            "tools": [
                _translate_allowed_function(
                    tool, f"{param}.allowed_tools.tools[{index}]"
                )
                for index, tool in enumerate(raw_tools)
            ],
        }
    _raise(
        f"{param}.type", "only function and allowed_tools choices can be represented"
    )


def _translate_allowed_function(raw_tool: Any, param: str) -> dict[str, Any]:
    if not isinstance(raw_tool, dict):
        _raise(param, "must be an object")
    _reject_unexpected_fields(raw_tool, param, {"type", "function"})
    if raw_tool.get("type") != "function":
        _raise(f"{param}.type", "only function tools can be represented")
    function = raw_tool.get("function")
    if not isinstance(function, dict):
        _raise(f"{param}.function", "must be an object")
    _reject_unexpected_fields(function, f"{param}.function", {"name"})
    return {
        "type": "function",
        "name": _required_string(function.get("name"), f"{param}.function.name"),
    }


def _translate_function_choice(choice: Any, param: str) -> dict[str, Any] | str:
    if isinstance(choice, str):
        if choice in {"none", "auto"}:
            return choice
        _raise(param, "must be none, auto, or an object naming a function")
    if not isinstance(choice, dict):
        _raise(param, "must be a string or an object")
    _reject_unexpected_fields(choice, param, {"name"})
    return {
        "type": "function",
        "name": _required_string(choice.get("name"), f"{param}.name"),
    }


def _translate_text_config(body: dict[str, Any], out: dict[str, Any]) -> None:
    text: dict[str, Any] = {}
    if body.get("response_format") is not None:
        text["format"] = _translate_response_format(body["response_format"])
    if body.get("verbosity") is not None:
        text["verbosity"] = body["verbosity"]
    if text:
        out["text"] = text


def _translate_response_format(raw_format: Any) -> dict[str, Any]:
    if not isinstance(raw_format, dict):
        _raise("response_format", "must be an object")
    format_type = raw_format.get("type")
    if format_type in {"text", "json_object"}:
        _reject_unexpected_fields(raw_format, "response_format", {"type"})
        return {"type": format_type}
    if format_type != "json_schema":
        _raise("response_format.type", "must be text, json_object, or json_schema")

    _reject_unexpected_fields(raw_format, "response_format", {"type", "json_schema"})
    json_schema = raw_format.get("json_schema")
    if not isinstance(json_schema, dict):
        _raise("response_format.json_schema", "must be an object")
    _reject_unexpected_fields(
        json_schema,
        "response_format.json_schema",
        {"name", "description", "schema", "strict"},
    )
    translated: dict[str, Any] = {
        "type": "json_schema",
        "name": _required_string(
            json_schema.get("name"), "response_format.json_schema.name"
        ),
        "schema": json_schema.get("schema") or {},
        "strict": json_schema.get("strict")
        if json_schema.get("strict") is not None
        else False,
    }
    if (
        "schema" in json_schema
        and json_schema["schema"] is not None
        and not isinstance(json_schema["schema"], dict)
    ):
        _raise("response_format.json_schema.schema", "must be an object or null")
    if "description" in json_schema:
        description = json_schema["description"]
        if description is not None and not isinstance(description, str):
            _raise(
                "response_format.json_schema.description", "must be a string or null"
            )
        if description is not None:
            translated["description"] = description
    if (
        "strict" in json_schema
        and json_schema["strict"] is not None
        and not isinstance(json_schema["strict"], bool)
    ):
        _raise("response_format.json_schema.strict", "must be a boolean or null")
    return translated


def _translate_stream_options(body: dict[str, Any], out: dict[str, Any]) -> None:
    if body.get("stream_options") is None:
        return
    if body.get("stream") is not True:
        _raise("stream_options", "is only valid when stream is true")
    options = body["stream_options"]
    if not isinstance(options, dict):
        _raise("stream_options", "must be an object")
    _reject_unexpected_fields(
        options, "stream_options", {"include_obfuscation", "include_usage"}
    )
    if "include_obfuscation" in options:
        out["stream_options"] = {"include_obfuscation": options["include_obfuscation"]}


def _required_string(value: Any, param: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        _raise(
            param,
            "must be a non-empty string" if not allow_empty else "must be a string",
        )
    return value


def _reject_unexpected_fields(
    value: dict[str, Any], param: str, allowed: set[str]
) -> None:
    for field in value:
        if field not in allowed:
            _raise(f"{param}.{field}", "is not supported by the Responses equivalent")


def _raise(param: str, detail: str) -> None:
    raise ChatRequestError(f"{param} {detail}", param)
