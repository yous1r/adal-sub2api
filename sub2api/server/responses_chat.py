"""Translate OpenAI Responses payloads and streams into Chat Completions.

The Responses endpoint is richer than Chat Completions, but CLIProxyAPI consumes
its Chat-compatible surface.  This module preserves the information that has a
Chat equivalent (text, refusals, function calls, reasoning summaries, finish
state, and usage) without taking ownership of upstream transport lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from anyio import CancelScope

from ..compat.aggregate import SSEDecoder
from .openai import build_chunk, build_completion, completion_id, now_epoch

logger = logging.getLogger(__name__)


class ResponsesBridgeError(ValueError):
    """A Responses payload that cannot be represented as Chat Completions."""

    def __init__(self, error: dict[str, Any]) -> None:
        self.error = _openai_error(error, "Responses upstream returned an error")
        super().__init__(str(self.error["message"]))


def _openai_error(value: Any, fallback: str) -> dict[str, Any]:
    """Keep an upstream OpenAI error intact while guaranteeing Chat fields."""
    if isinstance(value, dict):
        nested = value.get("error")
        if isinstance(nested, dict):
            error = dict(nested)
        elif isinstance(value.get("message"), str):
            error = dict(value)
        else:
            error = {key: value[key] for key in ("code", "param") if key in value}
    else:
        error = {}

    message = error.get("message")
    error["message"] = message if isinstance(message, str) and message else fallback
    if not isinstance(error.get("type"), str):
        error["type"] = "api_error"
    return error


def _bridge_error(value: Any, fallback: str) -> ResponsesBridgeError:
    return ResponsesBridgeError(_openai_error(value, fallback))


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _output_index(frame: dict[str, Any]) -> int:
    index = frame.get("output_index")
    if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
        return index
    return 0


def _content_index(frame: dict[str, Any]) -> int:
    index = frame.get("content_index")
    if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
        return index
    return 0


def _summary_index(frame: dict[str, Any]) -> int:
    index = frame.get("summary_index")
    if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
        return index
    return _content_index(frame)


def _part_value(part: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = part.get(key)
        if isinstance(value, str):
            return value
    return ""


def _usage_from_response(response: dict[str, Any]) -> dict[str, Any] | None:
    """Map only usage fields actually supplied by the Responses envelope."""
    source = response.get("usage")
    if not isinstance(source, dict):
        return None

    usage: dict[str, Any] = {}
    input_tokens = source.get("input_tokens")
    output_tokens = source.get("output_tokens")
    if isinstance(input_tokens, int) and not isinstance(input_tokens, bool):
        usage["prompt_tokens"] = input_tokens
    if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
        usage["completion_tokens"] = output_tokens

    total_tokens = source.get("total_tokens")
    if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
        usage["total_tokens"] = total_tokens
    elif "prompt_tokens" in usage and "completion_tokens" in usage:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    input_details = source.get("input_tokens_details")
    if isinstance(input_details, dict):
        cached = input_details.get("cached_tokens")
        if isinstance(cached, int) and not isinstance(cached, bool):
            usage["prompt_tokens_details"] = {"cached_tokens": cached}

    output_details = source.get("output_tokens_details")
    if isinstance(output_details, dict):
        reasoning = output_details.get("reasoning_tokens")
        if isinstance(reasoning, int) and not isinstance(reasoning, bool):
            usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}

    return usage or None


def _function_call(item: dict[str, Any]) -> dict[str, Any]:
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments")
    if not isinstance(call_id, str) or not call_id:
        raise _bridge_error(item, "Function-call output is missing call_id")
    if not isinstance(name, str) or not name:
        raise _bridge_error(item, "Function-call output is missing name")
    if not isinstance(arguments, str):
        raise _bridge_error(item, "Function-call output has invalid arguments")
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _response_parts(
    response: dict[str, Any],
) -> tuple[str, str, str, list[dict[str, Any]]]:
    """Extract the Chat-representable pieces from a completed response."""
    output = response.get("output")
    if not isinstance(output, list):
        raise _bridge_error(response, "Responses payload is missing output")

    text: list[str] = []
    refusals: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            raise _bridge_error(
                response, "Responses payload has an invalid output item"
            )
        item_type = item.get("type")
        if item_type == "message":
            content = item.get("content")
            if not isinstance(content, list):
                raise _bridge_error(item, "Message output has invalid content")
            for part in content:
                if not isinstance(part, dict):
                    raise _bridge_error(
                        item, "Message output has an invalid content part"
                    )
                part_type = part.get("type")
                if part_type == "output_text":
                    text.append(_part_value(part, "text"))
                elif part_type == "refusal":
                    refusals.append(_part_value(part, "refusal", "text"))
        elif item_type == "reasoning":
            summary = item.get("summary")
            if summary is None:
                continue
            if not isinstance(summary, list):
                raise _bridge_error(item, "Reasoning output has invalid summary")
            for part in summary:
                if not isinstance(part, dict):
                    raise _bridge_error(
                        item, "Reasoning output has an invalid summary part"
                    )
                reasoning.append(_part_value(part, "text"))
        elif item_type == "function_call":
            tool_calls.append(_function_call(item))
        else:
            raise _bridge_error(item, "Responses output item has an unsupported type")

    return "".join(text), "".join(refusals), "".join(reasoning), tool_calls


def _finish_reason(response: dict[str, Any], has_tool_calls: bool) -> str:
    status = response.get("status")
    if status == "completed":
        return "tool_calls" if has_tool_calls else "stop"

    details = response.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    if reason == "content_filter":
        return "content_filter"
    if reason == "max_output_tokens":
        return "length"
    raise _bridge_error(
        response, "Responses upstream returned an unclassified incomplete response"
    )


def _response_failure(response: dict[str, Any], fallback: str) -> ResponsesBridgeError:
    error = response.get("error")
    return _bridge_error(error if error is not None else response, fallback)


def chat_completion_from_response(
    payload: dict[str, Any], *, model: str
) -> dict[str, Any]:
    """Translate a terminal Responses JSON envelope to a Chat completion."""
    if not isinstance(payload, dict):
        raise _bridge_error(payload, "Responses upstream returned an invalid payload")
    if payload.get("error") is not None:
        raise _response_failure(payload, "Responses upstream returned an error")

    status = payload.get("status")
    if status == "failed":
        raise _response_failure(payload, "Responses upstream failed")
    if status not in {"completed", "incomplete"}:
        raise _bridge_error(
            payload, "Responses upstream did not return a terminal response"
        )

    text, refusal, reasoning, tool_calls = _response_parts(payload)

    result = build_completion(
        id=completion_id(),
        created=now_epoch(),
        model=model,
        content=text,
        finish_reason=_finish_reason(payload, bool(tool_calls)),
    )
    message = result["choices"][0]["message"]
    if not text and (refusal or reasoning or tool_calls):
        message["content"] = None
    if refusal:
        message["refusal"] = refusal
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = _usage_from_response(payload)
    if usage is not None:
        result["usage"] = usage
    return result


@dataclass(slots=True)
class _ToolCallState:
    index: int
    call_id: str
    name: str
    emitted_arguments: int = 0


class _StreamTranslator:
    """Incremental state needed to convert Responses events without buffering."""

    def __init__(
        self, *, id: str, created: int, model: str, include_usage: bool
    ) -> None:
        self._id = id
        self._created = created
        self._model = model
        self._include_usage = include_usage
        self._role_sent = False
        self._terminal = False
        self._emitted_lengths: dict[tuple[str, int, int], int] = {}
        self._tools: dict[int, _ToolCallState] = {}
        self._item_indices: dict[str, int] = {}
        self._added_indices: list[int] = []

    def _chunk(
        self, delta: dict[str, Any], finish_reason: str | None = None
    ) -> dict[str, Any]:
        return build_chunk(
            id=self._id,
            created=self._created,
            model=self._model,
            delta=delta,
            finish_reason=finish_reason,
        )

    def _assistant_delta(self, delta: dict[str, Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        if not self._role_sent:
            self._role_sent = True
            chunks.append(self._chunk({"role": "assistant"}))
        chunks.append(self._chunk(delta))
        return chunks

    def _ensure_role(self, chunks: list[dict[str, Any]]) -> None:
        if not self._role_sent:
            self._role_sent = True
            chunks.append(self._chunk({"role": "assistant"}))

    def _snapshot_suffix(self, key: tuple[str, int, int], value: str) -> str:
        emitted = self._emitted_lengths.get(key, 0)
        if len(value) <= emitted:
            return ""
        self._emitted_lengths[key] = len(value)
        return value[emitted:]

    def _append_delta(self, key: tuple[str, int, int], value: str) -> str:
        self._emitted_lengths[key] = self._emitted_lengths.get(key, 0) + len(value)
        return value

    def _text_snapshot(
        self, kind: str, output_index: int, part_index: int, value: str
    ) -> list[dict[str, Any]]:
        suffix = self._snapshot_suffix((kind, output_index, part_index), value)
        return self._assistant_delta({kind: suffix}) if suffix else []

    def _text_delta(
        self, kind: str, output_index: int, part_index: int, value: str
    ) -> list[dict[str, Any]]:
        if not value:
            return []
        self._append_delta((kind, output_index, part_index), value)
        return self._assistant_delta({kind: value})

    def _remember_item(self, index: int, item: dict[str, Any]) -> None:
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            self._item_indices[item_id] = index
        if index not in self._added_indices:
            self._added_indices.append(index)

    def _tool_snapshot(
        self, output_index: int, item: dict[str, Any]
    ) -> list[dict[str, Any]]:
        state = self._tools.get(output_index)
        if state is None:
            call_id = item.get("call_id")
            name = item.get("name")
            if not isinstance(call_id, str) or not call_id:
                raise _bridge_error(item, "Function-call output is missing call_id")
            if not isinstance(name, str) or not name:
                raise _bridge_error(item, "Function-call output is missing name")
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                raise _bridge_error(item, "Function-call output has invalid arguments")
            state = _ToolCallState(
                index=len(self._tools),
                call_id=call_id,
                name=name,
                emitted_arguments=len(arguments),
            )
            self._tools[output_index] = state
            tool_call = {
                "index": state.index,
                "id": state.call_id,
                "type": "function",
                "function": {"name": state.name, "arguments": arguments},
            }
            return self._assistant_delta({"tool_calls": [tool_call]})

        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            raise _bridge_error(item, "Function-call output has invalid arguments")
        if len(arguments) <= state.emitted_arguments:
            return []
        suffix = arguments[state.emitted_arguments :]
        state.emitted_arguments = len(arguments)
        return self._assistant_delta(
            {"tool_calls": [{"index": state.index, "function": {"arguments": suffix}}]}
        )

    def _tool_delta(self, output_index: int, arguments: str) -> list[dict[str, Any]]:
        state = self._tools.get(output_index)
        if state is None:
            raise _bridge_error(
                {"message": "Function-call arguments arrived before its output item"},
                "Malformed Responses stream",
            )
        if not arguments:
            return []
        state.emitted_arguments += len(arguments)
        return self._assistant_delta(
            {
                "tool_calls": [
                    {"index": state.index, "function": {"arguments": arguments}}
                ]
            }
        )

    def _tool_done(self, output_index: int, arguments: str) -> list[dict[str, Any]]:
        state = self._tools.get(output_index)
        if state is None:
            raise _bridge_error(
                {"message": "Function-call completion arrived before its output item"},
                "Malformed Responses stream",
            )
        if len(arguments) <= state.emitted_arguments:
            return []
        suffix = arguments[state.emitted_arguments :]
        state.emitted_arguments = len(arguments)
        return self._assistant_delta(
            {"tool_calls": [{"index": state.index, "function": {"arguments": suffix}}]}
        )

    def _item_snapshot(
        self, output_index: int, item: dict[str, Any]
    ) -> list[dict[str, Any]]:
        self._remember_item(output_index, item)
        item_type = item.get("type")
        chunks: list[dict[str, Any]] = []
        if item_type == "message":
            content = item.get("content")
            if not isinstance(content, list):
                raise _bridge_error(item, "Message output has invalid content")
            for part_index, part in enumerate(content):
                if not isinstance(part, dict):
                    raise _bridge_error(
                        item, "Message output has an invalid content part"
                    )
                if part.get("type") == "output_text":
                    chunks.extend(
                        self._text_snapshot(
                            "content",
                            output_index,
                            part_index,
                            _part_value(part, "text"),
                        )
                    )
                elif part.get("type") == "refusal":
                    chunks.extend(
                        self._text_snapshot(
                            "refusal",
                            output_index,
                            part_index,
                            _part_value(part, "refusal", "text"),
                        )
                    )
        elif item_type == "reasoning":
            summary = item.get("summary")
            if summary is None:
                return chunks
            if not isinstance(summary, list):
                raise _bridge_error(item, "Reasoning output has invalid summary")
            for part_index, part in enumerate(summary):
                if not isinstance(part, dict):
                    raise _bridge_error(
                        item, "Reasoning output has an invalid summary part"
                    )
                chunks.extend(
                    self._text_snapshot(
                        "reasoning_content",
                        output_index,
                        part_index,
                        _part_value(part, "text"),
                    )
                )
        elif item_type == "function_call":
            chunks.extend(self._tool_snapshot(output_index, item))
        else:
            raise _bridge_error(item, "Responses output item has an unsupported type")
        return chunks

    def _terminal_chunks(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        if response.get("error") is not None:
            raise _response_failure(response, "Responses upstream returned an error")
        status = response.get("status")
        if status == "failed":
            raise _response_failure(response, "Responses upstream failed")
        if status not in {"completed", "incomplete"}:
            raise _bridge_error(
                response, "Responses stream did not return a terminal response"
            )
        output = response.get("output")
        if not isinstance(output, list):
            raise _bridge_error(response, "Responses payload is missing output")

        chunks: list[dict[str, Any]] = []
        for position, item in enumerate(output):
            if not isinstance(item, dict):
                raise _bridge_error(
                    response, "Responses payload has an invalid output item"
                )
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id in self._item_indices:
                output_index = self._item_indices[item_id]
            elif position < len(self._added_indices):
                output_index = self._added_indices[position]
            else:
                output_index = position
            chunks.extend(self._item_snapshot(output_index, item))

        self._ensure_role(chunks)
        chunks.append(
            self._chunk(
                {},
                finish_reason=_finish_reason(response, bool(self._tools)),
            )
        )
        if self._include_usage:
            usage = _usage_from_response(response)
            if usage is not None:
                chunks.append(
                    {
                        "id": self._id,
                        "object": "chat.completion.chunk",
                        "created": self._created,
                        "model": self._model,
                        "choices": [],
                        "usage": usage,
                    }
                )
        self._terminal = True
        return chunks

    def feed(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert one decoded Responses SSE frame to zero or more Chat chunks."""
        event_type = frame.get("type")
        if event_type in {"error", "response.error"}:
            error = frame.get("error")
            if error is None:
                error = {
                    key: frame[key]
                    for key in ("message", "code", "param")
                    if key in frame
                }
            raise _bridge_error(error, "Responses upstream returned a stream error")
        if event_type == "response.failed":
            response = frame.get("response")
            if isinstance(response, dict):
                raise _response_failure(response, "Responses upstream failed")
            raise _bridge_error(frame, "Responses upstream failed")
        if event_type in {"response.completed", "response.incomplete"}:
            response = frame.get("response")
            if not isinstance(response, dict):
                raise _bridge_error(frame, "Responses terminal event has no response")
            return self._terminal_chunks(response)
        if event_type in {"response.created", "response.in_progress"}:
            return []
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            item = frame.get("item")
            if not isinstance(item, dict):
                raise _bridge_error(frame, "Responses output event has no output item")
            return self._item_snapshot(_output_index(frame), item)
        if event_type == "response.output_text.delta":
            delta = frame.get("delta")
            if not isinstance(delta, str):
                raise _bridge_error(frame, "Responses text delta is invalid")
            return self._text_delta(
                "content", _output_index(frame), _content_index(frame), delta
            )
        if event_type == "response.output_text.done":
            text = frame.get("text")
            if not isinstance(text, str):
                raise _bridge_error(frame, "Responses text completion is invalid")
            return self._text_snapshot(
                "content", _output_index(frame), _content_index(frame), text
            )
        if event_type == "response.refusal.delta":
            delta = frame.get("delta")
            if not isinstance(delta, str):
                raise _bridge_error(frame, "Responses refusal delta is invalid")
            return self._text_delta(
                "refusal", _output_index(frame), _content_index(frame), delta
            )
        if event_type == "response.refusal.done":
            refusal = frame.get("refusal")
            if not isinstance(refusal, str):
                raise _bridge_error(frame, "Responses refusal completion is invalid")
            return self._text_snapshot(
                "refusal", _output_index(frame), _content_index(frame), refusal
            )
        if isinstance(event_type, str) and event_type.startswith("response.reasoning"):
            if event_type.endswith(("_part.added", "_part.done")):
                return []
            output_index = _output_index(frame)
            summary_index = _summary_index(frame)
            if event_type.endswith(".delta"):
                delta = frame.get("delta")
                if not isinstance(delta, str):
                    raise _bridge_error(frame, "Responses reasoning delta is invalid")
                return self._text_delta(
                    "reasoning_content", output_index, summary_index, delta
                )
            if event_type.endswith(".done"):
                text = frame.get("text")
                if not isinstance(text, str):
                    raise _bridge_error(
                        frame, "Responses reasoning completion is invalid"
                    )
                return self._text_snapshot(
                    "reasoning_content", output_index, summary_index, text
                )
            return []
        if event_type == "response.function_call_arguments.delta":
            delta = frame.get("delta")
            if not isinstance(delta, str):
                raise _bridge_error(
                    frame, "Responses function arguments delta is invalid"
                )
            return self._tool_delta(_output_index(frame), delta)
        if event_type == "response.function_call_arguments.done":
            arguments = frame.get("arguments")
            if not isinstance(arguments, str):
                raise _bridge_error(
                    frame, "Responses function arguments completion is invalid"
                )
            return self._tool_done(_output_index(frame), arguments)
        return []


def _sse(payload: dict[str, Any] | str) -> bytes:
    if isinstance(payload, str):
        encoded = payload
    else:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {encoded}\n\n".encode()


async def _close_early(upstream: AsyncIterator[bytes]) -> None:
    close = getattr(upstream, "aclose", None)
    if close is None:
        return
    try:
        with CancelScope(shield=True):
            await close()
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - source teardown must not mask cancellation
        # The upstream generator owns transport teardown.  A close failure must
        # not turn a client cancellation into a second terminal response.
        return


async def chat_chunks_from_responses(
    upstream: AsyncIterator[bytes], *, model: str, include_usage: bool = False
) -> AsyncIterator[bytes]:
    """Translate Responses SSE while keeping the forwarder responsible for I/O."""
    decoder = SSEDecoder()
    translator = _StreamTranslator(
        id=completion_id(),
        created=now_epoch(),
        model=model,
        include_usage=include_usage,
    )
    source_exhausted = False
    try:
        while not source_exhausted:
            try:
                data = await anext(upstream)
            except StopAsyncIteration:
                source_exhausted = True
                frames = decoder.flush()
            else:
                frames = decoder.feed(data)
            for frame in frames:
                for chunk in translator.feed(frame):
                    yield _sse(chunk)
                if translator._terminal:
                    yield _sse("[DONE]")
                    return
        raise _bridge_error(None, "Responses stream ended without a terminal response")
    except ResponsesBridgeError as exc:
        yield _sse({"error": exc.error})
        yield _sse("[DONE]")
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # transport failures must remain visible to Chat clients
        logger.exception("Responses-to-Chat stream conversion failed")
        yield _sse(
            {
                "error": _openai_error(
                    {"message": str(exc)}, "Responses upstream transport failed"
                )
            }
        )
        yield _sse("[DONE]")
    finally:
        if not source_exhausted:
            await _close_early(upstream)
