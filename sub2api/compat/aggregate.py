"""Reconstruct non-stream JSON envelopes from decoded SSE data frames.

The AdaL proxy fronts SDK wrappers that behave differently per provider:
``api.anthropic.com`` populates ``message_stop.message`` but ``zai``,
``minimax`` and ``xai`` send a bare ``{"type": "message_stop"}`` and report
usage solely via ``message_delta.usage`` (xai only ``output_tokens`` there).
The forwarder therefore forces ``stream: true`` upstream and reassembles the
non-stream envelope the client expects, rather than trusting the terminal
frame to carry it.

Everything here operates on already-decoded frames (plain ``dict`` objects):
no I/O, no asyncio, no httpx.  ``SSEDecoder`` is the one exception — it turns
raw ``bytes`` into those frames with an incremental line buffer, so a JSON
object split across TCP reads still decodes.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# SSE decoding
# ---------------------------------------------------------------------------


class SSEDecoder:
    """Incremental SSE decoder: ``feed`` raw bytes, get decoded data frames.

    Splits on ``\\n``, keeps a trailing partial line across calls, recognises
    ``data: <json>`` lines (ignoring ``event:``/``id:``/``retry:``/comment
    lines and blank separators), skips ``data: [DONE]``, and silently skips
    unparseable JSON.  Byte-exact-safe: a JSON object split mid-token across
    two ``feed()`` calls decodes correctly because incomplete lines are
    buffered until their terminator arrives.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[dict]:
        """Consume raw bytes, return every complete decoded data frame."""
        self._buffer.extend(data)
        frames: list[dict] = []
        while b"\n" in self._buffer:
            line, _, rest = bytes(self._buffer).partition(b"\n")
            self._buffer = bytearray(rest)
            frame = _decode_line(line)
            if frame is not None:
                frames.append(frame)
        return frames

    def flush(self) -> list[dict]:
        """Decode any trailing line that lacked a final newline."""
        line = bytes(self._buffer)
        self._buffer = bytearray()
        frame = _decode_line(line)
        return [frame] if frame is not None else []


def _decode_line(line: bytes) -> dict | None:
    text = line.decode("utf-8", errors="replace").rstrip("\r")
    if not text or text.startswith(":"):
        return None  # blank separator or comment line
    field, sep, value = text.partition(":")
    if sep and value.startswith(" "):
        value = value.removeprefix(" ")
    if field != "data" or value == "[DONE]":
        return None
    try:
        frame = json.loads(value)
    except (ValueError, TypeError):
        return None
    return frame if isinstance(frame, dict) else None


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _at(blocks: list[Any], index: int | None) -> Any | None:
    """Return the block at ``index``, tolerating None/gaps (list grows)."""
    if index is None:
        return blocks[-1] if blocks else None
    while len(blocks) <= index:
        blocks.append(None)
    return blocks[index]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicAggregator:
    """Rebuilds the non-stream ``/v1/messages`` envelope from a stream.

    ``message_stop.message`` is used only as an optional fast path because
    only ``api.anthropic.com`` populates it; a bare ``message_stop`` must
    leave the accumulated envelope intact.
    """

    def __init__(self) -> None:
        self._envelope: dict | None = None
        self._blocks: list[Any] = []
        self._json_buffers: dict[int, str] = {}

    def feed(self, frame: dict) -> None:
        ftype = frame.get("type")
        if ftype == "message_start":
            message = frame.get("message")
            if isinstance(message, dict):
                self._envelope = dict(message)
            else:
                self._envelope = {"type": "message", "role": "assistant"}
            self._envelope["content"] = self._blocks
        elif ftype == "content_block_start":
            index = frame.get("index")
            if index is None:
                index = len(self._blocks)
            block = frame.get("content_block")
            if not isinstance(block, dict):
                return
            while len(self._blocks) <= index:
                self._blocks.append(None)
            self._blocks[index] = dict(block)
        elif ftype == "content_block_delta":
            index = frame.get("index")
            block = _at(self._blocks, index)
            if not isinstance(block, dict):
                return
            delta = frame.get("delta")
            if not isinstance(delta, dict):
                return
            dtype = delta.get("type")
            if dtype == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif dtype == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get(
                    "thinking", ""
                )
            elif dtype == "signature_delta":
                block["signature"] = block.get("signature", "") + delta.get(
                    "signature", ""
                )
            elif dtype == "input_json_delta":
                idx = index if index is not None else max(len(self._blocks) - 1, 0)
                self._json_buffers[idx] = self._json_buffers.get(idx, "") + delta.get(
                    "partial_json", ""
                )
        elif ftype == "content_block_stop":
            index = frame.get("index")
            if index is None:
                return
            block = _at(self._blocks, index)
            if not isinstance(block, dict):
                return
            raw = self._json_buffers.pop(index, "")
            if not raw:
                return  # text/thinking blocks carry no buffered JSON
            try:
                block["input"] = json.loads(raw)
            except (ValueError, TypeError):
                block["input"] = {}
        elif ftype == "message_delta":
            envelope = self._envelope
            if envelope is None:
                envelope = self._envelope = {
                    "type": "message",
                    "role": "assistant",
                    "content": self._blocks,
                }
            delta = frame.get("delta")
            if isinstance(delta, dict):
                for key in ("stop_reason", "stop_sequence"):
                    if key in delta:
                        envelope[key] = delta[key]
            usage = frame.get("usage")
            if isinstance(usage, dict):
                env_usage = envelope.setdefault("usage", {})
                for key, value in usage.items():
                    if value is not None:
                        env_usage[key] = value
        elif ftype == "message_stop":
            message = frame.get("message")
            if isinstance(message, dict):
                # Fast path: only api.anthropic.com populates this key.
                self._envelope = dict(message)
                self._envelope.setdefault("content", self._blocks)
            # Bare message_stop: the accumulated envelope stays intact.
        # ping frames and unknown types: silently ignored

    def result(self) -> dict:
        """The assembled non-stream envelope (never None)."""
        if self._envelope is None:
            return {
                "type": "message",
                "role": "assistant",
                "content": self._blocks,
                "usage": {},
            }
        return self._envelope

    def usage(self) -> dict[str, int]:
        usage = self._envelope.get("usage", {}) if self._envelope else {}
        return {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_read_tokens": int(usage.get("cache_read_input_tokens") or 0),
            "cache_write_tokens": int(usage.get("cache_creation_input_tokens") or 0),
            "reasoning_tokens": 0,
        }


# ---------------------------------------------------------------------------
# OpenAI chat completions
# ---------------------------------------------------------------------------


class OpenAIChatAggregator:
    """Rebuilds a ``/v1/chat/completions`` object from streamed chunks.

    Usage comes from the trailing usage-only chunk (``choices: []``), which
    only appears when ``stream_options.include_usage`` is set upstream.
    """

    def __init__(self) -> None:
        self._envelope: dict | None = None
        self._usage: dict = {}
        self._tool_calls: dict[int, dict] = {}
        self._finish_reason: str | None = None

    def feed(self, frame: dict) -> None:
        choices = frame.get("choices")
        if not isinstance(choices, list):
            return
        envelope = self._envelope
        if envelope is None:
            envelope = self._envelope = {
                "id": frame.get("id", ""),
                "object": "chat.completion",
                "created": frame.get("created", 0),
                "model": frame.get("model", ""),
            }
        usage = frame.get("usage")
        if isinstance(usage, dict):
            self._usage = usage
        if not choices:
            return
        first = choices[0]
        if not isinstance(first, dict):
            return
        finish = first.get("finish_reason")
        if finish is not None:
            self._finish_reason = finish
        delta = first.get("delta")
        if not isinstance(delta, dict):
            return
        message = envelope.setdefault("message", {})
        content = delta.get("content")
        if isinstance(content, str):
            message["content"] = message.get("content", "") + content
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str):
            message["reasoning_content"] = (
                message.get("reasoning_content", "") + reasoning
            )
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                self._accumulate_tool_call(tc)

    def _accumulate_tool_call(self, tc: dict) -> None:
        index = tc.get("index", 0)
        slot = self._tool_calls.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if tc.get("id"):
            slot["id"] = tc["id"]
        function = tc.get("function")
        if isinstance(function, dict):
            if function.get("name"):
                slot["function"]["name"] = function["name"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                slot["function"]["arguments"] += arguments

    def result(self) -> dict:
        """The assembled chat.completion object."""
        envelope = self._envelope
        if envelope is None:
            envelope = {
                "id": "",
                "object": "chat.completion",
                "created": 0,
                "model": "",
            }
        envelope = dict(envelope)
        message = dict(envelope.get("message", {}))
        calls = [self._tool_calls[i] for i in sorted(self._tool_calls)]
        if calls:
            message["tool_calls"] = calls
        envelope["choices"] = [
            {"index": 0, "message": message, "finish_reason": self._finish_reason}
        ]
        if self._usage:
            envelope["usage"] = self._usage
        return envelope

    def usage(self) -> dict[str, int]:
        usage = self._usage
        prompt_details = usage.get("prompt_tokens_details")
        prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
        if "cached_tokens" in prompt_details:
            cache_read = prompt_details.get("cached_tokens") or 0
        else:
            # DeepSeek reports cache hits at the top level of usage.
            cache_read = usage.get("prompt_cache_hit_tokens") or 0
        completion_details = usage.get("completion_tokens_details")
        completion_details = (
            completion_details if isinstance(completion_details, dict) else {}
        )
        return {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "cache_read_tokens": int(cache_read),
            "cache_write_tokens": int(prompt_details.get("cache_write_tokens") or 0),
            "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
        }


# ---------------------------------------------------------------------------
# OpenAI responses
# ---------------------------------------------------------------------------


class ResponsesAggregator:
    """Rebuilds a ``/v1/responses`` object from streamed response events.

    ``response.completed`` carries the finished ``response`` verbatim; the
    incremental events (``output_item.added`` / ``output_text.delta`` /
    ``output_item.done``) are the fallback used when the terminal event is
    never seen.
    """

    def __init__(self) -> None:
        self._completed: dict | None = None
        self._response: dict = {}
        self._items: list[Any] = []

    def feed(self, frame: dict) -> None:
        etype = frame.get("type")
        if etype == "response.completed":
            response = frame.get("response")
            if isinstance(response, dict):
                self._completed = response
            return
        if etype in ("response.created", "response.in_progress"):
            response = frame.get("response")
            if isinstance(response, dict):
                self._response = dict(response)
            return
        if etype == "response.output_item.added":
            index = frame.get("output_index")
            if index is None:
                index = len(self._items)
            item = frame.get("item")
            if not isinstance(item, dict):
                return
            while len(self._items) <= index:
                self._items.append(None)
            self._items[index] = dict(item)
            return
        if etype == "response.output_text.delta":
            item = self._last_text_item(frame.get("output_index"))
            if item is None:
                return
            parts = item.setdefault("content", [])
            content = parts[-1] if parts else None
            if not isinstance(content, dict) or content.get("type") != "output_text":
                content = {"type": "output_text", "text": ""}
                parts.append(content)
            content["text"] = content.get("text", "") + frame.get("delta", "")
            return
        if etype == "response.output_item.done":
            index = frame.get("output_index")
            item = frame.get("item")
            if not isinstance(item, dict):
                return
            if index is None:
                index = max(len(self._items) - 1, 0)
            while len(self._items) <= index:
                self._items.append(None)
            self._items[index] = dict(item)
            return
        # usage arrives inside response.completed; other event types ignored

    def _last_text_item(self, output_index: int | None) -> dict | None:
        if output_index is not None:
            item = _at(self._items, output_index)
            return item if isinstance(item, dict) else None
        for item in reversed(self._items):
            if isinstance(item, dict):
                return item
        return None

    def result(self) -> dict:
        """The assembled response object (verbatim when completed was seen)."""
        if self._completed is not None:
            return self._completed
        response = dict(self._response)
        response["object"] = response.get("object", "response")
        response["output"] = [item for item in self._items if isinstance(item, dict)]
        return response

    def usage(self) -> dict[str, int]:
        source = self._completed if self._completed is not None else self._response
        usage = source.get("usage", {}) if isinstance(source, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        input_details = usage.get("input_tokens_details")
        input_details = input_details if isinstance(input_details, dict) else {}
        output_details = usage.get("output_tokens_details")
        output_details = output_details if isinstance(output_details, dict) else {}
        return {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_read_tokens": int(input_details.get("cached_tokens") or 0),
            "cache_write_tokens": int(input_details.get("cache_write_tokens") or 0),
            "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
        }
