"""Forwarding layer between sub2api's HTTP surface and the AdaL cloud proxy.

Three concerns live here, all of them consequences of what the upstream proxy
actually does:

* **auth retry** — the proxy answers HTTP 401 when a Clerk JWT expired
  mid-flight (they carry a 60 s TTL); one re-mint plus one retry is enough.
* **forced upstream streaming** (:func:`forward_collect`) — a non-stream
  Anthropic request with ``max_tokens > 21333`` is a hard upstream 500, and a
  request the SDK wrapper rejects surfaces as an opaque 500 instead of the
  clean ``event: error`` the streaming path returns.  Asking upstream to stream
  and re-assembling the envelope server-side removes both failure modes
  without capping the client's ``max_tokens``.
* **``event: error`` is a failure** — the proxy reports rejected requests as
  HTTP 200 with an error frame inside the stream.  Relayed naively that counts
  as a success, poisons slot health, and hands the client an empty answer.

Token accounting rides along: the same frames that are relayed feed a
:class:`~sub2api.compat.aggregate` aggregator, so metering never costs an
extra upstream call.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from ..compat.aggregate import (
    AnthropicAggregator,
    OpenAIChatAggregator,
    ResponsesAggregator,
)
from ..compat.errors import (
    anthropic_error,
    parse_error_frame,
    status_for_anthropic_error,
)
from ..core.pricing import cost_for
from . import openai as oai

_NON_STREAM_TIMEOUT = httpx.Timeout(300.0, connect=15.0, read=60.0)
_STREAM_TIMEOUT = httpx.Timeout(600.0, connect=15.0, read=None)

_AGGREGATORS = {
    "anthropic": AnthropicAggregator,
    "openai_chat": OpenAIChatAggregator,
    "responses": ResponsesAggregator,
}

ZERO_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "reasoning_tokens": 0,
}


def aggregator_for(protocol: str) -> Any:
    """A fresh aggregator for *protocol* (Anthropic shape when unknown)."""
    return _AGGREGATORS.get(protocol, AnthropicAggregator)()


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def usage_from_payload(protocol: str, payload: Any) -> dict[str, int]:
    """Normalized token counts from a non-stream response envelope.

    Mirrors the aggregator mappings for the direct (non-aggregated) paths:
    ``/v1/chat/completions`` and ``/v1/responses`` were both measured clean
    non-stream, so their envelopes are read here instead of being rebuilt.
    """
    if not isinstance(payload, dict):
        return dict(ZERO_USAGE)
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return dict(ZERO_USAGE)
    if protocol == "anthropic":
        return {
            "input_tokens": _int(usage.get("input_tokens")),
            "output_tokens": _int(usage.get("output_tokens")),
            "cache_read_tokens": _int(usage.get("cache_read_input_tokens")),
            "cache_write_tokens": _int(usage.get("cache_creation_input_tokens")),
            "reasoning_tokens": 0,
        }
    if protocol == "responses":
        in_details = usage.get("input_tokens_details")
        in_details = in_details if isinstance(in_details, dict) else {}
        out_details = usage.get("output_tokens_details")
        out_details = out_details if isinstance(out_details, dict) else {}
        return {
            "input_tokens": _int(usage.get("input_tokens")),
            "output_tokens": _int(usage.get("output_tokens")),
            "cache_read_tokens": _int(in_details.get("cached_tokens")),
            "cache_write_tokens": _int(in_details.get("cache_write_tokens")),
            "reasoning_tokens": _int(out_details.get("reasoning_tokens")),
        }
    prompt_details = usage.get("prompt_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    completion_details = usage.get("completion_tokens_details")
    completion_details = (
        completion_details if isinstance(completion_details, dict) else {}
    )
    cache_read = prompt_details.get("cached_tokens")
    if cache_read is None:
        # DeepSeek reports cache hits at the top level of `usage` instead.
        cache_read = usage.get("prompt_cache_hit_tokens")
    return {
        "input_tokens": _int(usage.get("prompt_tokens")),
        "output_tokens": _int(usage.get("completion_tokens")),
        "cache_read_tokens": _int(cache_read),
        "cache_write_tokens": _int(prompt_details.get("cache_write_tokens")),
        "reasoning_tokens": _int(completion_details.get("reasoning_tokens")),
    }


@dataclass(slots=True)
class UsageMeter:
    """One request's metering context; writes exactly one ``usage_events`` row.

    Recording is fire-and-forget and fully swallowed: a metering failure must
    never turn a served response into an error.
    """

    store: Any | None
    session_id: str
    model: str
    provider: str
    protocol: str
    path: str
    stream: bool
    started: float = field(default_factory=time.monotonic)
    recorded: bool = False

    def record(self, *, status: int, usage: dict[str, int] | None) -> None:
        if self.recorded:
            return
        self.recorded = True
        if self.store is None:
            return
        tokens = usage or ZERO_USAGE
        cost, estimated = cost_for(
            model=self.model,
            provider=self.provider,
            protocol=self.protocol,
            input_tokens=tokens.get("input_tokens", 0),
            output_tokens=tokens.get("output_tokens", 0),
            cache_read_tokens=tokens.get("cache_read_tokens", 0),
            cache_write_tokens=tokens.get("cache_write_tokens", 0),
        )
        row = {
            "ts": time.time(),
            "session_id": self.session_id,
            "model": self.model,
            "provider": self.provider,
            "path": self.path,
            "stream": int(self.stream),
            "status": status,
            "latency_ms": round((time.monotonic() - self.started) * 1000.0, 3),
            "input_tokens": tokens.get("input_tokens", 0),
            "output_tokens": tokens.get("output_tokens", 0),
            "cache_read_tokens": tokens.get("cache_read_tokens", 0),
            "cache_write_tokens": tokens.get("cache_write_tokens", 0),
            "reasoning_tokens": tokens.get("reasoning_tokens", 0),
            "cost_usd": cost,
            "rate_estimated": int(estimated),
        }
        store = self.store

        async def _write() -> None:
            try:
                await store.record_usage(**row)
            except Exception:  # metering is never load-bearing
                pass

        try:
            asyncio.get_running_loop().create_task(_write())
        except RuntimeError:  # no loop (sync caller): drop the row
            pass


def _decode_data_line(line: bytes) -> dict | None:
    """Decode one ``data: {...}`` SSE line, or ``None`` when it isn't one."""
    text = line.strip()
    if not text.startswith(b"data:"):
        return None
    payload = text[5:].strip()
    if not payload or payload == b"[DONE]":
        return None
    try:
        frame = json.loads(payload)
    except (ValueError, TypeError):
        return None
    return frame if isinstance(frame, dict) else None


def _is_usage_only_chunk(frame: dict) -> bool:
    """True for OpenAI's trailing usage-only chunk (``choices: []``)."""
    choices = frame.get("choices")
    return bool(frame.get("usage")) and isinstance(choices, list) and not choices


def with_upstream_stream(fwd_body: bytes, protocol: str) -> bytes:
    """Force ``stream: true`` upstream, plus usage on OpenAI chat streams."""
    try:
        body = json.loads(fwd_body) if fwd_body else {}
    except (ValueError, TypeError):
        return fwd_body
    if not isinstance(body, dict):
        return fwd_body
    body["stream"] = True
    if protocol == "openai_chat":
        options = body.get("stream_options")
        options = dict(options) if isinstance(options, dict) else {}
        options["include_usage"] = True
        body["stream_options"] = options
    return json.dumps(body).encode()


def error_response_for(protocol: str, err: dict) -> tuple[int, dict]:
    """Map an upstream error object to ``(status, native error body)``."""
    err_type = str(err.get("type") or "")
    message = str(err.get("message") or "upstream error")
    status = status_for_anthropic_error(err_type)
    if protocol == "anthropic":
        return status, anthropic_error(message, err_type or "api_error")
    return status, oai.openai_error(message, err_type=err_type or "api_error")


async def forward_to_proxy(
    *,
    channel: Any,
    url: str,
    fwd_body: bytes,
    slot: Any,
    sid: str,
    target: str,
    provider: str,
    stream: bool,
    protocol: str = "",
    meter: UsageMeter | None = None,
    strip_usage_chunk: bool = False,
) -> Any:
    """POST the payload to the cloud proxy with one auth-retry.

    On an HTTP 401 the channel re-mints the bearer from the account's Clerk
    cookies (:meth:`refresh_slot_auth`) and the request is retried once with
    rebuilt headers.  Returns the ``httpx.Response`` (non-streaming) or the
    streaming byte generator; the slot is released exactly once either way.

    In streaming mode the relayed frames are teed into an aggregator so the
    request can be metered, an ``event: error`` frame marks the slot failed,
    and ``strip_usage_chunk`` removes the usage-only chunk sub2api asked
    upstream for but the client did not.
    """
    client = channel._client
    assert client is not None

    def build_headers() -> dict[str, str]:
        return channel.proxy_headers(sid, target, provider, slot=slot)

    if not stream:
        try:
            resp = await client.post(
                url,
                content=fwd_body,
                headers=build_headers(),
                timeout=_NON_STREAM_TIMEOUT,
            )
            if resp.status_code == 401 and await channel.refresh_slot_auth(
                slot, force=True
            ):
                resp = await client.post(
                    url,
                    content=fwd_body,
                    headers=build_headers(),
                    timeout=_NON_STREAM_TIMEOUT,
                )
        except Exception:
            await channel.release_slot(slot, success=False)
            if meter is not None:
                meter.record(status=502, usage=None)
            raise
        await channel.release_slot(slot, success=resp.status_code < 500)
        return resp

    async def gen() -> AsyncIterator[bytes]:
        ok = True
        status = 502
        aggregator = aggregator_for(protocol)
        frames: list[dict] = []
        try:
            for attempt in range(2):
                retry = False
                async with client.stream(
                    "POST",
                    url,
                    content=fwd_body,
                    headers=build_headers(),
                    timeout=_STREAM_TIMEOUT,
                ) as resp:
                    status = resp.status_code
                    if resp.status_code == 401 and attempt == 0:
                        retry = True
                        await resp.aread()  # drain the error body
                    elif resp.status_code != 200:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    else:
                        buffer = bytearray()
                        drop_blank = False
                        async for chunk in resp.aiter_bytes():
                            buffer.extend(chunk)
                            while True:
                                cut = buffer.find(b"\n")
                                if cut < 0:
                                    break
                                line = bytes(buffer[: cut + 1])
                                del buffer[: cut + 1]
                                if drop_blank and not line.strip():
                                    drop_blank = False
                                    continue
                                drop_blank = False
                                frame = _decode_data_line(line)
                                if frame is not None:
                                    frames.append(frame)
                                    aggregator.feed(frame)
                                    if strip_usage_chunk and _is_usage_only_chunk(
                                        frame
                                    ):
                                        drop_blank = True
                                        continue
                                yield line
                        if buffer:
                            frame = _decode_data_line(bytes(buffer))
                            if frame is not None:
                                frames.append(frame)
                                aggregator.feed(frame)
                            yield bytes(buffer)
                if retry and await channel.refresh_slot_auth(slot, force=True):
                    continue
                break
        except Exception:
            ok = False
            raise
        finally:
            err = parse_error_frame(frames)
            if err is not None or status >= 500:
                # An `event: error` inside a 200 stream is a real failure; not
                # marking it pollutes slot health and hides the cause.
                ok = False
                if err is not None:
                    status, _body = error_response_for(protocol, err)
            await channel.release_slot(slot, success=ok)
            if meter is not None:
                meter.record(status=status, usage=aggregator.usage())

    return gen()


async def forward_collect(
    *,
    url: str,
    fwd_body: bytes,
    slot: Any,
    sid: str,
    target: str,
    provider: str,
    protocol: str,
    channel: Any,
    meter: UsageMeter | None = None,
) -> tuple[int, dict]:
    """Stream upstream, aggregate server-side, answer as one JSON envelope.

    This is what makes non-stream ``/v1/messages`` work at any
    ``max_tokens``: the upstream rejects a non-stream request above 21333 with
    a hard 500, and reports request-shape rejections as opaque 500s, while the
    streaming surface handles both cleanly.  The 401 re-mint retry is
    preserved.
    """
    client = channel._client
    assert client is not None
    body = with_upstream_stream(fwd_body, protocol)

    def build_headers() -> dict[str, str]:
        return channel.proxy_headers(sid, target, provider, slot=slot)

    aggregator = aggregator_for(protocol)
    frames: list[dict] = []
    status = 502
    payload: dict = {}
    saw_frame = False
    try:
        for attempt in range(2):
            retry = False
            aggregator = aggregator_for(protocol)
            frames = []
            saw_frame = False
            async with client.stream(
                "POST",
                url,
                content=body,
                headers=build_headers(),
                timeout=_STREAM_TIMEOUT,
            ) as resp:
                status = resp.status_code
                if resp.status_code == 401 and attempt == 0:
                    # Read the body even though a retry may follow: when the
                    # re-mint fails there is no second response, and dropping
                    # it would answer the client an empty 401.
                    retry = True
                    payload = _json_or_raw(await resp.aread())
                elif resp.status_code != 200:
                    payload = _json_or_raw(await resp.aread())
                else:
                    buffer = bytearray()
                    # Kept only until the first SSE frame decodes, so a proxy
                    # that ignores `stream: true` and answers one JSON
                    # envelope still reaches the client verbatim instead of
                    # aggregating to nothing.
                    prefix = bytearray()
                    async for chunk in resp.aiter_bytes():
                        if not saw_frame:
                            prefix.extend(chunk)
                        buffer.extend(chunk)
                        while True:
                            cut = buffer.find(b"\n")
                            if cut < 0:
                                break
                            line = bytes(buffer[: cut + 1])
                            del buffer[: cut + 1]
                            frame = _decode_data_line(line)
                            if frame is not None:
                                frames.append(frame)
                                aggregator.feed(frame)
                                saw_frame = True
                                prefix.clear()
                    if buffer:
                        frame = _decode_data_line(bytes(buffer))
                        if frame is not None:
                            frames.append(frame)
                            aggregator.feed(frame)
                            saw_frame = True
                            prefix.clear()
                    payload = (
                        aggregator.result()
                        if saw_frame
                        else _json_or_raw(bytes(prefix))
                    )
            if retry and await channel.refresh_slot_auth(slot, force=True):
                continue
            break
    except Exception:
        await channel.release_slot(slot, success=False)
        if meter is not None:
            meter.record(status=502, usage=None)
        raise

    err = parse_error_frame(frames)
    if err is not None:
        status, payload = error_response_for(protocol, err)
    await channel.release_slot(slot, success=err is None and status < 500)
    if meter is not None:
        usage = (
            aggregator.usage() if saw_frame else usage_from_payload(protocol, payload)
        )
        meter.record(status=status, usage=usage)
    return status, payload


def _json_or_raw(raw: bytes) -> dict:
    """Decode an upstream error body, keeping non-JSON payloads visible."""
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError):
        return {"raw": raw.decode("utf-8", errors="replace")[:2000]}
    return decoded if isinstance(decoded, dict) else {"raw": decoded}
