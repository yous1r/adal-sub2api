from __future__ import annotations

import pytest

from sub2api.core.aggregate import collect_answer
from sub2api.core.errors import UpstreamError
from sub2api.core.types import (
    ChatRequest,
    MessageCompleted,
    TextDelta,
    ThoughtDelta,
    TurnCompleted,
    TurnFailed,
)


async def _stream(items):
    for item in items:
        yield item


@pytest.mark.anyio
async def test_collect_joins_text_events_in_order():
    events = [
        ThoughtDelta(text="hmm"),  # not answer text
        TextDelta(text="Hello "),
        TextDelta(text="world"),
        MessageCompleted(text="!"),
        TurnCompleted(session_id="s1", model="m"),
    ]
    answer, terminal = await collect_answer(_stream(events))
    assert answer == "Hello world!"
    assert isinstance(terminal, TurnCompleted)
    assert terminal.session_id == "s1"


@pytest.mark.anyio
async def test_collect_captures_failure_terminal():
    events = [TextDelta(text="partial"), TurnFailed(code="upstream_failed", message="boom")]
    answer, terminal = await collect_answer(_stream(events))
    assert answer == "partial"
    assert isinstance(terminal, TurnFailed)
    assert terminal.code == "upstream_failed"


@pytest.mark.anyio
async def test_base_channel_normalizes_exceptions_into_turn_failed():
    from sub2api.core.channel import BaseChannel, ChannelConfig

    class Boom(BaseChannel):
        name = "boom"

        async def _chat(self, request):
            raise UpstreamError("exploded")
            yield  # pragma: no cover

    channel = Boom(ChannelConfig())
    answer, terminal = await collect_answer(channel.chat(ChatRequest(prompt="hi")))
    assert answer == ""
    assert isinstance(terminal, TurnFailed)
    assert terminal.code == "upstream_failed"
    assert "exploded" in terminal.message


@pytest.mark.anyio
async def test_base_channel_normalizes_unexpected_exceptions():
    from sub2api.core.channel import BaseChannel, ChannelConfig

    class Crash(BaseChannel):
        name = "crash"

        async def _chat(self, request):
            raise RuntimeError("disk on fire")
            yield  # pragma: no cover

    channel = Crash(ChannelConfig())
    _, terminal = await collect_answer(channel.chat(ChatRequest(prompt="hi")))
    assert isinstance(terminal, TurnFailed)
    assert terminal.code == "internal_error"
    assert "RuntimeError" in terminal.message
