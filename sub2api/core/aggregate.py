"""Common stream consumption: fold an event stream into a final answer."""

from __future__ import annotations

from typing import AsyncIterator

from .types import TERMINAL_EVENTS, Event, TextDelta, MessageCompleted, TurnCompleted, TurnFailed

TEXT_EVENT_TYPES = (TextDelta.type, MessageCompleted.type)


async def collect_answer(
    events: AsyncIterator[Event],
) -> tuple[str, TurnCompleted | TurnFailed | None]:
    """Concatenate all text-bearing events and capture the terminal event.

    Returns ``(answer_text, terminal_event)`` where ``terminal_event`` is
    ``None`` only if the stream ended without a terminal event (a channel
    contract violation callers should treat as a failure).
    """
    parts: list[str] = []
    terminal: TurnCompleted | TurnFailed | None = None
    async for event in events:
        if event.type in TEXT_EVENT_TYPES:
            parts.append(event.text)
        elif isinstance(event, TERMINAL_EVENTS):
            terminal = event
    return "".join(parts), terminal
