"""Echo channel: minimal reference implementation and dev backend.

Doubles as the living template for new channels — it exercises every
normalized event type with no external runtime.
"""

from __future__ import annotations

from typing import AsyncIterator, ClassVar

from ..core.channel import BaseChannel
from ..core.registry import register
from ..core.types import (
    ChatRequest,
    Event,
    TextDelta,
    ThoughtDelta,
    ToolCompleted,
    ToolStarted,
    TurnCompleted,
)


@register
class EchoChannel(BaseChannel):
    name: ClassVar[str] = "echo"
    display_name: ClassVar[str] = "Echo (reference)"
    models: ClassVar[tuple[str, ...]] = ("echo-mini",)

    async def _chat(self, request: ChatRequest) -> AsyncIterator[Event]:
        yield ThoughtDelta(text=f"echoing {len(request.prompt)} chars")
        yield ToolStarted(name="inspect", args={"chars": len(request.prompt)})
        yield ToolCompleted(name="inspect", status="success", result=None)
        text = request.prompt
        for i in range(0, len(text), 7):
            yield TextDelta(text=text[i : i + 7])
        yield TurnCompleted(
            session_id=request.native_session_id or f"echo-{request.session_id}",
            model=request.model or self.models[0],
        )
