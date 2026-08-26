"""Normalized contracts shared by every channel and the HTTP server.

These types are the *only* thing channels and the server have in common:
a channel converts its upstream protocol into the event stream defined
here, and the server speaks nothing else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

PERMISSION_MODES = ("default", "acceptEdits", "yolo")


@dataclass(slots=True)
class ChatRequest:
    """A single agent turn, normalized across channels.

    ``session_id`` is the server-issued public conversation id; it is stable
    across channels. ``native_session_id`` is the upstream conversation id a
    channel uses to resume (e.g. AdaL's session uuid). Channels only ever
    read ``native_session_id``; the server resolves it from the session store.
    """

    prompt: str
    session_id: str | None = None
    native_session_id: str | None = None
    model: str | None = None
    workspace: str | None = None
    permission_mode: str = "default"
    enabled_tools: tuple[str, ...] | None = None
    images: tuple[str, ...] | None = None
    context_files: tuple[str, ...] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.permission_mode not in PERMISSION_MODES:
            raise ValueError(f"permission_mode must be one of {PERMISSION_MODES}")


class Event:
    """Base class for the normalized event stream every channel emits.

    Contract: a channel yields zero or more of text/thought/tool events and
    ends its stream with exactly one terminal event (``TurnCompleted`` or
    ``TurnFailed``). Errors are normalized into ``TurnFailed`` by the base
    channel pipeline — events never raise across the boundary.
    """

    type: ClassVar[str] = "event"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        data.update(asdict(self))
        return data


@dataclass(slots=True)
class SessionStarted(Event):
    """Emitted by the server (not channels) as the first SSE frame."""

    type: ClassVar[str] = "session.started"
    session_id: str = ""
    channel: str = ""


@dataclass(slots=True)
class TextDelta(Event):
    type: ClassVar[str] = "text.delta"
    text: str = ""


@dataclass(slots=True)
class ThoughtDelta(Event):
    type: ClassVar[str] = "thought.delta"
    text: str = ""


@dataclass(slots=True)
class MessageCompleted(Event):
    """A complete assistant reply segment. Either deltas OR completed
    messages per turn — never both for the same text."""

    type: ClassVar[str] = "message.completed"
    text: str = ""


@dataclass(slots=True)
class ToolStarted(Event):
    type: ClassVar[str] = "tool.started"
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCompleted(Event):
    type: ClassVar[str] = "tool.completed"
    name: str = ""
    status: str = "success"  # success | error
    result: Any = None


@dataclass(slots=True)
class TurnCompleted(Event):
    """Terminal success. ``session_id`` carries the channel-native id; the
    server adopts it so later turns can resume."""

    type: ClassVar[str] = "turn.completed"
    session_id: str | None = None
    model: str | None = None


@dataclass(slots=True)
class TurnFailed(Event):
    """Terminal failure with a code from :mod:`sub2api.core.errors`."""

    type: ClassVar[str] = "turn.failed"
    code: str = "internal_error"
    message: str = ""


TERMINAL_EVENTS = (TurnCompleted, TurnFailed)
