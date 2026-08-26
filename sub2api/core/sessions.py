"""In-memory session registry.

Owns the public session-id namespace. Clients always use ``Session.id``;
the server translates it into the channel-native id before dispatching,
so channels never see server-side identifiers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import uuid4

from .errors import SessionNotFoundError


@dataclass(slots=True)
class Session:
    id: str
    channel: str
    native_session_id: str | None = None
    model: str | None = None
    turns: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.id,
            "channel": self.channel,
            "native_session_id": self.native_session_id,
            "model": self.model,
            "turns": self.turns,
        }


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, channel: str) -> Session:
        session = Session(id=uuid4().hex[:12], channel=channel)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"unknown session: {session_id}")
        return session

    def adopt_native(
        self, session: Session, native_session_id: str | None, model: str | None = None
    ) -> None:
        """Record the channel-native id returned by a completed turn."""
        if native_session_id:
            session.native_session_id = native_session_id
        if model:
            session.model = model

    def touch(self, session: Session) -> None:
        session.turns += 1
