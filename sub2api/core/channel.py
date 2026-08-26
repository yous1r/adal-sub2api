"""Channel abstraction: the seam between the gateway and one agent backend.

To integrate a new channel, subclass :class:`BaseChannel`, set ``name``,
implement :meth:`BaseChannel._chat` as an async generator of normalized
events, and decorate the class with
:func:`sub2api.core.registry.register`. Everything else — lifecycle,
error normalization, session bookkeeping — is inherited.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar

from .errors import Sub2ApiError
from .types import ChatRequest, Event, TurnFailed


@dataclass(slots=True)
class ChannelConfig:
    """Per-channel settings resolved from environment/config by the app."""

    workspace: str = "."
    auth_token: str | None = None
    runtime_path: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


class BaseChannel(ABC):
    """Common turn pipeline shared by all channels.

    Common logic (do NOT override): :meth:`chat` wraps the subclass stream
    with lazy startup and error normalization; :meth:`start`/``close``
    manage the lifecycle once per process.

    Channel-specific (override): ``name``/``display_name``/``models``,
    :meth:`_chat`, optionally :meth:`_start` and :meth:`runtime_available`.
    """

    name: ClassVar[str]
    display_name: ClassVar[str] = ""
    models: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: ChannelConfig) -> None:
        self.config = config
        self._started = False

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        await self._start()
        self._started = True

    async def _start(self) -> None:
        """Hook for eager resource setup (imports, subprocess pools)."""

    async def close(self) -> None:
        """Release resources. Must be idempotent."""

    @property
    def started(self) -> bool:
        return self._started

    # -- turn pipeline -----------------------------------------------------

    @abstractmethod
    def _chat(self, request: ChatRequest) -> AsyncIterator[Event]:
        """Produce the normalized event stream for one turn.

        Implementations may raise; :meth:`chat` converts any exception into
        a terminal ``TurnFailed`` event so consumers never see raises.
        """
        yield  # pragma: no cover - makes this an async generator signature

    async def chat(self, request: ChatRequest) -> AsyncIterator[Event]:
        """Final pipeline: lazy-start + error normalization. Never raises."""
        await self.start()
        try:
            async for event in self._chat(request):
                yield event
        except Sub2ApiError as exc:
            yield TurnFailed(code=exc.code, message=str(exc))
        except Exception as exc:  # boundary normalization - deliberate breadth
            yield TurnFailed(code="internal_error", message=f"{type(exc).__name__}: {exc}")

    # -- introspection -----------------------------------------------------

    def resolve_workspace(self, request: ChatRequest) -> str | None:
        return request.workspace or self.config.workspace or None

    def runtime_available(self) -> bool:
        """Whether the backing runtime is installed/reachable."""
        return True

    async def health(self) -> dict[str, Any]:
        return {
            "channel": self.name,
            "display_name": self.display_name or self.name,
            "models": list(self.models),
            "ready": self.runtime_available(),
            "started": self.started,
        }
