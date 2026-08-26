"""Channel registry.

Channels self-register at import time via the :func:`register` decorator;
the application imports :mod:`sub2api.channels` once to discover all of
them. Adding a channel therefore never touches server code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import UnknownChannelError

if TYPE_CHECKING:  # pragma: no cover
    from .channel import BaseChannel, ChannelConfig

_REGISTRY: dict[str, type["BaseChannel"]] = {}


def register(cls: type["BaseChannel"]) -> type["BaseChannel"]:
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name` class attribute")
    if name in _REGISTRY:
        raise ValueError(f"channel `{name}` is already registered")
    _REGISTRY[name] = cls
    return cls


def get_channel_class(name: str) -> type["BaseChannel"]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownChannelError(
            f"unknown channel `{name}`; available: {', '.join(sorted(_REGISTRY))}"
        ) from None


def create_channel(name: str, config: "ChannelConfig") -> "BaseChannel":
    return get_channel_class(name)(config)


def available_channels() -> list[str]:
    return sorted(_REGISTRY)


def clear_registry() -> None:
    """Test-only helper."""
    _REGISTRY.clear()
