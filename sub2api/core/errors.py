"""Unified error taxonomy.

Every failure surfaced to clients maps to one of these codes, no matter
which channel produced it. Channels raise subclasses of
:class:`Sub2ApiError`; the base-channel pipeline converts anything else
into ``internal_error``.
"""

from __future__ import annotations


class Sub2ApiError(Exception):
    code = "internal_error"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message or self.code)
        if code is not None:
            self.code = code


class UnknownChannelError(Sub2ApiError):
    code = "unknown_channel"


class RuntimeMissingError(Sub2ApiError):
    """The channel's runtime (CLI binary / SDK package) is not installed."""

    code = "runtime_missing"


class AuthError(Sub2ApiError):
    code = "auth_failed"


class UpstreamError(Sub2ApiError):
    """The upstream agent failed mid-turn."""

    code = "upstream_failed"


class ChannelNotReadyError(Sub2ApiError):
    code = "channel_not_ready"


class SessionNotFoundError(Sub2ApiError):
    code = "session_not_found"


class ChannelMismatchError(Sub2ApiError):
    """A session was reused against a different channel."""

    code = "channel_mismatch"
