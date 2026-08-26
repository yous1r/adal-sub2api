from __future__ import annotations

import pytest

from sub2api.core import registry
from sub2api.core.channel import BaseChannel, ChannelConfig
from sub2api.core.errors import UnknownChannelError
from sub2api.core.sessions import SessionStore, SessionNotFoundError
from sub2api.core.types import ChatRequest, Event


def _channel_cls(name: str):
    class _Chan(BaseChannel):
        async def _chat(self, request):
            yield

    _Chan.name = name
    return _Chan


def test_register_and_create():
    cls = _channel_cls("reg-test-a")
    registry.register(cls)
    assert "reg-test-a" in registry.available_channels()
    instance = registry.create_channel("reg-test-a", ChannelConfig())
    assert isinstance(instance, BaseChannel)
    registry._REGISTRY.pop("reg-test-a")


def test_duplicate_registration_rejected():
    cls = _channel_cls("reg-test-b")
    registry.register(cls)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(cls)
    registry._REGISTRY.pop("reg-test-b")


def test_unknown_channel_lists_available():
    with pytest.raises(UnknownChannelError, match="echo"):
        registry.get_channel_class("does-not-exist")


def test_builtin_channels_registered():
    # importing sub2api.channels happens via server import; ensure present
    import sub2api.channels  # noqa: F401

    assert {"echo", "adal-cli", "adal-sdk"} <= set(registry.available_channels())


# --- sessions ---------------------------------------------------------------


def test_session_store_roundtrip():
    store = SessionStore()
    session = store.create("echo")
    assert session.turns == 0
    store.touch(session)
    assert store.get(session.id) is session
    store.adopt_native(session, "native-1", model="m1")
    assert session.native_session_id == "native-1"
    assert session.model == "m1"


def test_session_store_missing_raises():
    store = SessionStore()
    with pytest.raises(SessionNotFoundError):
        store.get("nope")


# --- contracts --------------------------------------------------------------


def test_chat_request_validates_permission_mode():
    with pytest.raises(ValueError, match="permission_mode"):
        ChatRequest(prompt="x", permission_mode="bogus")


def test_event_to_dict_includes_type_and_fields():
    from sub2api.core.types import ToolStarted

    event = ToolStarted(name="read", args={"path": "a.py"})
    assert event.to_dict() == {"type": "tool.started", "name": "read", "args": {"path": "a.py"}}
    assert isinstance(event, Event)
