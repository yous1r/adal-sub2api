from __future__ import annotations

from sub2api.channels.adal_sdk import map_event
from sub2api.core.types import (
    MessageCompleted,
    TextDelta,
    ThoughtDelta,
    ToolCompleted,
    ToolStarted,
    TurnCompleted,
    TurnFailed,
)


def test_assistant_delta_maps_to_text_delta():
    assert map_event({"type": "assistant.delta", "text": "he"}) == [TextDelta(text="he")]


def test_message_completed_extracts_content():
    raw = {"type": "assistant.message.completed", "message": {"content": "done"}}
    assert map_event(raw) == [MessageCompleted(text="done")]


def test_thought_delta():
    assert map_event({"type": "thought.delta", "text": "t"}) == [ThoughtDelta(text="t")]


def test_tool_events_roundtrip():
    started = map_event({"type": "tool.started", "name": "bash", "args": {"cmd": "ls"}})
    assert started == [ToolStarted(name="bash", args={"cmd": "ls"})]
    completed = map_event({"type": "tool.completed", "name": "bash", "status": "error"})
    assert completed == [ToolCompleted(name="bash", status="error", result=None)]


def test_command_completed_maps_to_turn_completed():
    assert map_event({"type": "command.completed"}) == [TurnCompleted()]


def test_command_failed_maps_to_turn_failed():
    raw = {"type": "command.failed", "error": {"message": "quota exceeded"}}
    assert map_event(raw) == [TurnFailed(code="upstream_failed", message="quota exceeded")]


def test_ui_error_message_maps_to_turn_failed():
    raw = {"type": "ui.message.appended", "message": {"level": "error", "text": "bad"}}
    assert map_event(raw) == [TurnFailed(code="upstream_failed", message="bad")]


def test_ui_info_message_is_ignored():
    assert map_event({"type": "ui.message.appended", "message": {"level": "info"}}) == []


def test_unknown_event_yields_nothing():
    assert map_event({"type": "command.progress"}) == []


def test_sdk_runtime_probe_without_package(monkeypatch):
    from sub2api.channels.adal_sdk import AdalSdkChannel
    from sub2api.core.channel import ChannelConfig

    monkeypatch.setattr("sub2api.channels.adal_sdk.CREDENTIALS_PATH", __import__("pathlib").Path("/nonexistent"))
    channel = AdalSdkChannel(ChannelConfig())
    assert channel.runtime_available() is False
