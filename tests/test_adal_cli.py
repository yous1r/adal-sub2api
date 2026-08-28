from __future__ import annotations

import json
import shlex
import sys
import pytest

from sub2api.channels.adal_cli import AdalCliChannel, build_args, load_catalog, parse_line
from sub2api.core.aggregate import collect_answer
from sub2api.core.channel import ChannelConfig
from sub2api.core.errors import UpstreamError
from sub2api.core.types import (
    ChatRequest,
    MessageCompleted,
    ToolCompleted,
    ToolStarted,
    TurnCompleted,
    TurnFailed,
)


def test_build_args_full_mapping():
    request = ChatRequest(
        prompt="do it",
        native_session_id="sess-9",
        model="anthropic-claude-sonnet-4-6",
        permission_mode="yolo",
        enabled_tools=("Read", "Search"),
    )
    assert build_args("adal", request) == [
        "adal", "-q", "do it", "-o", "stream-json",
        "-m", "anthropic-claude-sonnet-4-6",
        "-r", "sess-9",
        "--enabled-default-tools", "Read,Search",
        "--yolo",
    ]


def test_build_args_minimal():
    assert build_args("adal", ChatRequest(prompt="hi")) == ["adal", "-q", "hi", "-o", "stream-json"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (json.dumps({"type": "tool_call", "name": "search", "args": {"pattern": "x"}}), ToolStarted),
        (json.dumps({"type": "tool_result", "name": "search", "status": "success"}), ToolCompleted),
        (json.dumps({"type": "answer", "content": "the answer"}), MessageCompleted),
        (
            json.dumps({"type": "complete", "exit_code": 0, "session_id": "abc", "model": "m"}),
            TurnCompleted,
        ),
    ],
)
def test_parse_line_maps_event_types(line, expected):
    assert isinstance(parse_line(line), expected)


def test_parse_line_complete_fields():
    event = parse_line(json.dumps({"type": "complete", "session_id": "abc", "model": "claude"}))
    assert isinstance(event, TurnCompleted)
    assert event.session_id == "abc"
    assert event.model == "claude"


def test_parse_line_error_raises_upstream():
    with pytest.raises(UpstreamError, match="auth expired"):
        parse_line(json.dumps({"type": "error", "message": "auth expired"}))


@pytest.mark.parametrize("line", ["", "   ", "not json", json.dumps({"type": "mystery"})])
def test_parse_line_ignores_noise(line):
    assert parse_line(line) is None


def _make_fake_runtime(tmp_path, body: str) -> str:
    """Create a platform-appropriate executable that runs `body` as Python."""
    inner = tmp_path / "_fake_inner.py"
    inner.write_text(body)
    if sys.platform == "win32":
        launcher = tmp_path / "adal.bat"
        launcher.write_text(f"@echo off\r\n{sys.executable} {inner} %*\r\n")
    else:
        launcher = tmp_path / "adal"
        launcher.write_text("#!/bin/sh\nexec %s %s \"$@\"\n" % (shlex.quote(sys.executable), shlex.quote(str(inner))))
        launcher.chmod(0o755)
    return str(launcher)


SUCCESS_BODY = """\
import json, sys
assert '-q' in sys.argv and 'hi' in sys.argv
for line in [
    {'type': 'tool_call', 'name': 'read', 'args': {'path': 'a.py'}},
    {'type': 'tool_result', 'name': 'read', 'status': 'success'},
    {'type': 'answer', 'content': 'world'},
    {'type': 'complete', 'exit_code': 0, 'session_id': 'native-7', 'model': 'claude'},
]:
    print(json.dumps(line), flush=True)
"""


@pytest.mark.anyio
async def test_cli_channel_end_to_end_with_fake_runtime(tmp_path):
    runtime = _make_fake_runtime(tmp_path, SUCCESS_BODY)
    channel = AdalCliChannel(ChannelConfig(runtime_path=runtime))

    events = [e async for e in channel.chat(ChatRequest(prompt="hi"))]

    assert not isinstance(events[-1], TurnFailed)
    completed = next(e for e in events if isinstance(e, TurnCompleted))
    assert completed.session_id == "native-7"
    assert completed.model == "claude"
    answers = [e for e in events if isinstance(e, MessageCompleted)]
    assert [a.text for a in answers] == ["world"]
    tools = [(e.name, type(e).__name__) for e in events if isinstance(e, (ToolStarted, ToolCompleted))]
    assert tools == [("read", "ToolStarted"), ("read", "ToolCompleted")]


@pytest.mark.anyio
async def test_cli_channel_resume_passes_native_session(tmp_path):
    captured = tmp_path / "captured.json"

    body = f"""\
import json, sys
json.dump(sys.argv, open({str(captured)!r}, 'w'))
print(json.dumps({{'type': 'complete', 'exit_code': 0, 'session_id': 'native-7'}}))
"""
    runtime = _make_fake_runtime(tmp_path, body)
    channel = AdalCliChannel(ChannelConfig(runtime_path=runtime))

    _, terminal = await collect_answer(
        channel.chat(ChatRequest(prompt="again", native_session_id="native-7"))
    )
    assert isinstance(terminal, TurnCompleted)
    argv = json.loads(captured.read_text())
    assert "-r" in argv and argv[argv.index("-r") + 1] == "native-7"


@pytest.mark.anyio
async def test_cli_channel_surfaces_nonzero_exit(tmp_path):
    body = "import sys\nsys.exit(3)\n"
    runtime = _make_fake_runtime(tmp_path, body)
    channel = AdalCliChannel(ChannelConfig(runtime_path=runtime))

    _, terminal = await collect_answer(channel.chat(ChatRequest(prompt="hi")))
    assert isinstance(terminal, TurnFailed)
    assert terminal.code == "upstream_failed"
    assert "exited with code 3" in terminal.message


@pytest.mark.anyio
async def test_cli_channel_upstream_error_event_aborts(tmp_path):
    body = "import json\nprint(json.dumps({'type': 'error', 'message': 'quota exhausted'}))\n"
    runtime = _make_fake_runtime(tmp_path, body)
    channel = AdalCliChannel(ChannelConfig(runtime_path=runtime))

    _, terminal = await collect_answer(channel.chat(ChatRequest(prompt="hi")))
    assert isinstance(terminal, TurnFailed)
    assert terminal.code == "upstream_failed"
    assert "quota exhausted" in terminal.message


def test_load_catalog_reads_keys(tmp_path):
    catalog = tmp_path / "model_catalog.json"
    catalog.write_text(json.dumps({
        "models": [
            {"key": "anthropic-claude-opus-5", "display_name": "Opus 5"},
            {"key": "openai-gpt-5.6-sol"},
            {"display_name": "no key -> skipped"},
        ],
    }))
    assert load_catalog(catalog) == ("anthropic-claude-opus-5", "openai-gpt-5.6-sol")


@pytest.mark.parametrize("payload", ["", "not json", '{"models": "bogus"}', "{}"])
def test_load_catalog_tolerates_garbage(tmp_path, payload):
    catalog = tmp_path / "model_catalog.json"
    catalog.write_text(payload)
    assert load_catalog(catalog) == ()


def test_load_catalog_missing_file(tmp_path):
    assert load_catalog(tmp_path / "absent.json") == ()


@pytest.mark.anyio
async def test_channel_start_adopts_instance_models(tmp_path):
    catalog = tmp_path / "model_catalog.json"
    catalog.write_text(json.dumps({"models": [{"key": "brand-new-pro-model"}]}))
    channel = AdalCliChannel(ChannelConfig(runtime_path=_make_fake_runtime(
        tmp_path, SUCCESS_BODY)))
    # Point the module-level default used by _start at the temp catalog.
    import sub2api.channels.adal_cli as mod

    original = mod.MODEL_CATALOG_PATH
    mod.MODEL_CATALOG_PATH = catalog
    try:
        await channel.start()
        assert channel.models == ("brand-new-pro-model",)
        assert AdalCliChannel.models != ("brand-new-pro-model",)  # class untouched
    finally:
        mod.MODEL_CATALOG_PATH = original


def test_build_args_includes_thinking_effort():
    request = ChatRequest(
        prompt="think hard",
        model="anthropic-claude-opus-5",
        permission_mode="yolo",
        thinking_effort="max",
    )
    args = build_args("adal", request)
    assert "--thinking-effort" in args
    assert args[args.index("--thinking-effort") + 1] == "max"
    assert "--yolo" in args


def test_build_args_omits_thinking_effort_when_none():
    request = ChatRequest(prompt="quick", model="anthropic-claude-sonnet-5")
    args = build_args("adal", request)
    assert "--thinking-effort" not in args


def test_chat_request_rejects_invalid_thinking_effort():
    with pytest.raises(ValueError, match="thinking_effort"):
        ChatRequest(prompt="hi", thinking_effort="turbo")


def test_chat_request_accepts_valid_thinking_effort():
    for level in ("low", "medium", "high", "max"):
        r = ChatRequest(prompt="hi", thinking_effort=level)
        assert r.thinking_effort == level


# -- adal_backend SSE frame parser tests -------------------------------------

from sub2api.channels.adal_backend import parse_sse_frame


def test_parse_sse_frame_message_completed():
    raw = 'data: {"type":"assistant.message.completed","message":{"content":"hello world"}}'
    event = parse_sse_frame(raw)
    assert isinstance(event, MessageCompleted)
    assert event.text == "hello world"


def test_parse_sse_frame_raw_response_event():
    raw = 'data: {"type":"raw_response_event","data":{"answer_text":"42"}}'
    event = parse_sse_frame(raw)
    assert isinstance(event, MessageCompleted)
    assert event.text == "42"


def test_parse_sse_frame_ignores_non_data():
    assert parse_sse_frame(": heartbeat") is None
    assert parse_sse_frame("") is None


def test_parse_sse_frame_ignores_unknown_type():
    raw = 'data: {"type":"heartbeat","timestamp":123}'
    assert parse_sse_frame(raw) is None


def test_parse_sse_frame_error_raises():
    raw = 'data: {"type":"error","message":"boom"}'
    with pytest.raises(Exception, match="boom"):
        parse_sse_frame(raw)


def test_parse_sse_frame_complete_event():
    raw = 'data: {"type":"complete","session_id":"s1","model":"anthropic-claude-opus-5"}'
    event = parse_sse_frame(raw)
    assert isinstance(event, TurnCompleted)
    assert event.session_id == "s1"
    assert event.model == "anthropic-claude-opus-5"
