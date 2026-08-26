"""AdaL channel backed by the headless CLI (`adal -q ... -o stream-json`).

Spawns one `adal` subprocess per turn and translates its NDJSON stream
into normalized events. Requires the AdaL CLI installed and authenticated
once interactively (`adal`) — credentials are then reused from
`~/.adal/adal_oauth_creds.json`, so subscription quota is consumed.

Docs: https://docs.sylph.ai/features/headless-mode
"""

from __future__ import annotations

import asyncio
import json
import shutil
from typing import AsyncIterator, ClassVar

from ..core.channel import BaseChannel
from ..core.errors import UpstreamError
from ..core.registry import register
from ..core.types import (
    ChatRequest,
    Event,
    MessageCompleted,
    ToolCompleted,
    ToolStarted,
    TurnCompleted,
    TurnFailed,
)

DEFAULT_RUNTIME = "adal"


def build_args(runtime: str, request: ChatRequest) -> list[str]:
    """Map a normalized request onto headless CLI flags.

    Note: the CLI exposes no flag for `default`/`acceptEdits` permission
    modes; only `yolo` is passed through.
    """
    args = [runtime, "-q", request.prompt, "-o", "stream-json"]
    if request.model:
        args += ["-m", request.model]
    if request.native_session_id:
        args += ["-r", request.native_session_id]
    if request.enabled_tools:
        args += ["--enabled-default-tools", ",".join(request.enabled_tools)]
    if request.permission_mode == "yolo":
        args += ["--yolo"]
    return args


def parse_line(line: str) -> Event | None:
    """Translate one NDJSON line into a normalized event.

    Returns None for blank/unparsable/unknown lines. Raises
    :class:`UpstreamError` for upstream `error` events so the caller can
    abort the subprocess; the base pipeline converts that into TurnFailed.
    """
    line = line.strip()
    if not line:
        return None
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None
    kind = raw.get("type")
    if kind == "tool_call":
        return ToolStarted(name=raw.get("name", ""), args=raw.get("args") or {})
    if kind == "tool_result":
        return ToolCompleted(
            name=raw.get("name", ""), status=raw.get("status") or "success"
        )
    if kind == "answer":
        return MessageCompleted(text=raw.get("content", ""))
    if kind == "error":
        raise UpstreamError(raw.get("message", "unknown upstream error"))
    if kind == "complete":
        return TurnCompleted(session_id=raw.get("session_id"), model=raw.get("model"))
    return None


@register
class AdalCliChannel(BaseChannel):
    name: ClassVar[str] = "adal-cli"
    display_name: ClassVar[str] = "AdaL (headless CLI)"
    models: ClassVar[tuple[str, ...]] = (
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-6",
    )

    @property
    def runtime(self) -> str:
        return self.config.runtime_path or DEFAULT_RUNTIME

    def runtime_available(self) -> bool:
        if "/" in self.runtime or "\\" in self.runtime or ":" in self.runtime:
            return True  # explicit path; existence checked at spawn time
        return shutil.which(self.runtime) is not None

    async def _chat(self, request: ChatRequest) -> AsyncIterator[Event]:
        args = build_args(self.runtime, request)
        cwd = self.resolve_workspace(request)
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdout is not None and proc.stderr is not None
        try:
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                try:
                    event = parse_line(line)
                except UpstreamError:
                    proc.kill()
                    raise
                if event is not None:
                    yield event
        finally:
            returncode = await proc.wait()
        if returncode != 0:
            stderr = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
            raise UpstreamError(f"adal exited with code {returncode}: {stderr[-500:]}")
