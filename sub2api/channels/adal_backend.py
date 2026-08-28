"""AdaL channel backed by the backend HTTP API (`/webthinker/stream` SSE).

A persistent `adal --sdk-runtime` subprocess hosts a local uvicorn server
whose `/webthinker/stream` endpoint returns SSE events.  This channel talks
to that endpoint directly via **httpx** (no curl subprocess), caches
model/effort state to skip redundant API calls, and streams SSE frames
as they arrive.

Protocol (reverse-engineered from `adal --sdk-runtime`):

  initialize  →  {"type":"initialize","protocol_version":1, ...}
  ready       ←  {"type":"ready","session_id":"...","protocol_version":1}
  query       →  {"type":"query","input":"...","model":"...","thinking_effort":"..."}
  events      ←  {"type":"session",...}
                 {"type":"raw_response_event","data":{"answer_text":"..."}}
                 {"type":"assistant.message.completed","message":{"content":"..."}}
                 {"type":"command.completed","result":{...}}
  shutdown    →  {"type":"shutdown"}

The backend also exposes `/model/config` (POST `{"effort":"..."}`),
`/model/switch` (POST `{"model_name":"..."}`), and `/webthinker/cancel`.

Docs: https://docs.sylph.ai/features/headless-mode
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any, AsyncIterator, ClassVar

import httpx

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
from .adal_cli import load_catalog

DEFAULT_RUNTIME = "adal"
SDK_PROTOCOL_VERSION = 1
CREDS_PATH = Path.home() / ".adal" / "adal_oauth_creds.json"
DEFAULT_MODEL = "anthropic-claude-sonnet-5"


def _read_access_token() -> str | None:
    """Read the cached OAuth access token (shared with the CLI channel)."""
    try:
        data = json.loads(CREDS_PATH.read_text(encoding="utf-8"))
        return data.get("access_token")
    except (OSError, ValueError, KeyError):
        return None


def parse_sse_frame(raw: str) -> Event | None:
    """Translate one SSE ``data: {...}`` frame into a normalized event.

    Returns ``None`` for non-data frames, heartbeat, and unknown types.
    Raises :class:`UpstreamError` for backend error frames.
    """
    if not raw.startswith("data: "):
        return None
    payload = raw[6:]
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    typ = obj.get("type") or obj.get("name") or ""

    if typ == "assistant.message.completed":
        msg = obj.get("message", {})
        return MessageCompleted(text=msg.get("content", ""))

    # raw_response_event carries the full answer_text; emit as MessageCompleted
    # so the OpenAI route can surface it immediately (no tool events follow
    # in this frame).
    if typ == "raw_response_event":
        data = obj.get("data")
        if isinstance(data, dict):
            answer = data.get("answer_text")
            if answer:
                return MessageCompleted(text=answer)
        return None

    if typ == "error":
        raise UpstreamError(obj.get("message", "unknown upstream error"))

    if typ == "complete":
        return TurnCompleted(
            session_id=obj.get("session_id"),
            model=obj.get("model"),
        )

    # run_item_stream_event may carry tool calls; map when present.
    if typ == "run_item_stream_event":
        item = obj.get("item", {})
        item_type = item.get("type", "")
        if item_type == "tool_call":
            d = item.get("data", {})
            return ToolStarted(name=d.get("name", ""), args=d.get("args") or {})
        if item_type == "tool_result":
            d = item.get("data", {})
            return ToolCompleted(
                name=d.get("name", ""), status=d.get("status") or "success"
            )
        return None

    return None


@register
class AdalBackendChannel(BaseChannel):
    """Persistent-backend HTTP channel: one ``adal --sdk-runtime`` subprocess.

    Faster than the CLI channel (no per-request spawn) and stream-friendly
    (SSE frames arrive as the backend produces them, not buffered to the
    end of the process).  Uses **httpx** for all HTTP calls — no curl
    subprocess overhead — and caches model/effort state to skip redundant
    ``/model/switch`` and ``/model/config`` round trips.
    """

    name: ClassVar[str] = "adal-backend"
    display_name: ClassVar[str] = "AdaL (backend HTTP API)"
    models: ClassVar[tuple[str, ...]] = (
        "anthropic-claude-sonnet-4-6",
        "anthropic-claude-opus-4-6",
        "anthropic-claude-sonnet-5",
        "anthropic-claude-opus-5",
    )

    _proc: asyncio.subprocess.Process | None
    _port: int | None
    _session_id: str | None
    _client: httpx.AsyncClient | None
    _current_model: str | None
    _current_effort: str | None

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._proc = None
        self._port = None
        self._session_id = None
        self._token = _read_access_token() or self.config.auth_token
        self._client = None
        self._current_model = None
        self._current_effort = None

    # -- lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        catalog = load_catalog()
        if catalog:
            self.models = catalog
        await self._launch_backend()
        # httpx client with long read timeout for LLM streaming
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=5.0),
            base_url=self.base_url,
            headers=self._headers(),
        )

    async def _launch_backend(self) -> None:
        """Spawn ``adal --sdk-runtime``, initialize, discover port."""
        runtime = self.config.runtime_path or DEFAULT_RUNTIME
        self._proc = await asyncio.create_subprocess_exec(
            runtime, "--sdk-runtime",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await self._send({"type": "initialize", "protocol_version": SDK_PROTOCOL_VERSION,
                          "workspace": self.config.workspace or ".",
                          "permission_mode": "yolo",
                          "model": DEFAULT_MODEL})
        ready = await asyncio.wait_for(self._read_line(), timeout=30)
        obj = json.loads(ready)
        if obj.get("type") != "ready":
            raise UpstreamError(f"backend did not become ready: {ready[:200]}")
        self._session_id = obj.get("session_id")
        self._current_model = DEFAULT_MODEL

        # Discover the backend's HTTP port from ss output.
        self._port = await self._discover_port()
        if not self._port:
            raise UpstreamError("could not discover adal-backend port")

    async def _discover_port(self) -> int | None:
        """Find the ephemeral port the backend uvicorn server listens on.

        The ``adal --sdk-runtime`` launcher spawns a child ``adal-backend``
        process with its own PID, so we match on the process name rather
        than the parent PID.
        """
        try:
            result = await asyncio.create_subprocess_exec(
                "ss", "-tlnp",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            for line in stdout.decode().splitlines():
                if "adal-backend" in line:
                    m = re.search(r"127\.0\.0\.1:(\d+)", line)
                    if m:
                        return int(m.group(1))
        except (OSError, ValueError):
            pass
        return None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._proc is not None:
            try:
                await self._send({"type": "shutdown"})
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
            self._proc = None
        self._port = None
        self._session_id = None

    # -- stdio helpers -----------------------------------------------------

    async def _send(self, obj: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self._proc.stdin.drain()

    async def _read_line(self) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        line = await self._proc.stdout.readline()
        if not line:
            raise UpstreamError("adal-backend stdin closed unexpectedly")
        return line.decode().strip()

    # -- HTTP helpers ------------------------------------------------------

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise UpstreamError("backend not started")
        return f"http://127.0.0.1:{self._port}"

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    async def _ensure_model(self, model: str | None, effort: str | None) -> None:
        """Switch model + set effort only when changed (cached state)."""
        if model and model != self._current_model:
            try:
                assert self._client is not None
                await self._client.post("/model/switch",
                                        json={"model_name": model}, timeout=10)
                self._current_model = model
            except (httpx.HTTPError, UpstreamError):
                pass  # non-fatal: backend may already be on this model
        if effort and effort != self._current_effort:
            try:
                assert self._client is not None
                await self._client.post("/model/config",
                                        json={"effort": effort}, timeout=10)
                self._current_effort = effort
            except (httpx.HTTPError, UpstreamError):
                pass  # non-fatal: some models don't support effort

    # -- turn pipeline -----------------------------------------------------

    def runtime_available(self) -> bool:
        runtime = self.config.runtime_path or DEFAULT_RUNTIME
        if "/" in runtime or "\\" in runtime or ":" in runtime:
            return True
        return shutil.which(runtime) is not None

    async def _chat(self, request: ChatRequest) -> AsyncIterator[Event]:
        assert self._client is not None and self._port is not None

        # Skip redundant model/effort round trips via cached state.
        await self._ensure_model(request.model, request.thinking_effort)

        # POST /webthinker/stream and parse SSE via httpx streaming.
        got_answer = False
        buf = b""
        async with self._client.stream("POST", "/webthinker/stream",
                                       json={"query": request.prompt}) as resp:
            async for chunk in resp.aiter_bytes(4096):
                buf += chunk
                while b"\n\n" in buf:
                    frame_bytes, buf = buf.split(b"\n\n", 1)
                    frame = frame_bytes.strip()
                    if not frame:
                        continue
                    try:
                        event = parse_sse_frame(frame.decode("utf-8", errors="replace"))
                    except UpstreamError:
                        raise
                    if event is not None:
                        if isinstance(event, MessageCompleted):
                            got_answer = True
                        yield event

        if not got_answer:
            raise UpstreamError("backend stream ended without an answer event")

    # -- health ------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        base = await super().health()
        base["backend_pid"] = self._proc.pid if self._proc else None
        base["backend_port"] = self._port
        base["current_model"] = self._current_model
        return base
