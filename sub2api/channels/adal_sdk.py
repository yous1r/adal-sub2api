"""AdaL channel backed by the official Python Agent SDK.

The SDK spawns `adal --sdk-runtime` and speaks NDJSON over stdio; this
adapter maps its event dicts onto the normalized stream. Authentication
reuses the CLI's cached OAuth credentials (`~/.adal/adal_oauth_creds.json`)
or an explicit JWT via ``SUB2API_AUTH_TOKEN`` — either way the AdaL
subscription pays for usage.

Docs: https://docs.sylph.ai/sdk/quickstart
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, AsyncIterator, ClassVar

from ..core.channel import BaseChannel
from ..core.errors import RuntimeMissingError
from ..core.registry import register
from ..core.types import (
    ChatRequest,
    Event,
    MessageCompleted,
    TextDelta,
    ThoughtDelta,
    ToolCompleted,
    ToolStarted,
    TurnCompleted,
    TurnFailed,
)

SDK_PACKAGE = "adal_agent_sdk"
CREDENTIALS_PATH = Path.home() / ".adal" / "adal_oauth_creds.json"


def map_event(raw: dict[str, Any]) -> list[Event]:
    """Pure translation of one SDK event dict into zero or more normalized
    events. Kept module-level and side-effect-free for direct unit testing."""
    kind = raw.get("type")
    if kind == "assistant.delta":
        return [TextDelta(text=raw.get("text", ""))]
    if kind == "assistant.message.completed":
        message = raw.get("message") or {}
        return [MessageCompleted(text=message.get("content", ""))]
    if kind == "thought.delta":
        return [ThoughtDelta(text=raw.get("text", ""))]
    if kind == "tool.started":
        return [ToolStarted(name=raw.get("name", ""), args=raw.get("args") or {})]
    if kind == "tool.completed":
        return [
            ToolCompleted(
                name=raw.get("name", ""),
                status=raw.get("status") or "success",
                result=raw.get("result"),
            )
        ]
    if kind == "command.completed":
        return [TurnCompleted()]
    if kind == "command.failed":
        error = raw.get("error") or {}
        return [TurnFailed(code="upstream_failed", message=error.get("message", "unknown"))]
    if kind == "ui.message.appended":
        message = raw.get("message") or {}
        if message.get("level") == "error":
            return [TurnFailed(code="upstream_failed", message=message.get("text", ""))]
        return []
    return []


@register
class AdalSdkChannel(BaseChannel):
    name: ClassVar[str] = "adal-sdk"
    display_name: ClassVar[str] = "AdaL (Python SDK)"
    models: ClassVar[tuple[str, ...]] = (
        "anthropic-claude-sonnet-4-6",
        "anthropic-claude-opus-4-6",
        "anthropic-claude-sonnet-5",
        "anthropic-claude-opus-5",
    )

    async def _start(self) -> None:
        if importlib.util.find_spec(SDK_PACKAGE) is None:
            raise RuntimeMissingError(
                f"`{SDK_PACKAGE}` is not installed; "
                "run: pip install git+https://github.com/SylphAI-Inc/adal-sdk.git"
            )
        self._sdk = importlib.import_module(SDK_PACKAGE)
        from .adal_cli import load_catalog

        catalog = load_catalog()
        if catalog:
            self.models = catalog  # instance-level: newest Pro catalog wins

    def runtime_available(self) -> bool:
        if importlib.util.find_spec(SDK_PACKAGE) is None:
            return False
        return bool(self.config.auth_token) or CREDENTIALS_PATH.exists()

    def _build_options(self, request: ChatRequest) -> dict[str, Any]:
        options: dict[str, Any] = {
            "workspace": self.resolve_workspace(request) or ".",
            "permission_mode": request.permission_mode,
        }
        if request.model:
            options["model"] = request.model
        if request.native_session_id:
            options["session_id"] = request.native_session_id
        if self.config.auth_token:
            options["auth_token"] = self.config.auth_token
        if request.enabled_tools:
            options["enabled_default_tools"] = list(request.enabled_tools)
        return options

    async def _chat(self, request: ChatRequest) -> AsyncIterator[Event]:
        client = self._sdk.AdalAgentClient(self._build_options(request))
        try:
            await client.query(
                request.prompt,
                context_files=list(request.context_files or []),
                images=list(request.images or []),
            )
            async for raw in client.receive_events():
                produced = map_event(raw)
                terminal = next((e for e in produced if isinstance(e, (TurnCompleted, TurnFailed))), None)
                for event in produced:
                    yield event
                if terminal is not None:
                    break
        finally:
            await client.close()
