"""Application settings resolved from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .channel import ChannelConfig

ENV_PREFIX = "SUB2API_"


@dataclass(slots=True)
class AppSettings:
    channel: str = "echo"
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str | None = None  # Bearer guard for /v1/* when set
    openai_permission_mode: str = "yolo"  # headless callers cannot approve tools
    enabled_tools: tuple[str, ...] | None = None  # deployment-wide tool whitelist
    proxy: str | None = None  # outbound HTTP(S) proxy for upstream
    web: bool = False  # mount the /admin management UI
    db: str | None = None  # SQLite path override (else SUB2API_DB)
    channel_config: ChannelConfig = field(default_factory=ChannelConfig)
    responses_only: bool = False  # use Responses upstream, including for Chat ingress

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> AppSettings:
        env = os.environ if environ is None else environ
        options: dict[str, Any] = {}
        raw_options = env.get(f"{ENV_PREFIX}CHANNEL_OPTIONS")
        if raw_options:
            options = json.loads(raw_options)
        enabled_tools: tuple[str, ...] | None = None
        raw_tools = env.get(f"{ENV_PREFIX}ENABLED_TOOLS")
        if raw_tools:
            enabled_tools = tuple(t.strip() for t in raw_tools.split(",") if t.strip())
        return cls(
            channel=env.get(f"{ENV_PREFIX}CHANNEL", "echo"),
            host=env.get(f"{ENV_PREFIX}HOST", "127.0.0.1"),
            port=int(env.get(f"{ENV_PREFIX}PORT", "8080")),
            api_key=env.get(f"{ENV_PREFIX}API_KEY") or None,
            openai_permission_mode=env.get(
                f"{ENV_PREFIX}OPENAI_PERMISSION_MODE", "yolo"
            ),
            enabled_tools=enabled_tools,
            proxy=env.get(f"{ENV_PREFIX}PROXY") or None,
            web=env.get(f"{ENV_PREFIX}WEB", "") == "1",
            responses_only=env.get(f"{ENV_PREFIX}RESPONSES_ONLY", "") == "1",
            db=env.get(f"{ENV_PREFIX}DB") or None,
            channel_config=ChannelConfig(
                workspace=env.get(f"{ENV_PREFIX}WORKSPACE", "."),
                auth_token=env.get(f"{ENV_PREFIX}AUTH_TOKEN"),
                runtime_path=env.get(f"{ENV_PREFIX}RUNTIME_PATH"),
                proxy=env.get(f"{ENV_PREFIX}PROXY") or None,
                options=options,
            ),
        )
