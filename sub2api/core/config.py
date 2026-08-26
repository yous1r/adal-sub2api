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
    channel_config: ChannelConfig = field(default_factory=ChannelConfig)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "AppSettings":
        env = os.environ if environ is None else environ
        options: dict[str, Any] = {}
        raw_options = env.get(f"{ENV_PREFIX}CHANNEL_OPTIONS")
        if raw_options:
            options = json.loads(raw_options)
        return cls(
            channel=env.get(f"{ENV_PREFIX}CHANNEL", "echo"),
            host=env.get(f"{ENV_PREFIX}HOST", "127.0.0.1"),
            port=int(env.get(f"{ENV_PREFIX}PORT", "8080")),
            channel_config=ChannelConfig(
                workspace=env.get(f"{ENV_PREFIX}WORKSPACE", "."),
                auth_token=env.get(f"{ENV_PREFIX}AUTH_TOKEN"),
                runtime_path=env.get(f"{ENV_PREFIX}RUNTIME_PATH"),
                options=options,
            ),
        )
