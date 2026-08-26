from __future__ import annotations

import httpx
import pytest

from sub2api.core.channel import ChannelConfig
from sub2api.core.config import AppSettings
from sub2api.server.app import create_app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def app():
    settings = AppSettings(
        channel="echo",
        host="testserver",
        port=0,
        channel_config=ChannelConfig(workspace="."),
    )
    return create_app(settings)


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


def parse_sse(text: str) -> list[dict]:
    frames = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if chunk.startswith("data: "):
            import json

            frames.append(json.loads(chunk[len("data: ") :]))
    return frames
