from __future__ import annotations

import httpx
import pytest

from sub2api.server.proxy import forward_to_proxy


class _Channel:
    def __init__(self, responses: list[httpx.Response]):
        self._responses = iter(responses)
        self.released: list[tuple[object, bool]] = []
        self._client = httpx.AsyncClient(transport=httpx.MockTransport(self._handler))

    async def _handler(self, request: httpx.Request) -> httpx.Response:
        return next(self._responses)

    def proxy_headers(self, *args, **kwargs) -> dict[str, str]:
        return {}

    async def refresh_slot_auth(self, slot, force: bool = False) -> bool:
        return False

    async def release_slot(self, slot, success: bool) -> None:
        self.released.append((slot, success))


@pytest.mark.anyio
async def test_stream_upstream_error_returns_response_with_status_and_json():
    channel = _Channel(
        [httpx.Response(402, json={"error": {"message": "insufficient_credits"}})]
    )
    try:
        result = await forward_to_proxy(
            channel=channel,
            url="http://upstream.test/proxy",
            fwd_body=b"{}",
            slot="slot",
            sid="sid",
            target="target",
            provider="provider",
            stream=True,
            protocol="responses",
        )
        assert isinstance(result, httpx.Response)
        assert result.status_code == 402
        assert result.json()["error"]["message"] == "insufficient_credits"
        assert channel.released == [("slot", True)]
    finally:
        await channel._client.aclose()


@pytest.mark.anyio
async def test_stream_success_preserves_sse_bytes():
    sse = b'data: {"type":"response.completed"}\n\n'
    channel = _Channel([httpx.Response(200, content=sse)])
    try:
        result = await forward_to_proxy(
            channel=channel,
            url="http://upstream.test/proxy",
            fwd_body=b"{}",
            slot="slot",
            sid="sid",
            target="target",
            provider="provider",
            stream=True,
            protocol="responses",
        )
        assert not isinstance(result, httpx.Response)
        assert b"".join([chunk async for chunk in result]) == sse
        assert channel.released == [("slot", True)]
    finally:
        await channel._client.aclose()
