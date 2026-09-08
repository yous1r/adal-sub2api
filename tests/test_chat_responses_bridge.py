"""Exercise the Chat compatibility bridge through real gateway and SDK handlers."""

from __future__ import annotations

import json

import anyio
import httpx
import pytest
from openai import APIError, AsyncOpenAI

import sub2api.channels.adal_cloud as cloud
import sub2api.server.app as app_module
from sub2api.core.channel import ChannelConfig
from sub2api.core.config import AppSettings
from sub2api.core.pool import AccountConfig, AccountPool, PoolConfig


@pytest.fixture
async def bridge_gateway(monkeypatch):
    clients = []
    pools = []

    def create(handler, *, responses_only=True):
        config = PoolConfig(
            accounts=[AccountConfig(token="test-token", session_id="bridge-session")]
        )
        channel = cloud.AdalCloudChannel(ChannelConfig())
        channel._started = True
        channel._catalog = {
            "models": [
                {
                    "key": "openai-gpt-6-astra",
                    "model_id": "gpt-6-astra",
                    "provider": "openai",
                    "config_options": {
                        "effort": ["low", "medium", "high", "xhigh", "max"],
                        "effort_path": "reasoning.effort",
                    },
                }
            ]
        }
        channel._pool = AccountPool(config)
        channel._pool_signature = channel._pool_signature_for(config)
        channel._registered = {"bridge-session"}
        channel._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        pools.append(channel._pool)
        clients.append(channel._client)
        monkeypatch.setattr(cloud, "load_pool_config", lambda: config)
        monkeypatch.setattr(app_module, "create_channel", lambda *args: channel)
        app = app_module.create_app(
            AppSettings(channel="adal-cloud", responses_only=responses_only)
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
        )
        clients.append(client)
        return client, channel

    yield create
    for client in clients:
        await client.aclose()
    for pool in pools:
        await pool.close()


def _response(output, *, response_id="resp_tools"):
    return {
        "id": response_id,
        "object": "response",
        "created_at": 10,
        "model": "gpt-6-astra",
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 5,
            "total_tokens": 17,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }


def _sse(events):
    return b"".join(
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
        for event in events
    )


@pytest.mark.anyio
async def test_chat_sdk_completes_reasoning_tool_round_trip(bridge_gateway):
    turns = []
    tool_call = {
        "type": "function_call",
        "id": "fc_internal_item",
        "call_id": "call_multiply",
        "name": "multiply",
        "arguments": '{"a":6,"b":7}',
        "status": "completed",
    }

    def upstream(request):
        assert request.url.path == "/proxy/v1/responses"
        body = json.loads(request.content)
        assert "messages" not in body
        assert body["model"] == "gpt-6-astra"
        assert body["reasoning"]["effort"] == "high"
        turns.append(body)
        if len(turns) == 1:
            return httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "type": "reasoning",
                            "id": "rs_internal_item",
                            "summary": [
                                {"type": "summary_text", "text": "Use multiply."}
                            ],
                        },
                        tool_call,
                    ]
                ),
            )
        calls = [item for item in body["input"] if item.get("type") == "function_call"]
        results = [
            item for item in body["input"] if item.get("type") == "function_call_output"
        ]
        assert calls[0]["call_id"] == results[0]["call_id"] == "call_multiply"
        assert json.loads(calls[0]["arguments"]) == {"a": 6, "b": 7}
        assert results[0]["output"] == "42"
        message = {
            "type": "message",
            "id": "msg_answer",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "42", "annotations": []}],
        }
        completed = _response([message], response_id="resp_answer")
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=_sse(
                [
                    {
                        "type": "response.created",
                        "response": {
                            **completed,
                            "status": "in_progress",
                            "output": [],
                            "usage": None,
                        },
                    },
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {**message, "status": "in_progress", "content": []},
                    },
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "42",
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": message,
                    },
                    {"type": "response.completed", "response": completed},
                ]
            ),
        )

    client, _ = bridge_gateway(upstream)
    async with AsyncOpenAI(
        base_url="http://gateway.test/v1", api_key="test", http_client=client
    ) as sdk:
        messages = [{"role": "user", "content": "Multiply 6 by 7 using the tool."}]
        options = {
            "model": "openai-gpt-6-astra",
            "reasoning_effort": "high",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "multiply",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        }
        first = await sdk.chat.completions.create(messages=messages, **options)
        assert first.choices[0].finish_reason == "tool_calls"
        call = first.choices[0].message.tool_calls[0]
        assert call.id == "call_multiply"
        assert call.function.name == "multiply"
        assert first.choices[0].message.reasoning_content == "Use multiply."
        assert first.usage.prompt_tokens_details.cached_tokens == 4
        assert first.usage.completion_tokens_details.reasoning_tokens == 3
        args = json.loads(call.function.arguments)
        messages += [
            first.choices[0].message.model_dump(exclude_none=True),
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(args["a"] * args["b"]),
            },
        ]
        stream = await sdk.chat.completions.create(
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **options,
        )
        chunks = [chunk async for chunk in stream]
    assert (
        "".join(
            choice.delta.content or "" for chunk in chunks for choice in chunk.choices
        )
        == "42"
    )
    assert [
        choice.finish_reason
        for chunk in chunks
        for choice in chunk.choices
        if choice.finish_reason is not None
    ] == ["stop"]
    usage_chunks = [chunk for chunk in chunks if not chunk.choices]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].usage.total_tokens == 17
    assert len({chunk.id for chunk in chunks}) == 1
    assert len(turns) == 2


@pytest.mark.anyio
async def test_unrepresentable_choices_fail_before_upstream(bridge_gateway):
    requests = []

    def upstream(request):
        requests.append(request)
        return httpx.Response(500)

    client, _ = bridge_gateway(upstream)
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-6-astra",
            "messages": [{"role": "user", "content": "hello"}],
            "n": 2,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "n"
    assert not requests


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_bridge_keeps_upstream_http_errors(bridge_gateway, stream):
    error = {
        "error": {"message": "credit limit reached", "code": "insufficient_credits"}
    }
    client, _ = bridge_gateway(lambda request: httpx.Response(402, json=error))
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-6-astra",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
        },
    )
    assert response.status_code == 402
    assert response.json() == error


@pytest.mark.anyio
async def test_failed_responses_stream_raises_chat_sdk_error(bridge_gateway):
    failed = {
        "type": "response.failed",
        "response": {
            "id": "resp_failed",
            "status": "failed",
            "error": {"code": "rate_limit_exceeded", "message": "quota exhausted"},
        },
    }
    client, _ = bridge_gateway(
        lambda request: httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=_sse([failed])
        )
    )
    async with AsyncOpenAI(
        base_url="http://gateway.test/v1", api_key="test", http_client=client
    ) as sdk:
        stream = await sdk.chat.completions.create(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        chunks = []
        with pytest.raises(APIError) as failure:
            async for chunk in stream:
                chunks.append(chunk)
    assert failure.value.code == "rate_limit_exceeded"
    assert all(
        choice.finish_reason is None for chunk in chunks for choice in chunk.choices
    )


@pytest.mark.anyio
async def test_cancelling_chat_stream_closes_transport_and_releases_capacity(
    bridge_gateway,
):
    reading = anyio.Event()
    closed = anyio.Event()

    class HangingResponse(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield _sse([{"type": "response.in_progress", "response": {}}])
            reading.set()
            await anyio.sleep_forever()

        async def aclose(self):
            await anyio.lowlevel.checkpoint()
            closed.set()

    client, channel = bridge_gateway(
        lambda request: httpx.Response(200, stream=HangingResponse())
    )

    async def consume():
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-6-astra",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    async with anyio.create_task_group() as group:
        group.start_soon(consume)
        await reading.wait()
        group.cancel_scope.cancel()

    assert closed.is_set()
    assert channel._pool.slots[0].in_flight == 0
