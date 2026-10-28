"""A disconnect must not release capacity while response cleanup is unfinished."""

import asyncio
import json

import httpx
import pytest
from test_gateway import AUTH, QUESTION, completion, gateway, make_settings


class SlowCleanupStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes | None = None):
        self.body = body
        self.body_started = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.allow_cleanup = asyncio.Event()
        self.cleanup_finished = asyncio.Event()

    async def __aiter__(self):
        self.body_started.set()
        if self.body is None:
            await asyncio.Event().wait()
        else:
            yield self.body

    async def aclose(self):
        self.cleanup_started.set()
        # Longer than the adapter's deadline. The event also keeps this test
        # deterministic if the test coroutine is scheduled after the sleep ends.
        await asyncio.sleep(0.12)
        await self.allow_cleanup.wait()
        self.cleanup_finished.set()


def request_scope():
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", AUTH["Authorization"].encode()),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }


async def assert_slot_held_past_deadline(task, app, stream):
    await asyncio.wait_for(stream.cleanup_started.wait(), timeout=1)
    # Cancellation has already reached response cleanup. Wait beyond the
    # original 40 ms budget: a second deadline cancellation used to interrupt it.
    await asyncio.sleep(0.06)
    assert not task.done()
    assert not stream.cleanup_finished.is_set()
    assert app.state.admission.active == 1


async def finish_task(task, stream):
    stream.allow_cleanup.set()
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_disconnect_overlapping_deadline_finishes_cleanup_before_releasing_slot():
    stream = SlowCleanupStream()
    disconnected = asyncio.Event()
    calls = 0

    def handler(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, headers={"Content-Type": "application/json"}, stream=stream)
        return httpx.Response(200, json=completion())

    settings = make_settings(max_in_flight=1, request_timeout_seconds=0.04)
    async with gateway(handler, settings=settings) as (client, app):
        received = False
        sent = []

        async def receive():
            nonlocal received
            if not received:
                received = True
                return {
                    "type": "http.request",
                    "body": json.dumps(QUESTION).encode(),
                    "more_body": False,
                }
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        task = asyncio.create_task(app(request_scope(), receive, send))
        try:
            await asyncio.wait_for(stream.body_started.wait(), timeout=1)
            disconnected.set()
            await assert_slot_held_past_deadline(task, app, stream)
            full = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
            assert full.status_code == 503
            assert full.json()["error"]["code"] == "gateway_overloaded"
            assert calls == 1

            stream.allow_cleanup.set()
            await asyncio.wait_for(task, timeout=1)
            assert stream.cleanup_finished.is_set()
            assert app.state.admission.active == 0
            assert sent[0]["status"] == 499
            assert sent[-1].get("body", b"") == b""
            recovered = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
            assert recovered.status_code == 200
            assert calls == 2
        finally:
            await finish_task(task, stream)


async def test_caller_cancellation_overlapping_deadline_finishes_cleanup_before_release():
    stream = SlowCleanupStream()
    settings = make_settings(max_in_flight=1, request_timeout_seconds=0.04)
    async with gateway(
        lambda _: httpx.Response(200, headers={"Content-Type": "application/json"}, stream=stream),
        settings=settings,
    ) as (client, app):
        task = asyncio.create_task(client.post("/v1/chat/completions", json=QUESTION, headers=AUTH))
        try:
            await asyncio.wait_for(stream.body_started.wait(), timeout=1)
            task.cancel()
            await assert_slot_held_past_deadline(task, app, stream)
            stream.allow_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            assert stream.cleanup_finished.is_set()
            assert app.state.admission.active == 0
        finally:
            await finish_task(task, stream)


async def test_successful_body_with_late_cleanup_returns_timeout_after_releasing_slot():
    stream = SlowCleanupStream(json.dumps(completion()).encode())
    calls = 0

    def handler(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, headers={"Content-Type": "application/json"}, stream=stream)
        return httpx.Response(200, json=completion())

    settings = make_settings(max_in_flight=1, request_timeout_seconds=0.04)
    async with gateway(handler, settings=settings) as (client, app):
        task = asyncio.create_task(client.post("/v1/chat/completions", json=QUESTION, headers=AUTH))
        try:
            await assert_slot_held_past_deadline(task, app, stream)
            full = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
            assert full.status_code == 503
            assert calls == 1
            stream.allow_cleanup.set()
            result = await asyncio.wait_for(task, timeout=1)
            assert result.status_code == 504
            assert result.json()["error"]["code"] == "backend_timeout"
            assert stream.cleanup_finished.is_set()
            assert app.state.admission.active == 0
            recovered = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
            assert recovered.status_code == 200
            assert calls == 2
        finally:
            await finish_task(task, stream)
