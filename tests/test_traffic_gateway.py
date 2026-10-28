"""HTTP-boundary tests for process-local request quotas and bounded admission."""

import asyncio
import hashlib
import json
from dataclasses import replace

import httpx
import pytest
from test_gateway import AUTH, QUESTION, completion, gateway, make_settings


def traffic_settings(*, burst=100, requests_per_second=0.01, **overrides):
    return make_settings(
        rate_limit={"burst": burst, "requests_per_second": requests_per_second}, **overrides
    )


async def cleanup(tasks, release):
    release.set()
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_simultaneous_requests_cannot_exceed_client_burst():
    burst, total = 7, 40
    release = asyncio.Event()
    admitted = asyncio.Event()
    finished = asyncio.Queue()
    calls = 0

    async def handler(_):
        nonlocal calls
        calls += 1
        if calls == burst:
            admitted.set()
        await release.wait()
        return httpx.Response(200, json=completion())

    async with gateway(handler, settings=traffic_settings(burst=burst)) as (client, _):

        async def request():
            result = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
            await finished.put(result)
            return result

        tasks = [asyncio.create_task(request()) for _ in range(total)]
        try:
            await asyncio.wait_for(admitted.wait(), timeout=2)
            rejected = [
                await asyncio.wait_for(finished.get(), timeout=2) for _ in range(total - burst)
            ]
            assert calls == burst
            assert all(response.status_code == 429 for response in rejected)
            assert all(
                response.json()["error"]["code"] == "rate_limit_exceeded" for response in rejected
            )
            assert all(int(response.headers["Retry-After"]) >= 1 for response in rejected)
            release.set()
            results = await asyncio.gather(*tasks)
            assert sum(response.status_code == 200 for response in results) == burst
        finally:
            await cleanup(tasks, release)


async def test_client_quotas_are_isolated_and_overrides_apply():
    second_key = "second-client-key-" + "b" * 32
    settings = traffic_settings(
        burst=1,
        client_rate_limits={"client-b": {"burst": 2, "requests_per_second": 0.01}},
    )
    settings = replace(
        settings,
        client_key_hashes=settings.client_key_hashes
        | {hashlib.sha256(second_key.encode()).digest(): "client-b"},
    )
    second_auth = {"Authorization": f"Bearer {second_key}"}
    async with gateway(settings=settings) as (client, _):
        first, second = await asyncio.gather(
            asyncio.gather(
                *[
                    client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
                    for _ in range(3)
                ]
            ),
            asyncio.gather(
                *[
                    client.post("/v1/chat/completions", json=QUESTION, headers=second_auth)
                    for _ in range(3)
                ]
            ),
        )
        assert sorted(response.status_code for response in first) == [200, 429, 429]
        assert sorted(response.status_code for response in second) == [200, 200, 429]


async def test_unauthorized_requests_liveness_and_model_listing_do_not_spend_quota():
    async with gateway(settings=traffic_settings(burst=1)) as (client, _):
        for _ in range(3):
            assert (await client.post("/v1/chat/completions", json=QUESTION)).status_code == 401
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/v1/models", headers=AUTH)).status_code == 200
        assert (
            await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        ).status_code == 200
        assert (
            await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        ).status_code == 429


@pytest.mark.parametrize(
    ("request_options", "first_status"),
    [
        ({"json": QUESTION | {"stream": True}}, 422),
        ({"content": b"malformed", "headers": AUTH | {"Content-Type": "text/plain"}}, 415),
        ({"json": QUESTION | {"model": "absent"}}, 404),
        ({"content": b" " * 65_537, "headers": AUTH | {"Content-Type": "application/json"}}, 413),
    ],
)
async def test_authenticated_rejected_attempts_still_spend_quota(request_options, first_status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion())

    async with gateway(handler, settings=traffic_settings(burst=1)) as (client, _):
        first = await client.post("/v1/chat/completions", **({"headers": AUTH} | request_options))
        second = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert first.status_code == first_status
        assert second.status_code == 429
        assert calls == []


async def test_full_admission_rejects_immediately_without_upstream_call_or_retry_hint():
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def handler(_):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return httpx.Response(200, json=completion())

    async with gateway(handler, settings=traffic_settings(max_in_flight=1)) as (client, app):
        first = asyncio.create_task(
            client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            full = await asyncio.wait_for(
                client.post("/v1/chat/completions", json=QUESTION, headers=AUTH), timeout=1
            )
            assert full.status_code == 503
            assert full.json()["error"]["code"] == "gateway_overloaded"
            assert "retry-after" not in full.headers
            unknown = await client.post(
                "/v1/chat/completions", json=QUESTION | {"model": "absent"}, headers=AUTH
            )
            assert unknown.status_code == 404
            assert calls == 1
            assert app.state.admission.active == 1
            release.set()
            assert (await first).status_code == 200
            assert app.state.admission.active == 0
        finally:
            await cleanup([first], release)


async def test_global_overload_attempt_consumes_client_request_quota():
    started, release = asyncio.Event(), asyncio.Event()

    async def handler(_):
        started.set()
        await release.wait()
        return httpx.Response(200, json=completion())

    async with gateway(handler, settings=traffic_settings(burst=2, max_in_flight=1)) as (client, _):
        first = asyncio.create_task(
            client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            assert (
                await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
            ).status_code == 503
            assert (
                await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
            ).status_code == 429
            release.set()
            assert (await first).status_code == 200
            assert (
                await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
            ).status_code == 429
        finally:
            await cleanup([first], release)


@pytest.mark.parametrize("first_outcome", ["success", "backend_error", "timeout"])
async def test_admission_recovers_after_success_backend_error_and_timeout(first_outcome):
    calls = 0

    async def handler(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            if first_outcome == "backend_error":
                return httpx.Response(503)
            if first_outcome == "timeout":
                await asyncio.Event().wait()
        return httpx.Response(200, json=completion())

    settings = traffic_settings(max_in_flight=1, request_timeout_seconds=0.02)
    async with gateway(handler, settings=settings) as (client, app):
        first = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert (
            first.status_code
            == {"success": 200, "backend_error": 503, "timeout": 504}[first_outcome]
        )
        assert app.state.admission.active == 0
        second = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert second.status_code == 200
        assert app.state.admission.active == 0
        assert calls == 2


async def test_cancelled_request_closes_response_and_releases_admission():
    started, closed = asyncio.Event(), asyncio.Event()
    calls = 0

    class WaitingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self):
            closed.set()

    def handler(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200, headers={"Content-Type": "application/json"}, stream=WaitingStream()
            )
        return httpx.Response(200, json=completion())

    async with gateway(handler, settings=traffic_settings(max_in_flight=1)) as (client, app):
        first = asyncio.create_task(
            client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert closed.is_set()
            assert app.state.admission.active == 0
            assert (
                await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
            ).status_code == 200
            assert app.state.admission.active == 0
        finally:
            await cleanup([first], asyncio.Event())


async def test_saturation_stays_bounded_and_excess_requests_do_not_queue():
    capacity = 2
    full, release = asyncio.Event(), asyncio.Event()
    calls, active, peak = 0, 0, 0

    async def handler(_):
        nonlocal calls, active, peak
        calls += 1
        active += 1
        peak = max(peak, active)
        if active == capacity:
            full.set()
        try:
            await release.wait()
            return httpx.Response(200, json=completion())
        finally:
            active -= 1

    async with gateway(handler, settings=traffic_settings(max_in_flight=capacity)) as (client, app):
        admitted = [
            asyncio.create_task(client.post("/v1/chat/completions", json=QUESTION, headers=AUTH))
            for _ in range(capacity)
        ]
        try:
            await asyncio.wait_for(full.wait(), timeout=1)
            excess = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
                        for _ in range(64)
                    ]
                ),
                timeout=2,
            )
            assert all(response.status_code == 503 for response in excess)
            assert all(
                response.json()["error"]["code"] == "gateway_overloaded" for response in excess
            )
            assert calls == capacity and peak == capacity
            assert app.state.admission.active == capacity
            release.set()
            assert all(result.status_code == 200 for result in await asyncio.gather(*admitted))
            assert app.state.admission.active == active == 0
            assert calls == capacity
            assert (
                await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
            ).status_code == 200
            assert calls == capacity + 1
        finally:
            await cleanup(admitted, release)


async def test_socket_disconnect_cancels_upstream_and_releases_admission():
    started, disconnected, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class WaitingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self):
            closed.set()

    async with gateway(
        lambda _: httpx.Response(
            200, headers={"Content-Type": "application/json"}, stream=WaitingStream()
        ),
        settings=traffic_settings(max_in_flight=1),
    ) as (_, app):
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

        scope = {
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
        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            assert app.state.admission.active == 1
            disconnected.set()
            await asyncio.wait_for(task, timeout=1)
            assert closed.is_set()
            assert app.state.admission.active == 0
            assert sent[0]["type"] == "http.response.start"
            assert sent[0]["status"] == 499
            assert sent[-1].get("body", b"") == b""
        finally:
            await cleanup([task], disconnected)
