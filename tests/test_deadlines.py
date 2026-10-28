"""Deadline scope, cleanup, and late-success regressions without model inference."""

import asyncio
import time

import httpx
from test_gateway import AUTH, QUESTION, completion, gateway, make_settings


def timings(response):
    return {
        name: float(duration)
        for item in response.headers["server-timing"].split(", ")
        for name, duration in [item.split(";dur=")]
    }


async def test_total_deadline_interrupts_a_trickling_response_and_releases_capacity():
    chunks = 0
    closed = False

    class Trickle(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal chunks
            while True:
                await asyncio.sleep(0.005)
                chunks += 1
                yield b" "

        async def aclose(self):
            nonlocal closed
            closed = True

    settings = make_settings(request_timeout_seconds=0.06, max_in_flight=1)
    async with gateway(
        lambda _: httpx.Response(
            200, headers={"Content-Type": "application/json"}, stream=Trickle()
        ),
        settings=settings,
    ) as (client, app):
        response = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert response.status_code == 504
        assert chunks >= 2 and closed
        assert app.state.admission.active == 0
        measured = timings(response)
        assert measured["gateway"] >= measured["adapter"] >= 50


async def test_slow_cleanup_is_in_adapter_timing_and_completes_before_slot_release():
    closed = False

    class SlowClose(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self):
            nonlocal closed
            await asyncio.sleep(0.06)
            closed = True

    settings = make_settings(request_timeout_seconds=0.02, max_in_flight=1)
    async with gateway(
        lambda _: httpx.Response(
            200, headers={"Content-Type": "application/json"}, stream=SlowClose()
        ),
        settings=settings,
    ) as (client, app):
        response = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert response.status_code == 504 and closed
        assert app.state.admission.active == 0
        measured = timings(response)
        assert measured["gateway"] >= measured["adapter"] >= 75


async def test_synchronous_deadline_overrun_cannot_return_late_success():
    def blocking_transport(_):
        # Deliberate test-only blocking: asyncio's timer cannot run during this.
        time.sleep(0.03)
        return httpx.Response(200, json=completion())

    async with gateway(
        blocking_transport,
        settings=make_settings(request_timeout_seconds=0.005, max_in_flight=1),
    ) as (client, app):
        response = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert response.status_code == 504
        assert timings(response)["adapter"] >= 25
        assert app.state.admission.active == 0


async def test_locally_rejected_requests_have_no_adapter_timing():
    async with gateway() as (client, _):
        unauthorized = await client.post("/v1/chat/completions", json=QUESTION)
        unknown = await client.post(
            "/v1/chat/completions", json=QUESTION | {"model": "absent"}, headers=AUTH
        )
        assert unauthorized.status_code == 401 and unknown.status_code == 404
        assert set(timings(unauthorized)) == set(timings(unknown)) == {"gateway"}
