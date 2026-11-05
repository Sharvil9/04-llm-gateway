"""Actual HTTP recovery evidence, separate from synthetic ASGI disconnect tests."""

import asyncio

import httpx
import pytest

from devtools import load_reliability as probe


async def test_actual_socket_disconnect_releases_permits_and_accepts_replacement():
    async with httpx.AsyncClient(
        trust_env=False, timeout=8.0, limits=httpx.Limits(max_connections=64)
    ) as client:
        result = await probe.cancellation_phase(client, cycles=1)
    row = result["rows"][0]
    assert row["overload_503"] == 20
    assert row["cancelled_client_requests"] == 3
    assert row["after_one_disconnect"] == 1
    assert row["after_all_disconnects"] == 0
    assert row["actual_upstream_calls"] == 4
    assert row["recovery_status"] == 200
    # An uncooperative remote application may outlive the gateway operation.
    # This must be explicit rather than confused with leaked gateway permits.
    assert row["uncooperative_backend_jobs_before_release"] == 3
    assert result["gateway_capacity"] == 2
    assert row["final_metrics"]["outcomes"]["disconnected"] == 3
    assert row["final_metrics"]["gauges"]["inference_in_flight"] == 0


async def test_actual_http_timeout_recovers_without_retained_gateway_capacity():
    async with httpx.AsyncClient(trust_env=False, timeout=8.0) as client:
        result = await probe.timeout_phase(client, cycles=1)
    row = result["rows"][0]
    assert row["status"] == 504
    assert row["gateway_inference_after_timeout"] == 0
    assert row["uncooperative_backend_jobs_after_timeout"] == 1
    assert row["actual_upstream_calls"] == 2
    assert row["recovery_status"] == 200
    assert row["final_metrics"]["status"]["exact"]["504"] == 1
    assert row["final_metrics"]["counts"]["successful_responses"] == 1


@pytest.mark.parametrize("cycles", [0, 11])
async def test_cycle_bounds_fail_before_launching_any_servers(cycles):
    with pytest.raises(ValueError, match="cycles"):
        await probe.cancellation_phase(None, cycles=cycles)
    with pytest.raises(ValueError, match="cycles"):
        await probe.timeout_phase(None, cycles=cycles)


async def test_polling_propagates_http_errors_instead_of_claiming_idle():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
        base_url="http://fixture.invalid",
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await probe.wait_json(client, "/stats", lambda _: True)


async def test_cancel_requests_awaits_cleanup_before_returning():
    started, finished = asyncio.Event(), asyncio.Event()

    async def held():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            finished.set()

    task = asyncio.create_task(held())
    await started.wait()
    await probe.cancel_requests([task])
    assert task.cancelled()
    assert finished.is_set()
