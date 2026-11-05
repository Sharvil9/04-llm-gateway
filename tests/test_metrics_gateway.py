"""Operational measurements at real middleware boundaries, without model loads."""

import asyncio
import hashlib
import json
from dataclasses import replace

import httpx
import pytest
from test_cache_gateway import (
    REQUEST,
    BackendProbe,
    cache_settings,
    chat,
    request_receive,
    request_scope,
)
from test_gateway import AUTH, CLIENT_KEY, QUESTION, completion, gateway, make_settings

from llm_gateway.app import create_app
from llm_gateway.config import load_settings

OPERATOR_KEY = "operator-metrics-key-" + "m" * 32
OPERATOR_AUTH = {"Authorization": f"Bearer {OPERATOR_KEY}"}


def operator_settings(settings=None):
    return replace(
        settings or make_settings(), metrics_key_hash=hashlib.sha256(OPERATOR_KEY.encode()).digest()
    )


async def snapshot(client):
    response = await client.get("/v1/metrics", headers=OPERATOR_AUTH)
    assert response.status_code == 200
    return response.json()


async def test_metrics_default_disabled_and_operator_auth_is_not_a_client_credential():
    async with gateway() as (client, _):
        assert (await client.get("/v1/metrics", headers=AUTH)).status_code == 404
    async with gateway(settings=operator_settings()) as (client, _):
        assert (await client.get("/v1/metrics")).status_code == 401
        assert (await client.get("/v1/metrics", headers=AUTH)).status_code == 401
        duplicate = [*OPERATOR_AUTH.items(), *OPERATOR_AUTH.items()]
        assert (await client.get("/v1/metrics", headers=duplicate)).status_code == 401
        assert (await client.get("/v1/models", headers=OPERATOR_AUTH)).status_code == 401
        assert (await snapshot(client))["counts"]["requests_started"] == 0


async def test_scrapes_do_not_spend_completion_quota_or_count_themselves():
    settings = operator_settings(
        make_settings(rate_limit={"burst": 1, "requests_per_second": 0.01})
    )
    async with gateway(settings=settings) as (client, _):
        before = await snapshot(client)
        for _ in range(3):
            assert await snapshot(client) == before
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/v1/models", headers=AUTH)).status_code == 200
        assert (
            await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        ).status_code == 200
        assert (
            await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        ).status_code == 429
        result = await snapshot(client)
        assert result["counts"]["responses_completed"] == 2
        assert result["counts"]["successful_responses"] == 1
        assert result["status"]["exact"]["429"] == 1
        assert result["counts"]["upstream_attempts"] == 1
        assert result["gauges"]["requests_in_flight"] == 0


async def test_exact_hits_preserve_stage_provenance_without_double_counting_usage():
    backend = BackendProbe()
    async with gateway(backend, settings=operator_settings(cache_settings())) as (client, _):
        first, second = await chat(client), await chat(client)
        assert first.content == second.content
        result = await snapshot(client)
        assert result["cache"]["completed_dispositions"]["MISS"] == 1
        assert result["cache"]["completed_dispositions"]["HIT"] == 1
        assert result["counts"]["upstream_attempts"] == 1
        assert result["usage"]["responses_with_reported_usage"] == 1
        assert result["usage"]["reported_total_tokens"] == 5
        assert result["latency_ms"]["adapter"]["count"] == 1
        assert result["latency_ms"]["exact_lookup"]["count"] == 2
        assert result["latency_ms"]["semantic"]["count"] == 0
        for response in (first, second):
            fields = dict(
                (part.split(";", 1)[0].strip(), float(part.split("dur=", 1)[1]))
                for part in response.headers["Server-Timing"].split(",")
            )
            assert fields["gateway"] >= fields["exact"]
            assert fields["residual"] >= 0
        assert "adapter;" not in second.headers["Server-Timing"]
        serialized = json.dumps(result)
        for sensitive in (
            CLIENT_KEY,
            OPERATOR_KEY,
            REQUEST["messages"][0]["content"],
            "backend.invalid",
        ):
            assert sensitive not in serialized


async def test_concurrent_gauges_and_overload_are_accurate_and_recover():
    entered, release = asyncio.Event(), asyncio.Event()

    async def backend(_):
        entered.set()
        await release.wait()
        return httpx.Response(200, json=completion())

    # This test deliberately holds inference while exercising metric snapshots
    # and overload. Give that coordination room on slow/contended hosts;
    # separate deadline tests verify timeout classification and cleanup.
    settings = operator_settings(make_settings(max_in_flight=1, request_timeout_seconds=30.0))
    async with gateway(backend, settings=settings) as (client, _):
        task = asyncio.create_task(client.post("/v1/chat/completions", json=QUESTION, headers=AUTH))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            held = await snapshot(client)
            assert held["gauges"]["requests_in_flight"] == 1
            assert held["gauges"]["inference_in_flight"] == 1
            results = await asyncio.gather(
                *(
                    client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
                    for _ in range(20)
                )
            )
            assert all(response.status_code == 503 for response in results)
            busy = await snapshot(client)
            assert busy["status"]["exact"]["503"] == 20
            assert busy["counts"]["upstream_attempts"] == 1
            release.set()
            assert (await task).status_code == 200
            recovered = await snapshot(client)
            assert recovered["gauges"]["requests_in_flight"] == 0
            assert recovered["gauges"]["inference_in_flight"] == 0
            assert recovered["counts"]["requests_started"] == 21
            assert recovered["counts"]["responses_completed"] == 21
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


async def test_timeout_and_auth_errors_count_without_logging_error_bodies():
    async def backend(_):
        await asyncio.sleep(1)

    settings = operator_settings(make_settings(request_timeout_seconds=0.01))
    async with gateway(backend, settings=settings) as (client, _):
        assert (await client.post("/v1/chat/completions", json=QUESTION)).status_code == 401
        assert (
            await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        ).status_code == 504
        result = await snapshot(client)
        assert result["status"]["by_family"]["4xx"] == 1
        assert result["status"]["exact"]["504"] == 1
        assert result["counts"]["http_error_responses"] == 2
        assert result["gauges"]["requests_in_flight"] == 0
        assert result["latency_ms"]["adapter"]["count"] == 1


async def test_metrics_usage_bounds_cannot_turn_a_supported_response_into_an_error():
    usage = {"prompt_tokens": 2**64, "completion_tokens": 1, "total_tokens": 2**64 + 1}
    backend = BackendProbe()
    backend.result = completion(usage=usage)
    async with gateway(backend, settings=operator_settings(cache_settings())) as (client, _):
        first, hit = await chat(client), await chat(client)
        assert first.status_code == hit.status_code == 200
        assert first.content == hit.content
        assert first.json()["usage"] == usage
        result = await snapshot(client)
        assert result["usage"]["ignored_usage_reports"] == 1
        assert result["usage"]["responses_with_reported_usage"] == 0


@pytest.mark.parametrize("failure", [OSError, asyncio.CancelledError])
async def test_failed_final_send_is_aborted_not_success_and_cannot_store(failure):
    app = create_app(
        operator_settings(cache_settings()), transport=httpx.MockTransport(BackendProbe())
    )
    async with app.router.lifespan_context(app):

        async def send(message):
            if message["type"] == "http.response.body":
                raise failure()

        with pytest.raises(failure):
            await app(request_scope(), request_receive(asyncio.Event()), send)
        result = app.state.metrics.snapshot()
        assert result["counts"]["successful_responses"] == 0
        assert result["counts"]["responses_completed"] == 0
        assert result["counts"]["aborted_requests"] == 1
        assert result["status"]["by_family"]["2xx"] == 1
        assert result["gauges"]["requests_in_flight"] == 0
        assert app.state.response_cache.snapshot()["entries"] == 0


async def test_upload_disconnect_and_backend_cancellation_finish_metrics_once():
    settings = operator_settings()
    app = create_app(settings)
    async with app.router.lifespan_context(app):

        async def receive():
            return {"type": "http.disconnect"}

        async def send(_):
            raise AssertionError("An upload-disconnected peer receives no response")

        await app(request_scope(), receive, send)
        result = app.state.metrics.snapshot()
        assert result["outcomes"]["disconnected"] == 1
        assert result["gauges"]["requests_in_flight"] == 0
        assert result["counts"]["requests_finished"] == 1

    entered = asyncio.Event()

    async def backend(_):
        entered.set()
        await asyncio.Event().wait()

    async with gateway(backend, settings=settings) as (client, app):
        task = asyncio.create_task(client.post("/v1/chat/completions", json=QUESTION, headers=AUTH))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = app.state.metrics.snapshot()
        assert result["outcomes"]["cancelled"] == 1
        assert result["counts"]["aborted_requests"] == 1
        assert result["gauges"]["requests_in_flight"] == 0
        assert app.state.admission.active == 0


async def test_optional_json_logging_contains_only_operational_fields(capsys):
    async with gateway(settings=operator_settings(make_settings(log_requests=True))) as (client, _):
        response = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert response.status_code == 200
    captured = capsys.readouterr().err
    record = json.loads(captured)
    assert set(record) == {"event", "status", "outcome", "cache", "duration_ms"}
    assert record["status"] == 200 and record["outcome"] == "completed"
    for sensitive in (CLIENT_KEY, OPERATOR_KEY, "Hello", "backend.invalid"):
        assert sensitive not in captured


@pytest.mark.parametrize("metrics_key", ["short", CLIENT_KEY])
def test_metrics_environment_secret_is_validated_and_must_be_distinct(monkeypatch, metrics_key):
    monkeypatch.setenv("GATEWAY_CONFIG", "config/mock.toml")
    monkeypatch.setenv("GATEWAY_CLIENT_KEYS_JSON", json.dumps({"client-a": CLIENT_KEY}))
    monkeypatch.setenv("GATEWAY_METRICS_KEY", metrics_key)
    with pytest.raises(RuntimeError) as error:
        load_settings()
    assert metrics_key not in str(error.value)
