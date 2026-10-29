"""Exact-cache safety and HTTP lifecycle integration tests; no semantic reuse."""

import asyncio
import hashlib
import json
from dataclasses import replace

import httpx
import pytest
from test_gateway import AUTH, BACKEND_KEY, completion, gateway, make_settings

PROMPT = "Explain the difference between a Python list and a tuple."
OTHER_PROMPT = "What does a Python dictionary store?"
REQUEST = {
    "model": "demo",
    "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 32,
    "seed": 7,
}
CACHE_AUTH = AUTH | {"X-Gateway-Cache": "exact"}


def cache_settings(**overrides):
    values = {
        "backends": {
            "local": {
                "base_url": "http://backend.invalid/v1",
                "api_key_env": "BACKEND_KEY",
                "revision": "backend-r1",
            }
        },
        "models": {
            "demo": {"backend": "local", "model": "physical-model", "revision": "weights-r1"}
        },
        "exact_cache": {
            "enabled": True,
            "ttl_seconds": 60.0,
            "max_entries": 128,
            "max_bytes": 8_388_608,
            "namespace": "public-static-v1",
            "approved_prompts": [PROMPT, OTHER_PROMPT],
        },
        "rate_limit": {"burst": 200, "requests_per_second": 0.01},
    }
    values.update(overrides)
    return make_settings(**values)


class BackendProbe:
    def __init__(self):
        self.calls = []
        self.status = 200
        self.result = completion()
        self.headers = {}

    def __call__(self, request):
        self.calls.append(request)
        return httpx.Response(self.status, json=self.result, headers=self.headers)


async def chat(client, payload=None, headers=None):
    return await client.post(
        "/v1/chat/completions",
        json=REQUEST if payload is None else payload,
        headers=CACHE_AUTH if headers is None else headers,
    )


def assert_empty(app):
    assert app.state.response_cache.snapshot()["entries"] == 0
    assert app.state.response_cache.snapshot()["bytes_used"] == 0
    assert app.state.response_cache.stats.stores == 0


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
        "headers": [(name.lower().encode(), value.encode()) for name, value in CACHE_AUTH.items()]
        + [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }


def request_receive(disconnected):
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {
                "type": "http.request",
                "body": json.dumps(REQUEST).encode(),
                "more_body": False,
            }
        await disconnected.wait()
        return {"type": "http.disconnect"}

    return receive


async def test_exact_hit_preserves_complete_wire_body_and_reported_usage():
    backend = BackendProbe()
    async with gateway(backend, settings=cache_settings()) as (client, app):
        first, second = await chat(client), await chat(client)
        assert first.status_code == second.status_code == 200
        assert first.headers["X-Gateway-Cache"] == "MISS"
        assert second.headers["X-Gateway-Cache"] == "HIT"
        assert first.content == second.content
        assert first.json() == completion(model="demo")
        assert first.headers["Cache-Control"] == second.headers["Cache-Control"] == "no-store"
        assert len(backend.calls) == 1
        assert backend.calls[0].headers["Authorization"] == f"Bearer {BACKEND_KEY}"
        assert "X-Gateway-Cache" not in backend.calls[0].headers
        stats = app.state.response_cache.stats
        assert (stats.hits, stats.misses, stats.bypasses, stats.stores, stats.upstream_calls) == (
            1,
            1,
            0,
            1,
            1,
        )
        assert app.state.response_cache.snapshot()["entries"] == 1
        assert app.state.response_cache.snapshot()["bytes_used"] >= len(first.content)


@pytest.mark.parametrize("disabled", [False, True])
async def test_cache_is_bypassed_without_opt_in_or_when_disabled(disabled):
    backend = BackendProbe()
    settings = cache_settings()
    settings.config.exact_cache.enabled = not disabled
    headers = CACHE_AUTH if disabled else AUTH
    async with gateway(backend, settings=settings) as (client, app):
        responses = [await chat(client, headers=headers) for _ in range(2)]
        assert all(response.status_code == 200 for response in responses)
        assert all(response.headers["X-Gateway-Cache"] == "BYPASS" for response in responses)
        assert len(backend.calls) == 2
        assert app.state.response_cache.stats.bypasses == 2
        assert_empty(app)


async def test_duplicate_cache_opt_in_headers_are_conservatively_bypassed():
    backend = BackendProbe()
    headers = [
        ("Authorization", AUTH["Authorization"]),
        ("X-Gateway-Cache", "exact"),
        ("X-Gateway-Cache", "exact"),
    ]
    async with gateway(backend, settings=cache_settings()) as (client, app):
        assert (await chat(client, headers=headers)).headers["X-Gateway-Cache"] == "BYPASS"
        assert_empty(app)


async def test_cache_entries_are_isolated_by_authenticated_client():
    backend = BackendProbe()
    second_key = "another-cache-client-" + "b" * 32
    settings = cache_settings()
    settings = replace(
        settings,
        client_key_hashes=settings.client_key_hashes
        | {hashlib.sha256(second_key.encode()).digest(): "client-b"},
    )
    second_auth = {"Authorization": f"Bearer {second_key}", "X-Gateway-Cache": "exact"}
    async with gateway(backend, settings=settings) as (client, app):
        first_a = await chat(client)
        backend.result = completion(id="chatcmpl-second-client")
        first_b = await chat(client, headers=second_auth)
        second_a, second_b = await chat(client), await chat(client, headers=second_auth)
        assert [r.headers["X-Gateway-Cache"] for r in [first_a, first_b, second_a, second_b]] == [
            "MISS",
            "MISS",
            "HIT",
            "HIT",
        ]
        assert second_a.content == first_a.content
        assert second_b.content == first_b.content
        assert second_a.content != second_b.content
        assert len(backend.calls) == 2
        assert app.state.response_cache.snapshot()["entries"] == 2


async def test_generation_settings_have_distinct_exact_keys():
    backend = BackendProbe()
    async with gateway(backend, settings=cache_settings()) as (client, _):
        original = await chat(client)
        for changes in [{"max_tokens": 33}, {"seed": 8}]:
            miss = await chat(client, REQUEST | changes)
            hit = await chat(client, REQUEST | changes)
            assert miss.headers["X-Gateway-Cache"] == "MISS"
            assert hit.headers["X-Gateway-Cache"] == "HIT"
            assert hit.content == miss.content
        assert (await chat(client)).content == original.content
        assert len(backend.calls) == 3


async def test_oversized_unicode_wire_body_does_not_evict_existing_cached_response():
    settings = cache_settings()
    settings.config.exact_cache.max_bytes = 1024
    backend = BackendProbe()
    async with gateway(backend, settings=settings) as (client, app):
        original = await chat(client)
        backend.result = completion(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "漢" * 300},
                    "finish_reason": "stop",
                }
            ]
        )
        large_request = REQUEST | {"messages": [{"role": "user", "content": OTHER_PROMPT}]}
        for _ in range(2):
            response = await chat(client, large_request)
            assert response.status_code == 200
            assert response.headers["X-Gateway-Cache"] == "MISS"
            assert len(response.content) > 1024
        hit = await chat(client)
        assert hit.headers["X-Gateway-Cache"] == "HIT" and hit.content == original.content
        snapshot = app.state.response_cache.snapshot()
        assert snapshot["entries"] == 1 and snapshot["bytes_used"] <= 1024
        assert snapshot["stores"] == 1 and snapshot["evictions"] == 0
        assert len(backend.calls) == snapshot["upstream_calls"] == 3


async def test_sampling_unapproved_prompts_and_context_are_bypassed():
    backend = BackendProbe()
    async with gateway(backend, settings=cache_settings()) as (client, app):
        original = await chat(client)
        changes = [
            {"temperature": 0.2},
            {"top_p": 0.9},
            {"seed": None},
            {"messages": [{"role": "user", "content": "My private session question"}]},
            {
                "messages": [
                    {"role": "system", "content": "Use confidential session state"},
                    *REQUEST["messages"],
                ]
            },
            {
                "messages": [
                    *REQUEST["messages"],
                    {"role": "assistant", "content": "Prior state"},
                    *REQUEST["messages"],
                ]
            },
        ]
        for change in changes:
            result = await chat(client, REQUEST | change)
            assert result.status_code == 200
            assert result.headers["X-Gateway-Cache"] == "BYPASS"
        assert (await chat(client)).content == original.content
        assert len(backend.calls) == len(changes) + 1
        assert app.state.response_cache.stats.stores == 1


async def test_model_backend_and_policy_revisions_partition_entries():
    backend = BackendProbe()
    async with gateway(backend, settings=cache_settings()) as (client, app):
        original = await chat(client)
        config = app.state.settings.config
        for target, field, changed in [
            (config.models["demo"], "revision", "weights-r2"),
            (config.backends["local"], "revision", "backend-r2"),
            (config.backends["local"], "base_url", "http://other-backend.invalid/v1"),
            (config.exact_cache, "namespace", "public-static-v2"),
        ]:
            before = getattr(target, field)
            setattr(target, field, changed)
            assert (await chat(client)).headers["X-Gateway-Cache"] == "MISS"
            setattr(target, field, before)
            restored = await chat(client)
            assert restored.headers["X-Gateway-Cache"] == "HIT"
            assert restored.content == original.content
        assert len(backend.calls) == 5


@pytest.mark.parametrize("missing_pin", ["model", "backend"])
async def test_missing_revision_pin_disables_exact_cache(missing_pin):
    settings, backend = cache_settings(), BackendProbe()
    if missing_pin == "model":
        settings.config.models["demo"].revision = None
    else:
        settings.config.backends["local"].revision = None
    async with gateway(backend, settings=settings) as (client, app):
        assert (await chat(client)).headers["X-Gateway-Cache"] == "BYPASS"
        assert (await chat(client)).headers["X-Gateway-Cache"] == "BYPASS"
        assert len(backend.calls) == 2
        assert_empty(app)


async def test_expiration_uses_ttl_without_real_time_sleep():
    backend, now = BackendProbe(), [1000.0]
    async with gateway(backend, settings=cache_settings()) as (client, app):
        app.state.response_cache._clock = lambda: now[0]
        assert (await chat(client)).headers["X-Gateway-Cache"] == "MISS"
        now[0] += 59.0
        assert (await chat(client)).headers["X-Gateway-Cache"] == "HIT"
        now[0] += 1.0
        assert (await chat(client)).headers["X-Gateway-Cache"] == "MISS"
        assert len(backend.calls) == 2
        assert app.state.response_cache.stats.stores == 2


async def test_cache_hits_still_require_authentication_and_consume_request_quota():
    backend = BackendProbe()
    settings = cache_settings(rate_limit={"burst": 2, "requests_per_second": 0.000001})
    async with gateway(backend, settings=settings) as (client, _):
        assert (await chat(client)).headers["X-Gateway-Cache"] == "MISS"
        unauthorized = await chat(client, headers={"X-Gateway-Cache": "exact"})
        assert unauthorized.status_code == 401
        assert (await chat(client)).headers["X-Gateway-Cache"] == "HIT"
        rejected = await chat(client)
        assert rejected.status_code == 429
        assert int(rejected.headers["Retry-After"]) > 0
        assert len(backend.calls) == 1


async def test_hit_succeeds_while_inference_capacity_is_full():
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def handler(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            started.set()
            await release.wait()
        return httpx.Response(200, json=completion())

    async with gateway(handler, settings=cache_settings(max_in_flight=1)) as (client, app):
        primed = await chat(client)
        held = asyncio.create_task(
            chat(client, REQUEST | {"messages": [{"role": "user", "content": OTHER_PROMPT}]})
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            hit = await asyncio.wait_for(chat(client), timeout=1)
            assert hit.status_code == 200 and hit.headers["X-Gateway-Cache"] == "HIT"
            assert hit.content == primed.content
            assert calls == 2 and app.state.admission.active == 1
            cold = await chat(client, REQUEST | {"max_tokens": 34})
            assert cold.status_code == 503
            assert cold.json()["error"]["code"] == "gateway_overloaded"
            assert calls == 2
            release.set()
            assert (await held).status_code == 200
        finally:
            release.set()
            if not held.done():
                held.cancel()
            await asyncio.gather(held, return_exceptions=True)


async def test_simultaneous_hits_do_not_forward_or_acquire_inference_slots():
    backend = BackendProbe()
    async with gateway(backend, settings=cache_settings(max_in_flight=1)) as (client, app):
        original = await chat(client)
        responses = await asyncio.gather(*[chat(client) for _ in range(40)])
        assert all(response.status_code == 200 for response in responses)
        assert all(response.headers["X-Gateway-Cache"] == "HIT" for response in responses)
        assert all(response.content == original.content for response in responses)
        assert len(backend.calls) == 1
        assert app.state.response_cache.stats.hits == 40
        assert app.state.admission.active == 0


async def test_duplicate_cold_misses_are_not_coalesced_and_remain_capacity_bounded():
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def handler(_):
        nonlocal calls
        calls += 1
        number = calls
        if calls == 2:
            started.set()
        await release.wait()
        return httpx.Response(200, json=completion(id=f"chatcmpl-cold-{number}"))

    async with gateway(handler, settings=cache_settings(max_in_flight=2)) as (client, app):
        tasks = [asyncio.create_task(chat(client)) for _ in range(2)]
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            assert_empty(app)
            assert (await chat(client)).status_code == 503
            assert calls == 2 and app.state.admission.active == 2
            release.set()
            misses = await asyncio.gather(*tasks)
            assert all(response.status_code == 200 for response in misses)
            assert all(response.headers["X-Gateway-Cache"] == "MISS" for response in misses)
            hit = await chat(client)
            assert hit.headers["X-Gateway-Cache"] == "HIT"
            assert hit.content in [response.content for response in misses]
            assert calls == 2
            assert app.state.response_cache.stats.stores == 1
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize(
    "outcome", ["backend_error", "invalid_response", "truncation", "empty", "timeout"]
)
async def test_failed_incomplete_or_empty_completions_are_never_stored(outcome):
    calls = 0

    async def handler(_):
        nonlocal calls
        calls += 1
        if outcome == "timeout":
            await asyncio.Event().wait()
        if outcome == "backend_error":
            return httpx.Response(503)
        if outcome == "invalid_response":
            return httpx.Response(200, json={"invalid": "response"})
        result = completion()
        if outcome == "truncation":
            result["choices"][0]["finish_reason"] = "length"
        if outcome == "empty":
            result["choices"][0]["message"]["content"] = ""
        return httpx.Response(200, json=result)

    async with gateway(handler, settings=cache_settings(request_timeout_seconds=0.02)) as (
        client,
        app,
    ):
        responses = [await chat(client) for _ in range(2)]
        expected = {
            "backend_error": 503,
            "invalid_response": 502,
            "truncation": 200,
            "empty": 200,
            "timeout": 504,
        }
        assert all(response.status_code == expected[outcome] for response in responses)
        assert calls == 2
        assert_empty(app)


@pytest.mark.parametrize(
    "provider_headers",
    [
        {"Cache-Control": "private"},
        {"Cache-Control": "no-store"},
        {"Cache-Control": "no-cache"},
        {"Cache-Control": "public, max-age=3600"},
        {"Expires": "Wed, 31 Dec 2099 23:59:59 GMT"},
        {"Set-Cookie": "fixture-session=private"},
        {"Vary": "Authorization"},
    ],
)
async def test_provider_privacy_headers_prevent_cache_storage(provider_headers):
    backend = BackendProbe()
    backend.headers = provider_headers
    async with gateway(backend, settings=cache_settings()) as (client, app):
        assert (await chat(client)).status_code == 200
        assert (await chat(client)).headers["X-Gateway-Cache"] == "MISS"
        assert len(backend.calls) == 2
        assert_empty(app)


async def test_request_cache_control_and_cookies_bypass_existing_entries():
    backend = BackendProbe()
    async with gateway(backend, settings=cache_settings()) as (client, app):
        original = await chat(client)
        for extra in [
            {"Cache-Control": "no-store"},
            {"Cache-Control": "no-cache"},
            {"Pragma": "no-cache"},
            {"Cookie": "private-session=fixture"},
        ]:
            result = await chat(client, headers=CACHE_AUTH | extra)
            assert result.status_code == 200
            assert result.headers["X-Gateway-Cache"] == "BYPASS"
        assert (await chat(client)).content == original.content
        assert len(backend.calls) == 5
        assert app.state.response_cache.stats.stores == 1


async def test_shared_backend_client_rejects_cookies_between_authenticated_clients():
    backend = BackendProbe()
    backend.headers = {"Set-Cookie": "backend-session=private-alice; Path=/; HttpOnly"}
    second_key = "second-stateless-client-" + "b" * 32
    settings = cache_settings()
    settings = replace(
        settings,
        client_key_hashes=settings.client_key_hashes
        | {hashlib.sha256(second_key.encode()).digest(): "client-b"},
    )
    second_auth = {"Authorization": f"Bearer {second_key}", "X-Gateway-Cache": "exact"}
    async with gateway(backend, settings=settings) as (client, app):
        first = await chat(client)
        assert first.status_code == 200
        assert "Set-Cookie" not in first.headers
        assert len(app.state.http_client.cookies) == 0
        assert_empty(app)
        second = await chat(client, headers=second_auth)
        assert second.status_code == 200
        assert len(backend.calls) == 2
        assert "Cookie" not in backend.calls[1].headers
        assert len(app.state.http_client.cookies) == 0
        assert_empty(app)


@pytest.mark.parametrize("mutation_point", ["in_flight", "final_send"])
async def test_revision_changed_during_request_prevents_old_fingerprint_commit(mutation_point):
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def handler(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            if mutation_point == "in_flight":
                await release.wait()
        return httpx.Response(200, json=completion())

    async with gateway(handler, settings=cache_settings()) as (client, app):
        route = app.state.settings.config.models["demo"]
        original_revision = route.revision
        sent = []

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                route.revision = "weights-r2"

        async def invoke():
            if mutation_point == "in_flight":
                return await chat(client)
            await app(request_scope(), request_receive(asyncio.Event()), send)

        task = asyncio.create_task(invoke())
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            if mutation_point == "in_flight":
                route.revision = "weights-r2"
                release.set()
            result = await task
            if mutation_point == "in_flight":
                assert result.status_code == 200
            else:
                assert sent[0]["status"] == 200
            assert route.revision == "weights-r2"
            assert_empty(app)
            assert app.state.response_cache.stats.misses == 1
            assert app.state.response_cache.stats.hits == 0
            route.revision = original_revision
            assert (await chat(client)).headers["X-Gateway-Cache"] == "MISS"
            assert (await chat(client)).headers["X-Gateway-Cache"] == "HIT"
            assert calls == 2
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("failure", [OSError, asyncio.CancelledError])
async def test_final_response_send_failure_or_cancellation_prevents_cache_commit(failure):
    backend = BackendProbe()
    async with gateway(backend, settings=cache_settings()) as (client, app):

        async def send(message):
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                raise failure("Simulated response send failure")

        with pytest.raises(failure):
            await app(request_scope(), request_receive(asyncio.Event()), send)
        assert len(backend.calls) == 1
        assert_empty(app)
        assert (await chat(client)).headers["X-Gateway-Cache"] == "MISS"
        assert (await chat(client)).headers["X-Gateway-Cache"] == "HIT"
        assert len(backend.calls) == 2


async def test_observed_disconnect_racing_backend_completion_prevents_insertion():
    backend = BackendProbe()
    disconnected = asyncio.Event()
    disconnected.set()
    async with gateway(backend, settings=cache_settings()) as (client, app):
        scope = request_scope()

        async def send(_message):
            pass

        await app(scope, request_receive(disconnected), send)
        assert scope["state"]["client_disconnected"] is True
        assert_empty(app)
        assert (await chat(client)).headers["X-Gateway-Cache"] == "MISS"


@pytest.mark.parametrize("socket_disconnect", [False, True])
async def test_cancelled_or_disconnected_request_never_inserts_cache_entry(socket_disconnect):
    started, disconnected, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
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

    async with gateway(handler, settings=cache_settings()) as (client, app):
        sent = []

        async def send(message):
            sent.append(message)

        task = (
            asyncio.create_task(app(request_scope(), request_receive(disconnected), send))
            if socket_disconnect
            else asyncio.create_task(chat(client))
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            if socket_disconnect:
                disconnected.set()
                await asyncio.wait_for(task, timeout=1)
                assert sent[0]["status"] == 499
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert closed.is_set()
            assert app.state.admission.active == 0
            assert_empty(app)
            assert (await chat(client)).headers["X-Gateway-Cache"] == "MISS"
            assert calls == 2
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
