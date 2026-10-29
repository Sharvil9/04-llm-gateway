"""Guarded semantic-cache HTTP integration; synthetic embeddings, no model loading."""

import asyncio
import hashlib
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx
import pytest
from test_gateway import AUTH, completion, make_settings

from llm_gateway.app import create_app

np = pytest.importorskip("numpy", reason="semantic cache uses the optional NumPy dependency")

FRANCE = "What is the capital of France? Answer with only the city name."
PARAPHRASE = "Name France's capital city. Reply with the city name only."
THIRD = "Which city is the capital of France? Give only its name."
JAPAN = "What is the capital of Japan? Answer with only the city name."
JAPAN_PARAPHRASE = "Name Japan's capital city. Reply with the city name only."
SEMANTIC_AUTH = AUTH | {"X-Gateway-Cache": "semantic"}


def payload(prompt=FRANCE, **changes):
    return {
        "model": "demo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32,
        "seed": 7,
    } | changes


def semantic_settings(**changes):
    return make_settings(
        **(
            {
                "backends": {
                    "local": {"base_url": "http://backend.invalid/v1", "revision": "backend-r1"}
                },
                "models": {
                    "demo": {
                        "backend": "local",
                        "model": "physical-model",
                        "revision": "weights-r1",
                    },
                    "other": {"backend": "local", "model": "other-model", "revision": "weights-r2"},
                },
                "exact_cache": {
                    "enabled": True,
                    "ttl_seconds": 60.0,
                    "max_entries": 32,
                    "max_bytes": 65536,
                },
                "semantic_cache": {
                    "enabled": True,
                    "threshold": 0.90,
                    "timeout_seconds": 5.0,
                    "max_entries": 16,
                    "families": [
                        {
                            "id": "france",
                            "domain": "public-geography",
                            "answer_contract": "Only the name Paris; no commentary.",
                            "prompts": [FRANCE, PARAPHRASE, THIRD],
                        },
                        {
                            "id": "japan",
                            "domain": "public-geography",
                            "answer_contract": "Only the name Tokyo; no commentary.",
                            "prompts": [JAPAN, JAPAN_PARAPHRASE],
                        },
                    ],
                },
                "rate_limit": {"burst": 200, "requests_per_second": 0.01},
            }
            | changes
        )
    )


class FakeEncoder:
    fingerprint = "synthetic-normalized-384-v1"
    max_tokens = 256

    def __init__(self):
        self.vectors = {}
        self.calls = []
        self.held_prompt = None
        self.started = threading.Event()
        self.finished = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.peak = 0

    def encode(self, text):
        self.calls.append(text)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if text == self.held_prompt:
                self.started.set()
                if not self.release.wait(timeout=5):
                    raise RuntimeError("test failed to release the embedding worker")
            result = np.zeros(384, dtype=np.float32)
            result[self.vectors.get(text, 0)] = 1.0
            return result
        finally:
            self.active -= 1
            self.finished.set()


class BackendProbe:
    def __init__(self):
        self.calls = []

    def __call__(self, request):
        self.calls.append(request)
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        result = completion(id=f"chatcmpl-{len(self.calls)}", model=body["model"])
        result["choices"][0]["message"]["content"] = "Tokyo" if "Japan" in prompt else "Paris"
        return httpx.Response(200, json=result)


@asynccontextmanager
async def semantic_gateway(encoder=None, backend=None, settings=None):
    encoder = encoder or FakeEncoder()
    backend = backend or BackendProbe()
    app = create_app(
        settings or semantic_settings(),
        transport=httpx.MockTransport(backend),
        embedding_encoder=encoder,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as client:
            try:
                yield client, app, encoder, backend
            finally:
                encoder.release.set()


async def chat(client, prompt=FRANCE, *, headers=None, **changes):
    return await client.post(
        "/v1/chat/completions",
        json=payload(prompt, **changes),
        headers=SEMANTIC_AUTH if headers is None else headers,
    )


def scope_and_receive(disconnected):
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {
                "type": "http.request",
                "body": json.dumps(payload()).encode(),
                "more_body": False,
            }
        await disconnected.wait()
        return {"type": "http.disconnect"}

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
            (name.lower().encode(), value.encode()) for name, value in SEMANTIC_AUTH.items()
        ]
        + [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }
    return scope, receive


async def wait_for_thread(event):
    assert await asyncio.to_thread(event.wait, 2), "embedding worker did not start"


async def test_paraphrase_reuses_actual_answer_and_wire_bytes_without_alias_insertion():
    async with semantic_gateway() as (client, app, encoder, backend):
        first = await chat(client)
        reused = await chat(client, PARAPHRASE)
        assert first.status_code == reused.status_code == 200
        assert first.headers["X-Gateway-Cache"] == "MISS"
        assert reused.headers["X-Gateway-Cache"] == "SEMANTIC_HIT"
        assert reused.json()["choices"][0]["message"]["content"] == "Paris"
        assert reused.content == first.content
        assert "adapter;" not in reused.headers["Server-Timing"]
        assert "semantic;" in reused.headers["Server-Timing"]
        assert reused.headers["Cache-Control"] == "no-store"
        assert app.state.response_cache.snapshot()["entries"] == 1
        assert len(backend.calls) == 1
        calls = len(encoder.calls)
        exact = await chat(client)
        assert exact.headers["X-Gateway-Cache"] == "HIT"
        assert len(encoder.calls) == calls
        assert app.state.semantic_cache.stats.hits == 1


async def test_same_reviewed_family_below_threshold_runs_inference():
    encoder = FakeEncoder()
    encoder.vectors[PARAPHRASE] = 1
    async with semantic_gateway(encoder) as (client, app, _, backend):
        await chat(client)
        result = await chat(client, PARAPHRASE)
        assert result.headers["X-Gateway-Cache"] == "MISS"
        assert result.json()["id"] == "chatcmpl-2"
        assert len(backend.calls) == 2
        assert app.state.semantic_cache.stats.hits == 0


async def test_all_frozen_cases_at_wire_boundary_with_explicit_context_mapping():
    from devtools.semantic_eval import load_dataset

    dataset, families, registry = load_dataset()
    settings = semantic_settings()
    from llm_gateway.config import SemanticCacheConfig

    settings.config.semantic_cache = SemanticCacheConfig(
        enabled=True,
        families=[
            {
                "id": family["id"],
                "domain": family["scope"],
                "answer_contract": family["response_requirement"],
                "prompts": [family["canonical"]] + sum(family["paraphrases"].values(), []),
            }
            for family in families.values()
        ],
    )
    answers = {
        "france-capital": "Paris",
        "japan-capital": "Tokyo",
        "python3-list-tuple": (
            "- Lists are mutable; tuples are immutable.\n"
            "- List literals use []; tuple literals use commas, commonly with ()."
        ),
    }
    calls = 0

    def backend(request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        family = registry.get(body["messages"][-1]["content"])
        result = completion(id=f"chatcmpl-{calls}")
        result["choices"][0]["message"]["content"] = (
            answers[family["id"]] if family else "Uncached negative fixture answer."
        )
        return httpx.Response(200, json=result)

    async with semantic_gateway(settings=settings, backend=backend) as (client, app, encoder, _):
        originals = {}
        for family in families.values():
            response = await chat(client, family["canonical"])
            assert response.headers["X-Gateway-Cache"] == "MISS"
            originals[family["id"]] = response.content
        contextual = positives = negatives = 0
        for case in dataset["cases"]:
            family = registry.get(case["text"])
            messages = [{"role": "user", "content": case["text"]}]
            if family and any(
                case[field] != family[field] for field in ("scope", "response_requirement")
            ):
                # These five frozen cases carry external metadata, not wire
                # fields. Express it explicitly through supported system context.
                # Sending the bare approved user text alone remains eligible.
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": (
                            f"Context: {case['scope']}. "
                            f"Required response contract: {case['response_requirement']}."
                        ),
                    },
                )
                contextual += 1
            encoded_before = len(encoder.calls)
            result = await chat(client, case["text"], messages=messages)
            assert result.status_code == 200
            if expected := case["expected_family"]:
                positives += 1
                assert result.headers["X-Gateway-Cache"] == "SEMANTIC_HIT"
                assert result.content == originals[expected]
                assert result.json()["choices"][0]["message"]["content"] == answers[expected]
            else:
                negatives += 1
                assert result.headers["X-Gateway-Cache"] == "BYPASS"
                assert len(encoder.calls) == encoded_before
        assert (positives, negatives, contextual) == (18, 36, 5)
        assert calls == app.state.response_cache.stats.upstream_calls == 39
        assert app.state.semantic_cache.stats.hits == 18


async def test_identical_vectors_cannot_cross_reviewed_family_or_unknown_constraint():
    async with semantic_gateway() as (client, app, encoder, backend):
        original = await chat(client)
        japanese = await chat(client, JAPAN)
        assert japanese.headers["X-Gateway-Cache"] == "MISS"
        assert japanese.json()["choices"][0]["message"]["content"] == "Tokyo"
        encoded = len(encoder.calls)
        for prompt in [
            "What is NOT the capital of France?",
            "Name two cities in France.",
            "Name France's capital and explain its history.",
            "What is my private account's recovery code?",
        ]:
            rejected = await chat(client, prompt)
            assert rejected.headers["X-Gateway-Cache"] == "BYPASS"
            assert rejected.content != original.content
        assert len(encoder.calls) == encoded
        assert app.state.semantic_cache.stats.hits == 0
        assert len(backend.calls) == 6


@pytest.mark.parametrize("changes", [{"seed": 8}, {"max_tokens": 16}, {"model": "other"}])
async def test_semantic_reuse_is_partitioned_by_generation_and_model(changes):
    async with semantic_gateway() as (client, _, _, backend):
        await chat(client)
        isolated = await chat(client, PARAPHRASE, **changes)
        assert isolated.headers["X-Gateway-Cache"] == "MISS"
        assert isolated.json()["id"] == "chatcmpl-2"
        assert len(backend.calls) == 2
        assert (await chat(client, PARAPHRASE, **changes)).headers["X-Gateway-Cache"] == "HIT"


async def test_semantic_entries_never_cross_authenticated_clients():
    key = "semantic-second-client-" + "z" * 32
    settings = semantic_settings()
    settings = replace(
        settings,
        client_key_hashes=settings.client_key_hashes
        | {hashlib.sha256(key.encode()).digest(): "client-b"},
    )
    other_auth = {"Authorization": f"Bearer {key}", "X-Gateway-Cache": "semantic"}
    async with semantic_gateway(settings=settings) as (client, _, _, backend):
        original = await chat(client)
        other = await chat(client, PARAPHRASE, headers=other_auth)
        assert other.headers["X-Gateway-Cache"] == "MISS"
        assert other.json()["id"] == "chatcmpl-2"
        own = await chat(client, PARAPHRASE)
        assert own.headers["X-Gateway-Cache"] == "SEMANTIC_HIT"
        assert own.content == original.content
        assert len(backend.calls) == 2


async def test_semantic_hits_require_auth_and_still_consume_request_quota():
    settings = semantic_settings(rate_limit={"burst": 2, "requests_per_second": 0.01})
    async with semantic_gateway(settings=settings) as (client, _, encoder, backend):
        await chat(client)
        assert (
            await chat(client, PARAPHRASE, headers={"X-Gateway-Cache": "semantic"})
        ).status_code == 401
        assert (await chat(client, PARAPHRASE)).headers["X-Gateway-Cache"] == "SEMANTIC_HIT"
        encoded = len(encoder.calls)
        denied = await chat(client, THIRD)
        assert denied.status_code == 429
        assert len(encoder.calls) == encoded
        assert len(backend.calls) == 1


async def test_worker_busy_falls_back_immediately_and_semantic_hit_bypasses_full_capacity():
    encoder = FakeEncoder()
    backend = BackendProbe()
    upstream_started, release_upstream = asyncio.Event(), asyncio.Event()

    async def handler(request):
        if json.loads(request.content)["messages"][-1]["content"] == "live uncached request":
            upstream_started.set()
            await release_upstream.wait()
        return backend(request)

    async with semantic_gateway(encoder, handler, semantic_settings(max_in_flight=1)) as (
        client,
        app,
        _,
        _,
    ):
        original = await chat(client)
        encoder.held_prompt = PARAPHRASE
        held = asyncio.create_task(chat(client, PARAPHRASE))
        live = None
        try:
            await wait_for_thread(encoder.started)
            assert app.state.admission.active == 0
            assert (await asyncio.wait_for(client.get("/healthz"), 1)).status_code == 200
            live = asyncio.create_task(chat(client, "live uncached request"))
            await asyncio.wait_for(upstream_started.wait(), 1)
            rejected = await asyncio.wait_for(chat(client, THIRD), 1)
            assert rejected.status_code == 503
            assert rejected.json()["error"]["code"] == "gateway_overloaded"
            assert encoder.calls == [FRANCE, PARAPHRASE]
            assert app.state.semantic_cache.stats.busy == 1
            encoder.release.set()
            result = await asyncio.wait_for(held, 2)
            assert result.headers["X-Gateway-Cache"] == "SEMANTIC_HIT"
            assert result.content == original.content
            assert app.state.admission.active == 1
            assert encoder.peak == 1
            release_upstream.set()
            assert (await live).status_code == 200
            assert len(backend.calls) == 2
        finally:
            encoder.release.set()
            release_upstream.set()
            for task in [held, live]:
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*[task for task in [held, live] if task], return_exceptions=True)


async def test_expired_candidate_is_rechecked_after_embedding_without_sliding_ttl():
    encoder = FakeEncoder()
    now = [0.0]
    settings = semantic_settings(
        exact_cache={
            "enabled": True,
            "ttl_seconds": 10.0,
            "max_entries": 32,
            "max_bytes": 65536,
        }
    )
    async with semantic_gateway(encoder, settings=settings) as (client, app, _, backend):
        app.state.response_cache._clock = lambda: now[0]
        await chat(client)
        encoder.held_prompt = PARAPHRASE
        task = asyncio.create_task(chat(client, PARAPHRASE))
        try:
            await wait_for_thread(encoder.started)
            now[0] = 11.0
            assert app.state.response_cache.snapshot()["entries"] == 0
            encoder.release.set()
            response = await asyncio.wait_for(task, 2)
            assert response.headers["X-Gateway-Cache"] == "MISS"
            assert response.json()["id"] == "chatcmpl-2"
            assert len(backend.calls) == 2
        finally:
            encoder.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_embedding_timeout_falls_back_without_releasing_a_running_native_job():
    encoder = FakeEncoder()
    encoder.held_prompt = FRANCE
    settings = semantic_settings()
    settings.config.semantic_cache.timeout_seconds = 0.02
    async with semantic_gateway(encoder, settings=settings) as (client, app, _, backend):
        first = await asyncio.wait_for(chat(client), 1)
        assert first.status_code == 200
        assert first.headers["X-Gateway-Cache"] == "MISS"
        assert encoder.active == 1
        assert app.state.semantic_cache.stats.failures == 1
        assert (await client.get("/healthz")).status_code == 200
        fallback = await asyncio.wait_for(chat(client, PARAPHRASE), 1)
        assert fallback.status_code == 200
        assert fallback.headers["X-Gateway-Cache"] == "MISS"
        assert encoder.calls == [FRANCE]
        assert app.state.semantic_cache.stats.busy == 1
        assert len(backend.calls) == 2
        encoder.release.set()
        await wait_for_thread(encoder.finished)
        # Completing the abandoned embedding does not publish a late response.
        assert app.state.response_cache.snapshot()["entries"] == 2


async def test_cancelling_embedding_keeps_worker_busy_and_publishes_no_late_candidate():
    encoder = FakeEncoder()
    encoder.held_prompt = FRANCE
    async with semantic_gateway(encoder) as (client, app, _, backend):
        task = asyncio.create_task(chat(client))
        try:
            await wait_for_thread(encoder.started)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            assert app.state.response_cache.snapshot()["entries"] == 0
            assert app.state.admission.active == 0
            assert encoder.active == 1
            fallback = await asyncio.wait_for(chat(client, PARAPHRASE), 1)
            assert fallback.status_code == 200
            assert fallback.headers["X-Gateway-Cache"] == "MISS"
            assert encoder.calls == [FRANCE]
            assert len(backend.calls) == 1
            encoder.release.set()
            await wait_for_thread(encoder.finished)
            assert app.state.response_cache.snapshot()["entries"] == 1
        finally:
            encoder.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("outcome", ["backend_error", "timeout", "length", "blank", "private"])
async def test_failed_or_unsuitable_backend_results_never_retain_semantic_entries(outcome):
    async def handler(_):
        if outcome == "backend_error":
            return httpx.Response(503)
        if outcome == "timeout":
            await asyncio.Event().wait()
        result = completion()
        result["choices"][0]["message"]["content"] = "Paris"
        headers = {}
        if outcome == "length":
            result["choices"][0]["finish_reason"] = "length"
        elif outcome == "blank":
            result["choices"][0]["message"]["content"] = " "
        elif outcome == "private":
            headers["Cache-Control"] = "private"
        return httpx.Response(200, json=result, headers=headers)

    async with semantic_gateway(
        backend=handler, settings=semantic_settings(request_timeout_seconds=0.02)
    ) as (client, app, _, _):
        response = await chat(client)
        assert response.status_code == {"backend_error": 503, "timeout": 504}.get(outcome, 200)
        assert app.state.response_cache.snapshot()["entries"] == 0
        assert app.state.response_cache.stats.stores == 0


@pytest.mark.parametrize("failure", [OSError, asyncio.CancelledError])
async def test_final_send_failure_does_not_publish_a_semantic_candidate(failure):
    async with semantic_gateway() as (client, app, _, backend):
        scope, receive = scope_and_receive(asyncio.Event())

        async def send(message):
            if message["type"] == "http.response.body":
                raise failure("simulated final-send failure")

        with pytest.raises(failure):
            await app(scope, receive, send)
        assert app.state.response_cache.snapshot()["entries"] == 0
        next_response = await chat(client, PARAPHRASE)
        assert next_response.headers["X-Gateway-Cache"] == "MISS"
        assert len(backend.calls) == 2


@pytest.mark.parametrize("disconnect", [False, True])
async def test_cancelled_or_disconnected_inference_never_publishes_semantic_entry(disconnect):
    started, closed, departed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    backend = BackendProbe()
    calls = 0

    class WaitingBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self):
            closed.set()

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200, headers={"Content-Type": "application/json"}, stream=WaitingBody()
            )
        return backend(request)

    async with semantic_gateway(backend=handler) as (client, app, _, _):
        scope, receive = scope_and_receive(departed)
        sent = []

        async def send(message):
            sent.append(message)

        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(started.wait(), 1)
            if disconnect:
                departed.set()
                await asyncio.wait_for(task, 1)
                assert sent[0]["status"] == 499
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert closed.is_set()
            assert app.state.admission.active == 0
            assert app.state.response_cache.snapshot()["entries"] == 0
            assert (await chat(client, PARAPHRASE)).headers["X-Gateway-Cache"] == "MISS"
        finally:
            departed.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
