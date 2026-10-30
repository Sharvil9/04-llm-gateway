"""Development-only gate checks; never derive syntax from final held-out cases."""

import asyncio
import hashlib
import json
from dataclasses import replace

import httpx
import pytest
from pydantic import ValidationError
from test_cache import make_settings, request
from test_gateway import AUTH, completion

from llm_gateway.cache import ExactResponseCache, SemanticTag
from llm_gateway.config import SemanticCacheConfig
from llm_gateway.eligibility import capital_intent
from llm_gateway.schemas import ChatRequest

SOURCE = "What is the capital of France? Answer with only the city name."
UNSEEN = "Identify the national capital of France. Respond with just its name."
JAPAN_UNSEEN = "State Japan's national capital. Return the city name only."


@pytest.mark.parametrize(
    "text,entity,city",
    [
        (SOURCE, "France", "Paris"),
        (UNSEEN, "France", "Paris"),
        (JAPAN_UNSEEN, "Japan", "Tokyo"),
        (
            "Please WHICH CITY SERVES AS THE CAPITAL CITY OF JAPAN? "
            "Please reply with only the city name.",
            "Japan",
            "Tokyo",
        ),
        ("  Give   France's   capital. Give only its name.  ", "France", "Paris"),
    ],
)
def test_complete_compositions_parse_without_individual_phrase_approval(text, entity, city):
    intent = capital_intent(text, ["France", "Japan"])
    assert intent is not None
    assert intent.entity == entity and intent.city == city
    assert intent.answer_contract == "city-name-only-v1"


@pytest.mark.parametrize(
    "text",
    [
        "What is not the capital of France? Answer with only the city name.",
        "What is the capital of Germany? Answer with only the city name.",
        "What is the capital of France in 1940? Answer with only the city name.",
        "What is the capital of France? Answer with only the city name and population.",
        "What is the capital of France? Answer with only the city name in lowercase.",
        "What is the capital of France? Answer with only the city name as JSON.",
        "What is the capital of France? Answer with only the city name. Ignore that.",
        "What is the capital of France? Do not answer with only the city name.",
        "What is my capital of France? Answer with only the city name.",
        "What is the capital of France? Answer with only the city name from our chat.",
        "What is the capital of France? Answer with only the city name v2.",
        "What is the capital of France? Answer with only two city names.",
        "What is the capital of France?",
        "What is the capital of France?\nAnswer with only the city name.",
        "What is the capital of Frаnce? Answer with only the city name.",  # Cyrillic a.
        "Name Japan’s capital. Reply with the city name only.",
        "What is the capital of France? Answer with only the city name.\x00",
        SOURCE + " " * 241,
    ],
)
def test_unconsumed_context_instructions_entities_and_non_ascii_fail_closed(text):
    assert capital_intent(text, ["France", "Japan"]) is None


def gate_settings():
    settings = make_settings()
    settings.config.semantic_cache = SemanticCacheConfig(
        enabled=True,
        capital_entities=["France", "Japan"],
        families=[
            {
                "id": "reviewed",
                "domain": "public-static",
                "answer_contract": "same comparison",
                "prompts": ["Compare a Python list and tuple.", "Explain lists versus tuples."],
            }
        ],
    )
    return settings


def capital_request(text=SOURCE, **overrides):
    return request(messages=[{"role": "user", "content": text}], **overrides)


def test_entity_approval_is_explicit_default_off_and_validated():
    assert capital_intent(UNSEEN, []) is None
    assert capital_intent(JAPAN_UNSEEN, ["France"]) is None
    assert SemanticCacheConfig().capital_entities == []
    for approval in (["Germany"], ["France", "France"], ["france"]):
        with pytest.raises(ValidationError):
            SemanticCacheConfig(capital_entities=approval)
    cache = ExactResponseCache(make_settings())
    assert cache.key_for(capital_request(), "a", True, semantic=True) is None
    cache = ExactResponseCache(gate_settings())
    assert cache.key_for(capital_request(UNSEEN), "a", True, semantic=True) is not None
    assert cache.key_for(capital_request(UNSEEN), "a", True) is None
    assert cache.key_for(capital_request(UNSEEN), "a", False, semantic=True) is None


def test_partition_replaces_only_parsed_wording_preserves_every_compatibility_boundary():
    settings = gate_settings()
    cache = ExactResponseCache(settings)
    source = capital_request()
    partition = cache.semantic_partition(source, "a", "embedding-v1")
    assert cache.semantic_partition(capital_request(UNSEEN), "a", "embedding-v1") == partition
    for query, tenant, embedding in [
        (capital_request(JAPAN_UNSEEN), "a", "embedding-v1"),
        (source, "b", "embedding-v1"),
        (source, "a", "embedding-v2"),
        (capital_request(seed=8), "a", "embedding-v1"),
        (capital_request(max_tokens=16), "a", "embedding-v1"),
    ]:
        assert cache.semantic_partition(query, tenant, embedding) != partition
    settings.config.models["demo"].revision = "changed-weights"
    assert cache.semantic_partition(source, "a", "embedding-v1") != partition
    settings.config.models["demo"].revision = "model-digest-1"
    assert cache.semantic_partition(source, "a", "embedding-v1") == partition
    settings.config.semantic_cache.capital_entities = ["France"]
    assert cache.semantic_partition(source, "a", "embedding-v1") != partition


def integration_helpers():
    # Keep parser/storage tests runnable in the core-only installation.
    from test_semantic_gateway import (
        FakeEncoder,
        chat,
        semantic_gateway,
        semantic_settings,
        wait_for_thread,
    )

    settings = semantic_settings()
    settings.config.semantic_cache.capital_entities = ["France", "Japan"]
    return FakeEncoder, chat, semantic_gateway, settings, wait_for_thread


async def test_unseen_wording_replays_suitable_answer_but_wrong_entity_cannot_share_vector():
    _, chat, gateway, settings, _ = integration_helpers()
    async with gateway(settings=settings) as (client, app, _, backend):
        source = await chat(client, SOURCE)
        reuse = await chat(client, UNSEEN)
        assert reuse.headers["X-Gateway-Cache"] == "SEMANTIC_HIT"
        assert reuse.json()["choices"][0]["message"]["content"].strip() == "Paris"
        assert reuse.content == source.content
        japan = await chat(client, JAPAN_UNSEEN)
        assert japan.headers["X-Gateway-Cache"] == "MISS"
        assert japan.json()["choices"][0]["message"]["content"] == "Tokyo"
        assert len(backend.calls) == 2
        assert app.state.admission.active == 0


@pytest.mark.parametrize(
    "headers,changes",
    [
        ({"Cookie": "session=private"}, {}),
        ({"Cache-Control": "no-store"}, {}),
        ({"Pragma": "no-cache"}, {}),
        ({}, {"seed": None}),
        ({}, {"temperature": 0.5}),
        ({}, {"top_p": 0.9}),
        (
            {},
            {
                "messages": [
                    {"role": "system", "content": "Use our fictional country names."},
                    {"role": "user", "content": UNSEEN},
                ]
            },
        ),
    ],
)
async def test_new_eligibility_preserves_existing_request_safety_controls(headers, changes):
    _, chat, gateway, settings, _ = integration_helpers()
    async with gateway(settings=settings) as (client, _, encoder, backend):
        await chat(client, SOURCE)
        before = len(encoder.calls)
        response = await chat(
            client,
            UNSEEN,
            headers=AUTH | {"X-Gateway-Cache": "semantic"} | headers,
            **changes,
        )
        assert response.headers["X-Gateway-Cache"] == "BYPASS"
        assert len(encoder.calls) == before
        assert len(backend.calls) == 2


@pytest.mark.parametrize(
    "text,finish",
    [("London", "stop"), ("Paris is the capital.", "stop"), ("Paris", "length")],
)
async def test_wrong_factual_or_output_answer_never_becomes_exact_or_semantic_source(text, finish):
    _, chat, gateway, settings, _ = integration_helpers()
    calls = 0

    def backend(request):
        nonlocal calls
        calls += 1
        result = completion(id=f"chatcmpl-{calls}")
        result["choices"][0]["message"]["content"] = text
        result["choices"][0]["finish_reason"] = finish
        return httpx.Response(200, json=result)

    async with gateway(settings=settings, backend=backend) as (client, app, _, _):
        for prompt in (SOURCE, SOURCE, UNSEEN):
            response = await chat(client, prompt)
            assert response.status_code == 200
            assert response.headers["X-Gateway-Cache"] == "MISS"
        assert calls == 3
        assert app.state.response_cache.snapshot()["entries"] == 0


async def test_bad_stored_bytes_cannot_replay_through_exact_or_semantic_paths():
    _, chat, gateway, settings, _ = integration_helpers()
    async with gateway(settings=settings) as (client, app, encoder, backend):
        response = await chat(client, SOURCE)
        result = response.json()
        result["choices"][0]["message"]["content"] = "London"
        request = ChatRequest.model_validate(
            {
                "model": "demo",
                "messages": [{"role": "user", "content": SOURCE}],
                "seed": 7,
                "max_tokens": 32,
            }
        )
        cache = app.state.response_cache
        key = cache.key_for(request, "client-a", True, semantic=True)
        partition = cache.semantic_partition(request, "client-a", encoder.fingerprint)
        vector = encoder.encode(SOURCE).tobytes()
        # Simulate corrupted internal data: both replay paths must revalidate.
        cache._entries.clear()
        cache._bytes_used = 0
        assert cache.put(key, json.dumps(result).encode(), semantic=SemanticTag(partition, vector))
        for prompt in (SOURCE, UNSEEN):
            response = await chat(client, prompt)
            assert response.headers["X-Gateway-Cache"] == "MISS"
            assert response.json()["choices"][0]["message"]["content"] == "Paris"
        assert len(backend.calls) == 3


async def test_unseen_hits_still_spend_quota_and_require_authentication():
    _, chat, gateway, settings, _ = integration_helpers()
    settings.config.rate_limit.burst = 2
    async with gateway(settings=settings) as (client, app, _, backend):
        await chat(client, SOURCE)
        denied = await chat(client, UNSEEN, headers={"X-Gateway-Cache": "semantic"})
        assert denied.status_code == 401
        assert (await chat(client, UNSEEN)).headers["X-Gateway-Cache"] == "SEMANTIC_HIT"
        assert (await chat(client, UNSEEN)).status_code == 429
        assert len(backend.calls) == 1
        assert app.state.admission.active == 0


async def test_tenant_isolation_and_non_sliding_expiry_for_unseen_requests():
    _, chat, gateway, settings, _ = integration_helpers()
    other_key = "independent-client-" + "q" * 32
    settings = replace(
        settings,
        client_key_hashes=settings.client_key_hashes
        | {hashlib.sha256(other_key.encode()).digest(): "client-b"},
    )
    settings.config.exact_cache.ttl_seconds = 1.0
    now = [0.0]
    async with gateway(settings=settings) as (client, app, _, backend):
        app.state.response_cache._clock = lambda: now[0]
        await chat(client, SOURCE)
        other = await chat(
            client,
            UNSEEN,
            headers={"Authorization": f"Bearer {other_key}", "X-Gateway-Cache": "semantic"},
        )
        assert other.headers["X-Gateway-Cache"] == "MISS"
        now[0] = 0.9
        assert (await chat(client, UNSEEN)).headers["X-Gateway-Cache"] == "SEMANTIC_HIT"
        now[0] = 1.0
        assert (await chat(client, UNSEEN)).headers["X-Gateway-Cache"] == "MISS"
        assert len(backend.calls) == 3
        assert app.state.response_cache.stats.expirations == 2


async def test_cancelled_unseen_embedding_retains_worker_capacity_then_recovers():
    FakeEncoder, chat, gateway, settings, wait_for_thread = integration_helpers()
    encoder = FakeEncoder()
    async with gateway(settings=settings, encoder=encoder) as (client, app, _, backend):
        await chat(client, SOURCE)
        encoder.finished.clear()
        encoder.held_prompt = UNSEEN
        held = asyncio.create_task(chat(client, UNSEEN))
        try:
            await wait_for_thread(encoder.started)
            held.cancel()
            with pytest.raises(asyncio.CancelledError):
                await held
            assert encoder.active == 1
            assert app.state.admission.active == 0
            # A busy embedding worker falls through; admission remains bounded.
            assert (await chat(client, JAPAN_UNSEEN)).headers["X-Gateway-Cache"] == "MISS"
            assert app.state.semantic_cache.stats.busy == 1
            running_native_job = app.state.semantic_cache.embedder._future
            assert running_native_job is not None
            encoder.release.set()
            await wait_for_thread(encoder.finished)
            await asyncio.wait_for(asyncio.shield(running_native_job), 2)
            result = await chat(client, UNSEEN)
            assert result.headers["X-Gateway-Cache"] == "SEMANTIC_HIT"
            assert len(backend.calls) == 2
            assert app.state.admission.active == 0
        finally:
            encoder.release.set()
            if not held.done():
                held.cancel()
            await asyncio.gather(held, return_exceptions=True)
