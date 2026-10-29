"""Policy/storage invariants independent of optional native embedding packages."""

import asyncio
import struct
import time

import pytest
from pydantic import ValidationError
from test_cache import make_settings, request

from llm_gateway.cache import ExactResponseCache, SemanticTag
from llm_gateway.config import GatewayConfig, SemanticCacheConfig
from llm_gateway.semantic import SemanticCache

PHRASE = "Explain the difference between a Python list and tuple."
PARAPHRASE = "Compare a Python list with a tuple."
VECTOR = struct.pack("384f", 1.0, *([0.0] * 383))


def settings(**overrides):
    result = make_settings()
    result.config.semantic_cache = SemanticCacheConfig.model_validate(
        {
            "enabled": True,
            "threshold": 0.9,
            "max_entries": 2,
            "families": [
                {
                    "id": "python",
                    "domain": "public-static-v1",
                    "answer_contract": "same comparison",
                    "prompts": [PHRASE, PARAPHRASE],
                }
            ],
            **overrides,
        }
    )
    return result


def test_semantic_approval_is_explicit_unambiguous_and_requires_exact_store():
    with pytest.raises(ValidationError):
        SemanticCacheConfig(enabled=True)
    family = settings().config.semantic_cache.families[0].model_dump()
    with pytest.raises(ValidationError):
        SemanticCacheConfig(families=[family, family | {"id": "other"}])
    with pytest.raises(ValidationError):
        SemanticCacheConfig(families=[family | {"prompts": [PHRASE, PHRASE]}])
    values = settings().config.model_dump()
    values["exact_cache"]["enabled"] = False
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate(values)


def test_only_wording_varies_in_semantic_partition_all_other_boundaries_stay():
    config = settings()
    cache = ExactResponseCache(config)
    first = request()
    second = request(messages=[{"role": "user", "content": PARAPHRASE}])
    partition = cache.semantic_partition(first, "a", "embedding-v1")
    assert cache.semantic_partition(second, "a", "embedding-v1") == partition
    assert cache.key_for(second, "a", True) is None
    assert cache.key_for(second, "a", True, semantic=True) is not None
    assert cache.key_for(second, "a", False, semantic=True) is None
    for client, fingerprint, payload in (
        ("b", "embedding-v1", first),
        ("a", "embedding-v2", first),
        ("a", "embedding-v1", request(seed=43)),
        ("a", "embedding-v1", request(max_tokens=32)),
    ):
        assert cache.semantic_partition(payload, client, fingerprint) != partition
    config.config.models["demo"].revision = "another-artifact"
    assert cache.semantic_partition(first, "a", "embedding-v1") != partition
    for phrase in ("Compare my private lists.", PHRASE + " ", "Do not " + PHRASE):
        payload = request(messages=[{"role": "user", "content": phrase}])
        assert cache.key_for(payload, "a", True, semantic=True) is None
        with pytest.raises(ValueError):
            cache.semantic_partition(payload, "a", "embedding-v1")


def test_vectors_share_response_byte_budget_ttl_lru_and_never_extend_lifetime():
    now = 0.0
    config = settings(max_entries=1)
    config.config.exact_cache.ttl_seconds = 2.0
    config.config.exact_cache.max_bytes = 2048
    cache = ExactResponseCache(config, clock=lambda: now)
    tag = SemanticTag(b"p" * 32, VECTOR)
    key, other = b"a" * 32, b"b" * 32
    assert cache.put(key, b"response", semantic=tag)
    assert cache.snapshot()["bytes_used"] == 32 + 8 + 32 + 1536
    assert cache.semantic_candidates(tag.partition) == [(key, VECTOR)]
    assert cache.semantic_get(key, b"q" * 32) is None
    assert cache.semantic_get(key, tag.partition) == b"response"
    assert cache.snapshot()["hits"] == 0  # Exact and semantic counters differ.
    assert not cache.put(other, b"x" * 1000, semantic=tag)
    assert cache.semantic_get(key, tag.partition) == b"response"
    assert cache.put(other, b"new", semantic=tag)
    assert cache.semantic_get(key, tag.partition) is None
    assert cache.stats.evictions == 1
    now = 1.0
    assert cache.semantic_get(other, tag.partition) == b"new"
    now = 2.0
    assert cache.semantic_candidates(tag.partition) == []
    assert cache.semantic_get(other, tag.partition) is None
    assert cache.snapshot()["bytes_used"] == 0
    assert cache.stats.expirations == 1


def test_response_eviction_cannot_leave_a_stale_semantic_index():
    config = settings()
    config.config.exact_cache.max_entries = 1
    cache = ExactResponseCache(config)
    tag = SemanticTag(b"p" * 32, VECTOR)
    assert cache.put(b"a" * 32, b"old", semantic=tag)
    assert cache.put(b"b" * 32, b"exact only")
    assert cache.semantic_candidates(tag.partition) == []
    assert cache.snapshot()["bytes_used"] == 42


async def test_lookup_rechecks_expiry_after_async_search_and_does_not_refresh():
    now = 0.0
    config = settings()
    config.config.exact_cache.ttl_seconds = 1.0
    cache = ExactResponseCache(config, clock=lambda: now)
    partition = cache.semantic_partition(request(), "a", "fake-v1")
    key = cache.key_for(request(), "a", True, semantic=True)
    cache.put(key, b"Paris", semantic=SemanticTag(partition, VECTOR))

    class Matcher:
        fingerprint = "fake-v1"

        async def match(self, text, candidates, threshold):
            nonlocal now
            assert candidates == [(key, VECTOR)]
            now = 1.0
            await asyncio.sleep(0)
            return VECTOR, key, 1.0

    semantic = SemanticCache(cache, Matcher(), config.config.semantic_cache)
    payload, tag = await semantic.lookup(request(), "a")
    assert payload is None and tag == SemanticTag(partition, VECTOR)
    assert semantic.stats.misses == 1
    assert cache.snapshot()["entries"] == 0


async def test_embedding_failure_busy_and_cancel_do_not_make_cached_answers():
    cache = ExactResponseCache(settings())

    class Matcher:
        fingerprint = "fake-v1"
        failure = ValueError("private text must not escape")

        async def match(self, *args):
            if self.failure:
                raise self.failure
            return None

    matcher = Matcher()
    semantic = SemanticCache(cache, matcher, settings().config.semantic_cache)
    assert await semantic.lookup(request(), "a") == (None, None)
    matcher.failure = None
    assert await semantic.lookup(request(), "a") == (None, None)
    matcher.failure = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await semantic.lookup(request(), "a")
    assert semantic.snapshot() == {"hits": 0, "misses": 2, "busy": 1, "failures": 1}
    assert cache.snapshot()["entries"] == 0


async def test_starved_loop_cannot_accept_a_semantic_result_after_lookup_budget():
    config = settings(timeout_seconds=0.01)
    cache = ExactResponseCache(config)
    partition = cache.semantic_partition(request(), "a", "fake-v1")
    key = cache.key_for(request(), "a", True)
    cache.put(key, b"old answer", semantic=SemanticTag(partition, VECTOR))

    class BlockingMatcher:
        fingerprint = "fake-v1"

        async def match(self, *args):
            # Deliberately simulate loop starvation; production encoder work
            # runs in a dedicated thread. A timer alone cannot preempt this.
            time.sleep(0.03)  # noqa: ASYNC251 -- deliberate loop starvation regression
            return VECTOR, key, 1.0

    semantic = SemanticCache(cache, BlockingMatcher(), config.config.semantic_cache)
    assert await semantic.lookup(request(), "a") == (None, None)
    assert semantic.stats.failures == 1 and semantic.stats.hits == 0
