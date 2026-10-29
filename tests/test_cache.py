import asyncio
import hashlib
import json

import pytest

import llm_gateway.cache as cache_module
from llm_gateway.cache import ExactResponseCache
from llm_gateway.config import GatewayConfig, Settings
from llm_gateway.schemas import ChatRequest

PUBLIC_PROMPT = "Explain the difference between a Python list and tuple."


def make_settings(**cache_overrides) -> Settings:
    return Settings(
        config=GatewayConfig.model_validate(
            {
                "backends": {
                    "local": {
                        "base_url": "http://backend.invalid/v1",
                        "api_key_env": "BACKEND_KEY",
                        "revision": "backend-build-1",
                    }
                },
                "models": {
                    "demo": {
                        "backend": "local",
                        "model": "physical-model",
                        "revision": "model-digest-1",
                    }
                },
                "exact_cache": {
                    "enabled": True,
                    "approved_prompts": [PUBLIC_PROMPT],
                    **cache_overrides,
                },
            }
        ),
        client_key_hashes={hashlib.sha256(b"test-client").digest(): "client-a"},
        backend_tokens={"local": "test-backend-token"},
    )


def request(**overrides) -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "model": "demo",
            "messages": [{"role": "user", "content": PUBLIC_PROMPT}],
            "seed": 42,
            **overrides,
        }
    )


def key(cache: ExactResponseCache, **overrides) -> bytes:
    result = cache.key_for(request(**overrides), "client-a", True)
    assert result is not None
    return result


def test_eligible_key_is_ephemeral_opaque_and_normalizes_default_fields() -> None:
    settings = make_settings()
    cache = ExactResponseCache(settings)
    minimal = request()
    explicit = ChatRequest.model_validate(minimal.model_dump())
    digest = cache.key_for(minimal, "client-a", True)
    assert digest == cache.key_for(explicit, "client-a", True)
    assert isinstance(digest, bytes) and len(digest) == 32
    assert PUBLIC_PROMPT.encode() not in digest
    assert settings.backend_tokens["local"].encode() not in digest
    assert digest != ExactResponseCache(settings).key_for(minimal, "client-a", True)
    assert cache.stats.bypasses == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": 0.1},
        {"top_p": 0.9},
        {"seed": None},
        {"messages": [{"role": "user", "content": PUBLIC_PROMPT + " "}]},
        {"messages": [{"role": "user", "content": "My private account balance?"}]},
        {
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": PUBLIC_PROMPT},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": PUBLIC_PROMPT},
            ]
        },
        {"model": "unknown"},
    ],
)
def test_sampling_private_contextual_and_unapproved_requests_bypass(changes) -> None:
    cache = ExactResponseCache(make_settings())
    assert cache.key_for(request(**changes), "client-a", True) is None
    assert cache.stats.bypasses == 1
    assert cache.stats.misses == 0


def test_disabled_cache_missing_opt_in_and_unpinned_identity_bypass() -> None:
    settings = make_settings(enabled=False)
    cache = ExactResponseCache(settings)
    assert cache.key_for(request(), "client-a", True) is None
    settings.config.exact_cache.enabled = True
    assert cache.key_for(request(), "client-a", False) is None
    settings.config.models["demo"].revision = None
    assert cache.key_for(request(), "client-a", True) is None
    settings.config.models["demo"].revision = "model-digest-1"
    settings.config.backends["local"].revision = None
    assert cache.key_for(request(), "client-a", True) is None
    assert cache.stats.bypasses == 4


def test_validation_bypasses_still_refuse_streaming_and_non_user_message() -> None:
    cache = ExactResponseCache(make_settings())
    assert cache.key_for(request().model_copy(update={"stream": True}), "client-a", True) is None
    invalid = request().model_copy(
        update={"messages": [request().messages[0].model_copy(update={"role": "assistant"})]}
    )
    assert cache.key_for(invalid, "client-a", True) is None
    assert cache.stats.bypasses == 2


def test_keys_partition_clients_request_parameters_config_and_credential() -> None:
    settings = make_settings()
    cache = ExactResponseCache(settings)
    baseline = key(cache)
    assert cache.key_for(request(), "client-b", True) != baseline
    assert key(cache, seed=43) != baseline
    assert key(cache, max_tokens=64) != baseline

    settings.config.models["demo"].revision = "model-digest-2"
    assert key(cache) != baseline
    settings.config.models["demo"].revision = "model-digest-1"
    settings.config.backends["local"].revision = "backend-build-2"
    assert key(cache) != baseline
    settings.config.backends["local"].revision = "backend-build-1"
    settings.config.backends["local"].base_url = "http://another-backend.invalid/v1"
    assert key(cache) != baseline
    settings.config.backends["local"].base_url = "http://backend.invalid/v1"
    settings.config.exact_cache.namespace = "public-static-v2"
    assert key(cache) != baseline
    settings.config.exact_cache.namespace = "public-static-v1"
    settings.config.max_response_bytes += 1
    assert key(cache) != baseline
    settings.config.max_response_bytes -= 1
    settings.backend_tokens["local"] = "rotated-backend-token"
    assert key(cache) != baseline
    settings.backend_tokens["local"] = "test-backend-token"
    assert key(cache) == baseline


def test_policy_code_version_invalidates_key(monkeypatch) -> None:
    cache = ExactResponseCache(make_settings())
    baseline = key(cache)
    monkeypatch.setattr(cache_module, "_POLICY_VERSION", "exact-v2")
    assert key(cache) != baseline


def test_ttl_is_non_sliding_and_duplicate_writer_does_not_replace_or_refresh() -> None:
    now = 10.0
    cache = ExactResponseCache(make_settings(ttl_seconds=2.0), clock=lambda: now)
    digest = key(cache)
    assert cache.get(digest) is None
    assert cache.put(digest, b"first")
    now = 11.0
    assert cache.get(digest) == b"first"
    assert not cache.put(digest, b"second")
    now = 11.999
    assert cache.get(digest) == b"first"
    now = 12.0
    assert cache.get(digest) is None
    assert cache.snapshot() == {
        "hits": 2,
        "misses": 2,
        "bypasses": 0,
        "stores": 1,
        "evictions": 0,
        "expirations": 1,
        "upstream_calls": 0,
        "entries": 0,
        "bytes_used": 0,
    }
    assert cache.put(digest, b"new")
    assert cache.get(digest) == b"new"


def test_lru_entry_bound_evicts_least_recently_read() -> None:
    cache = ExactResponseCache(make_settings(max_entries=2))
    first, second, third = (key(cache, seed=seed) for seed in range(3))
    assert cache.put(first, b"first")
    assert cache.put(second, b"second")
    assert cache.get(first) == b"first"
    assert cache.put(third, b"third")
    assert cache.get(second) is None
    assert cache.get(first) == b"first"
    assert cache.get(third) == b"third"
    snapshot = cache.snapshot()
    assert snapshot["entries"] == 2
    assert snapshot["evictions"] == 1
    assert snapshot["bytes_used"] == 2 * 32 + len(b"firstthird")


def test_payload_budget_includes_keys_and_oversized_skip_keeps_existing_entries() -> None:
    cache = ExactResponseCache(make_settings(max_bytes=1024))
    first, second, third = (key(cache, seed=seed) for seed in range(3))
    assert cache.put(first, b"a" * 500)
    assert cache.put(second, b"b" * 460)
    assert cache.snapshot()["bytes_used"] == 1024
    assert not cache.put(third, b"oversized" * 200)
    assert cache.get(first) == b"a" * 500
    assert cache.get(second) == b"b" * 460
    assert cache.stats.evictions == 0
    assert cache.put(third, b"c")
    assert cache.get(first) is None
    assert cache.get(second) == b"b" * 460
    assert cache.stats.evictions == 1
    assert cache.snapshot()["bytes_used"] == 32 + 460 + 32 + 1


def test_expiry_precedes_capacity_eviction_and_snapshot_reaps_once() -> None:
    now = 0.0
    cache = ExactResponseCache(make_settings(max_entries=1, ttl_seconds=1.0), clock=lambda: now)
    first, second = key(cache, seed=1), key(cache, seed=2)
    assert cache.put(first, b"first")
    now = 1.0
    assert cache.put(second, b"second")
    assert cache.stats.expirations == 1
    assert cache.stats.evictions == 0
    now = 2.0
    assert cache.snapshot()["entries"] == 0
    assert cache.snapshot()["expirations"] == 2
    assert cache.snapshot()["bytes_used"] == 0


def test_cached_bytes_are_immutable_and_decoded_responses_do_not_share_mutable_state() -> None:
    cache = ExactResponseCache(make_settings())
    digest = key(cache)
    payload = b'{"choices":[{"message":{"content":"safe answer"}}]}'
    assert cache.put(digest, payload)
    first = json.loads(cache.get(digest))
    first["choices"][0]["message"]["content"] = "mutated"
    assert json.loads(cache.get(digest))["choices"][0]["message"]["content"] == "safe answer"
    with pytest.raises(TypeError):
        cache.put(key(cache, seed=99), bytearray(payload))
    snapshot = cache.snapshot()
    snapshot["hits"] = -1
    assert cache.stats.hits == 2
    assert cache.stats.stores == 1


async def test_concurrent_writers_keep_first_payload_and_respect_entry_bound() -> None:
    cache = ExactResponseCache(make_settings(max_entries=1))
    digest = key(cache)

    async def writer(index: int) -> bool:
        await asyncio.sleep(0)
        return cache.put(digest, str(index).encode())

    results = await asyncio.gather(*(writer(index) for index in range(50)))
    assert results.count(True) == 1
    assert cache.get(digest) == b"0"
    assert cache.stats.stores == 1
    assert cache.snapshot()["entries"] == 1
