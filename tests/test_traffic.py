import asyncio

import pytest

from llm_gateway.config import RateLimitConfig
from llm_gateway.traffic import CapacityFull, InferenceAdmission, RequestLimiter


def test_limiter_refill_retry_ceiling_and_denial_preserve_tokens() -> None:
    now = 0.0
    limiter = RequestLimiter(
        ["client"],
        RateLimitConfig(burst=2, requests_per_second=0.25),
        {},
        clock=lambda: now,
    )
    assert limiter.consume("client") is None
    assert limiter.consume("client") is None
    assert limiter.consume("client") == 4
    now = 1.0
    assert limiter.consume("client") == 3
    now = 2.5
    assert limiter.consume("client") == 2
    now = 4.0
    assert limiter.consume("client") is None
    assert limiter.consume("client") == 4


def test_limiter_caps_refill_and_ignores_backward_clock() -> None:
    now = 10.0
    limiter = RequestLimiter(
        ["client"],
        RateLimitConfig(burst=2, requests_per_second=1.0),
        {},
        clock=lambda: now,
    )
    assert limiter.consume("client") is None
    now = 100.0
    assert limiter.consume("client") is None
    assert limiter.consume("client") is None
    assert limiter.consume("client") == 1
    now = 99.0
    assert limiter.consume("client") == 1
    now = 100.5
    assert limiter.consume("client") == 1
    now = 101.0
    assert limiter.consume("client") is None


async def test_concurrent_burst_is_atomic_and_client_override_is_isolated() -> None:
    limiter = RequestLimiter(
        ["a", "b"],
        RateLimitConfig(burst=7, requests_per_second=0.5),
        {"b": RateLimitConfig(burst=3, requests_per_second=0.25)},
        clock=lambda: 0.0,
    )

    async def consume(client_id: str) -> int | None:
        await asyncio.sleep(0)
        return limiter.consume(client_id)

    results_a = await asyncio.gather(*(consume("a") for _ in range(100)))
    results_b = await asyncio.gather(*(consume("b") for _ in range(100)))
    assert results_a.count(None) == 7
    assert results_a.count(2) == 93
    assert results_b.count(None) == 3
    assert results_b.count(4) == 97
    with pytest.raises(KeyError):
        limiter.consume("unconfigured")


async def test_admission_immediately_rejects_and_releases_after_cancellation() -> None:
    admission = InferenceAdmission(1)
    entered = asyncio.Event()
    wait = asyncio.Event()

    async def occupied() -> None:
        with admission.slot():
            entered.set()
            await wait.wait()

    task = asyncio.create_task(occupied())
    await entered.wait()
    assert admission.active == 1
    with pytest.raises(CapacityFull), admission.slot():
        pytest.fail("saturated admission must reject without entering")
    assert admission.active == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert admission.active == 0
    with admission.slot():
        assert admission.active == 1
    assert admission.active == 0


def test_admission_releases_on_exception_and_enforces_bound() -> None:
    admission = InferenceAdmission(2)
    with admission.slot():
        with pytest.raises(RuntimeError), admission.slot():
            assert admission.active == 2
            with pytest.raises(CapacityFull), admission.slot():
                pytest.fail("third inference must be rejected")
            raise RuntimeError("inference failed")
        assert admission.active == 1
    assert admission.active == 0
