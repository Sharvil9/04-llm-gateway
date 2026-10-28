"""Process-local traffic controls for one asyncio event loop.

Check/update sections contain no awaits: another task cannot interrupt them on
the owning event loop. These objects are not thread-safe and do not coordinate
multiple workers. Never share them between event loops or threads.
"""

import math
import time
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_gateway.config import RateLimitConfig


@dataclass
class _Bucket:
    tokens: float
    updated_at: float
    capacity: int
    refill_per_second: float


class RequestLimiter:
    """A token bucket for each configured stable client ID.

    Buckets start full. Denials retain their available fractional tokens, and
    elapsed time refills each bucket up to its burst capacity. Unknown IDs raise
    KeyError; requests cannot allocate unbounded per-client state.
    """

    def __init__(
        self,
        client_ids: Collection[str],
        default: "RateLimitConfig",
        overrides: dict[str, "RateLimitConfig"],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        now = clock()
        self._buckets: dict[str, _Bucket] = {}
        for client_id in client_ids:
            policy = overrides.get(client_id, default)
            self._buckets[client_id] = _Bucket(
                tokens=float(policy.burst),
                updated_at=now,
                capacity=policy.burst,
                refill_per_second=policy.requests_per_second,
            )

    def consume(self, client_id: str) -> int | None:
        """Consume one request or return a positive integer Retry-After.

        The refill timestamp never moves backwards, even with an injected clock.
        Retry-After describes the current deficit; other requests may consume
        subsequently replenished tokens before a rejected caller retries.
        """
        bucket = self._buckets[client_id]
        now = max(bucket.updated_at, self._clock())
        bucket.tokens = min(
            bucket.capacity,
            bucket.tokens + (now - bucket.updated_at) * bucket.refill_per_second,
        )
        bucket.updated_at = now
        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return None
        return max(1, math.ceil((1 - bucket.tokens) / bucket.refill_per_second))


class CapacityFull(Exception):
    """All inference slots are occupied; admission has no waiting queue."""


class InferenceAdmission:
    """Immediate admission with cancellation-safe release on the owning loop."""

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("inference limit must be a positive integer")
        self._limit = limit
        self._active = 0

    @property
    def active(self) -> int:
        return self._active

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Reserve one slot, rejecting immediately when capacity is full."""
        if self._active >= self._limit:
            raise CapacityFull
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1
