"""Closed-vocabulary semantic retrieval; reviewed equivalence authorizes reuse."""

import asyncio
from dataclasses import asdict, dataclass

from llm_gateway.cache import ExactResponseCache, SemanticTag
from llm_gateway.config import SemanticCacheConfig
from llm_gateway.schemas import ChatRequest


@dataclass
class SemanticStats:
    hits: int = 0
    misses: int = 0
    busy: int = 0
    failures: int = 0


class SemanticCache:
    def __init__(self, cache: ExactResponseCache, embedder, policy: SemanticCacheConfig):
        self.cache = cache
        self.embedder = embedder
        self.policy = policy
        self.stats = SemanticStats()

    async def lookup(
        self, request: ChatRequest, client_id: str
    ) -> tuple[bytes | None, SemanticTag | None]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.policy.timeout_seconds
        partition = self.cache.semantic_partition(request, client_id, self.embedder.fingerprint)
        candidates = self.cache.semantic_candidates(partition)
        try:
            async with asyncio.timeout_at(deadline):
                match = await self.embedder.match(
                    request.messages[0].content,
                    candidates,
                    self.policy.threshold,
                )
            # Cooperative timers cannot preempt a blocked loop. Reject a late
            # result even if native completion won the scheduling race.
            if loop.time() >= deadline:
                raise TimeoutError
        except Exception:
            # Optional caching cannot make inference unavailable. Do not expose
            # model errors/input text. CancelledError remains propagated.
            self.stats.failures += 1
            self.stats.misses += 1
            return None, None
        if match is None:
            self.stats.busy += 1
            self.stats.misses += 1
            return None, None
        vector, candidate, _score = match
        try:
            current = self.cache.semantic_partition(request, client_id, self.embedder.fingerprint)
        except ValueError:
            current = None
        if partition != current:
            self.stats.misses += 1
            return None, None
        if candidate is not None:
            cached = self.cache.semantic_get(candidate, partition)
            if cached is not None:
                self.stats.hits += 1
                return cached, None
        self.stats.misses += 1
        return None, SemanticTag(partition, vector)

    def snapshot(self) -> dict[str, int]:
        return asdict(self.stats)
