"""Conservative exact-response cache, owned by one asyncio event loop.

All state changes are synchronous and contain no awaits. This cache is neither
thread-safe nor shared between workers. Keys are ephemeral keyed digests; the
cache retains response bytes but does not retain prompt text as keys.
"""

import hashlib
import hmac
import json
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass

from llm_gateway.config import SemanticFamily, Settings
from llm_gateway.schemas import ChatRequest

_POLICY_VERSION = "exact-v1"


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    bypasses: int = 0
    stores: int = 0
    evictions: int = 0
    expirations: int = 0
    upstream_calls: int = 0


@dataclass(frozen=True)
class SemanticTag:
    partition: bytes
    vector: bytes

    @property
    def size(self) -> int:
        return len(self.partition) + len(self.vector)


@dataclass(frozen=True)
class _Entry:
    payload: bytes
    expires_at: float
    semantic: SemanticTag | None = None

    @property
    def size(self) -> int:
        return len(self.payload) + (self.semantic.size if self.semantic else 0)


class ExactResponseCache:
    """Bounded TTL/LRU storage for explicitly approved deterministic requests.

    Configuration is a startup contract. Fingerprints nevertheless read its
    current values so a changed context or route cannot reuse an earlier key.
    The byte budget counts serialized keys and values, not Python/RSS overhead;
    max_entries separately bounds per-entry metadata.
    """

    def __init__(self, settings: Settings, *, clock: Callable[[], float] = time.monotonic):
        self._settings = settings
        self._clock = clock
        self._secret = secrets.token_bytes(32)
        self._entries: OrderedDict[bytes, _Entry] = OrderedDict()
        self._bytes_used = 0
        self.stats = CacheStats()

    def family_for(self, request: ChatRequest) -> SemanticFamily | None:
        if len(request.messages) != 1 or request.messages[0].role != "user":
            return None
        return next(
            (
                family
                for family in self._settings.config.semantic_cache.families
                if request.messages[0].content in family.prompts
            ),
            None,
        )

    def key_for(
        self, request: ChatRequest, client_id: str, opt_in: bool, *, semantic: bool = False
    ) -> bytes | None:
        """Return an isolated exact key, or count a bypass and refuse caching."""
        config = self._settings.config
        policy = config.exact_cache
        route = config.models.get(request.model)
        backend = config.backends.get(route.backend) if route is not None else None
        if (
            not policy.enabled
            or not opt_in
            or route is None
            or backend is None
            or not route.revision
            or not backend.revision
            or len(request.messages) != 1
            or request.messages[0].role != "user"
            or (
                request.messages[0].content not in policy.approved_prompts
                and not (semantic and config.semantic_cache.enabled and self.family_for(request))
            )
            or request.temperature != 0.0
            or request.top_p != 1.0
            or request.seed is None
            or request.stream
        ):
            self.stats.bypasses += 1
            return None

        return self._fingerprint(request.model_dump(mode="json"), client_id, route.backend)

    def semantic_partition(
        self, request: ChatRequest, client_id: str, embedding_fingerprint: str
    ) -> bytes:
        """Replace only reviewed wording; every other request/config field remains."""
        family = self.family_for(request)
        if family is None:
            raise ValueError("semantic partition requires reviewed membership")
        values = request.model_dump(mode="json")
        values["messages"] = [{"role": "user", "reviewed_family": family.model_dump()}]
        values["embedding_fingerprint"] = embedding_fingerprint
        values["semantic_policy"] = "reviewed-equivalence-v1"
        route = self._settings.config.models[request.model]
        return self._fingerprint(values, client_id, route.backend)

    def _fingerprint(self, values: dict, client_id: str, backend_name: str) -> bytes:
        config = self._settings.config
        token = self._settings.backend_tokens.get(backend_name) or ""
        canonical = json.dumps(
            {
                "policy_version": _POLICY_VERSION,
                "client_id": client_id,
                "request": values,
                "config": config.model_dump(mode="json"),
                "backend_credential_sha256": hashlib.sha256(token.encode()).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hmac.digest(self._secret, canonical, "sha256")

    def _expire(self, now: float) -> None:
        # LRU access order differs from expiry order. Scan the bounded metadata
        # rather than introducing an unbounded heap or a background sweeper.
        for key, entry in list(self._entries.items()):
            if entry.expires_at <= now:
                del self._entries[key]
                self._bytes_used -= len(key) + entry.size
                self.stats.expirations += 1

    def get(self, key: bytes) -> bytes | None:
        """Read fresh bytes and refresh LRU order, without extending the TTL."""
        self._expire(self._clock())
        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        self._entries.move_to_end(key)
        self.stats.hits += 1
        return entry.payload

    def semantic_candidates(self, partition: bytes) -> list[tuple[bytes, bytes]]:
        self._expire(self._clock())
        return [
            (key, entry.semantic.vector)
            for key, entry in self._entries.items()
            if entry.semantic is not None and entry.semantic.partition == partition
        ]

    def semantic_get(self, key: bytes, partition: bytes) -> bytes | None:
        # Recheck freshness and compatibility after off-loop embedding/search.
        self._expire(self._clock())
        entry = self._entries.get(key)
        if entry is None or entry.semantic is None or entry.semantic.partition != partition:
            return None
        self._entries.move_to_end(key)
        return entry.payload

    def put(self, key: bytes, payload: bytes, *, semantic: SemanticTag | None = None) -> bool:
        """Store a fresh entry; duplicates and oversized responses are skipped."""
        if not isinstance(key, bytes) or not isinstance(payload, bytes):
            raise TypeError("cache keys and response payloads must be immutable bytes")
        if semantic is not None and (
            not isinstance(semantic.partition, bytes)
            or not isinstance(semantic.vector, bytes)
            or len(semantic.partition) != 32
            or len(semantic.vector) != 384 * 4
        ):
            raise ValueError("semantic metadata must be an opaque partition and float32 vector")
        now = self._clock()
        self._expire(now)
        if key in self._entries:
            return False

        policy = self._settings.config.exact_cache
        size = len(key) + len(payload) + (semantic.size if semantic else 0)
        if size > policy.max_bytes:
            return False
        if semantic is not None:
            tagged = [key for key, entry in self._entries.items() if entry.semantic is not None]
            if len(tagged) >= self._settings.config.semantic_cache.max_entries:
                previous_key = tagged[0]
                previous = self._entries.pop(previous_key)
                self._bytes_used -= len(previous_key) + previous.size
                self.stats.evictions += 1
        while (
            len(self._entries) >= policy.max_entries or self._bytes_used + size > policy.max_bytes
        ):
            previous_key, previous = self._entries.popitem(last=False)
            self._bytes_used -= len(previous_key) + previous.size
            self.stats.evictions += 1
        self._entries[key] = _Entry(
            payload=payload, expires_at=now + policy.ttl_seconds, semantic=semantic
        )
        self._bytes_used += size
        self.stats.stores += 1
        return True

    def snapshot(self) -> dict[str, int]:
        """Return counters and current occupancy, without prompt/response data."""
        self._expire(self._clock())
        return asdict(self.stats) | {
            "entries": len(self._entries),
            "bytes_used": self._bytes_used,
        }
