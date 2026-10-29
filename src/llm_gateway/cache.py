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

from llm_gateway.config import Settings
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
class _Entry:
    payload: bytes
    expires_at: float


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

    def key_for(self, request: ChatRequest, client_id: str, opt_in: bool) -> bytes | None:
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
            or request.messages[0].content not in policy.approved_prompts
            or request.temperature != 0.0
            or request.top_p != 1.0
            or request.seed is None
            or request.stream
        ):
            self.stats.bypasses += 1
            return None

        token = self._settings.backend_tokens.get(route.backend) or ""
        canonical = json.dumps(
            {
                "policy_version": _POLICY_VERSION,
                "client_id": client_id,
                "request": request.model_dump(mode="json"),
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
                self._bytes_used -= len(key) + len(entry.payload)
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

    def put(self, key: bytes, payload: bytes) -> bool:
        """Store a fresh entry; duplicates and oversized responses are skipped."""
        if not isinstance(key, bytes) or not isinstance(payload, bytes):
            raise TypeError("cache keys and response payloads must be immutable bytes")
        now = self._clock()
        self._expire(now)
        if key in self._entries:
            return False

        policy = self._settings.config.exact_cache
        size = len(key) + len(payload)
        if size > policy.max_bytes:
            return False
        while (
            len(self._entries) >= policy.max_entries or self._bytes_used + size > policy.max_bytes
        ):
            previous_key, previous = self._entries.popitem(last=False)
            self._bytes_used -= len(previous_key) + len(previous.payload)
            self.stats.evictions += 1
        self._entries[key] = _Entry(payload=payload, expires_at=now + policy.ttl_seconds)
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
