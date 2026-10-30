"""Fixed-cardinality, process-local metrics owned by one asyncio event loop.

Mutations contain no awaits. This module performs no I/O and provides no thread
or multi-worker consistency. The caller must pair each start with exactly one
finish in its finally block. Observed HTTP status does not establish completed
delivery: final ASGI body acceptance and outcome=completed are both required.
"""

import math
import sys
from bisect import bisect_left
from itertools import accumulate

LATENCY_LABELS = (
    "gateway_to_start",
    "request_lifecycle",
    "adapter",
    "exact_lookup",
    "semantic",
    "gateway_residual",
)
BUCKET_BOUNDS_MS = (
    0.1,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
    30000.0,
    60000.0,
    300000.0,
)
OUTCOMES = (
    "completed",
    "cancelled",
    "disconnected",
    "send_error",
    "unhandled_error",
    "aborted",
    "other",
)
CACHE_DISPOSITIONS = ("HIT", "SEMANTIC_HIT", "MISS", "BYPASS", "EARLY")


class _Histogram:
    """Store only aggregate values and exclusive fixed buckets, never samples."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.maximum = None
        self.sum_saturated = False
        self.buckets = [0] * (len(BUCKET_BOUNDS_MS) + 1)

    def observe(self, value: float) -> None:
        self.count += 1
        total = self.total + value
        if not math.isfinite(total):
            total = sys.float_info.max
            self.sum_saturated = True
        self.total = total
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.buckets[bisect_left(BUCKET_BOUNDS_MS, value)] += 1

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "sum": self.total,
            "max": self.maximum,
            "sum_saturated": self.sum_saturated,
            "buckets": [
                {"upper_bound_ms": bound, "count": count}
                for bound, count in zip(
                    (*BUCKET_BOUNDS_MS, None), accumulate(self.buckets), strict=True
                )
            ],
        }


def _rate(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


class Metrics:
    def __init__(self) -> None:
        self._counts = dict.fromkeys(
            (
                "requests_started",
                "requests_finished",
                "responses_completed",
                "successful_responses",
                "http_error_responses",
                "aborted_requests",
                "ignored_duration_samples",
                "ignored_cache_dispositions",
            ),
            0,
        )
        self._in_flight = self._peak_in_flight = 0
        self._status_families = dict.fromkeys(
            ("1xx", "2xx", "3xx", "4xx", "5xx", "none", "other"), 0
        )
        self._exact_status = dict.fromkeys(("429", "503", "504"), 0)
        self._outcomes = dict.fromkeys(OUTCOMES, 0)
        self._cache = dict.fromkeys(CACHE_DISPOSITIONS, 0)
        self._histograms = {label: _Histogram() for label in LATENCY_LABELS}
        self._usage = dict.fromkeys(
            (
                "responses_with_reported_usage",
                "reported_prompt_tokens",
                "reported_completion_tokens",
                "reported_total_tokens",
                "ignored_usage_reports",
            ),
            0,
        )

    def start(self) -> None:
        self._counts["requests_started"] += 1
        self._in_flight += 1
        self._peak_in_flight = max(self._peak_in_flight, self._in_flight)

    def finish(
        self,
        *,
        status: int | None,
        complete: bool,
        outcome: str,
        durations_ms: dict[str, float],
        cache_status: str | None,
    ) -> None:
        if self._in_flight <= 0:
            raise RuntimeError("Metrics.finish requires a matching unfinished start")
        self._in_flight -= 1
        self._counts["requests_finished"] += 1
        family = "none" if status is None else "other"
        if isinstance(status, int) and not isinstance(status, bool):
            if 100 <= status <= 599:
                family = f"{status // 100}xx"
            if status in (429, 503, 504):
                self._exact_status[str(status)] += 1
        self._status_families[family] += 1
        outcome = outcome if isinstance(outcome, str) and outcome in self._outcomes else "other"
        self._outcomes[outcome] += 1
        completed = complete is True and outcome == "completed"
        if completed:
            self._counts["responses_completed"] += 1
            if family == "2xx":
                self._counts["successful_responses"] += 1
            elif family in {"4xx", "5xx"}:
                self._counts["http_error_responses"] += 1
            if cache_status is None:
                self._cache["EARLY"] += 1
            elif isinstance(cache_status, str) and cache_status in self._cache:
                self._cache[cache_status] += 1
            else:
                self._counts["ignored_cache_dispositions"] += 1
        else:
            self._counts["aborted_requests"] += 1
        for label, value in durations_ms.items():
            histogram = self._histograms.get(label)
            if histogram is None or isinstance(value, bool) or not isinstance(value, (float, int)):
                self._counts["ignored_duration_samples"] += 1
                continue
            try:
                value = float(value)
            except OverflowError:
                self._counts["ignored_duration_samples"] += 1
                continue
            if not math.isfinite(value) or value < 0:
                self._counts["ignored_duration_samples"] += 1
                continue
            histogram.observe(value)

    def record_upstream_usage(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int
    ) -> None:
        """Record only new provider-reported inference usage, never cached usage.

        Callers establish provenance. Estimates and historical cached response
        fields must not call this method. Invalid input never changes counters.
        """
        values = (prompt_tokens, completion_tokens, total_tokens)
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1
                for value in values
            )
            or prompt_tokens + completion_tokens != total_tokens
        ):
            raise ValueError("Reported usage must contain consistent nonnegative integer counts")
        self._usage["responses_with_reported_usage"] += 1
        self._usage["reported_prompt_tokens"] += prompt_tokens
        self._usage["reported_completion_tokens"] += completion_tokens
        self._usage["reported_total_tokens"] += total_tokens

    def ignore_upstream_usage(self) -> None:
        """Count unusable telemetry without changing an otherwise valid HTTP response."""
        self._usage["ignored_usage_reports"] += 1

    def snapshot(self) -> dict:
        counts = self._counts.copy()
        eligible = sum(self._cache[label] for label in ("HIT", "SEMANTIC_HIT", "MISS"))
        return {
            "schema_version": 1,
            "scope": "single_process_single_event_loop",
            "counts": counts,
            "gauges": {
                "requests_in_flight": self._in_flight,
                "peak_requests_in_flight": self._peak_in_flight,
            },
            "status": {
                "by_family": self._status_families.copy(),
                "exact": self._exact_status.copy(),
            },
            "outcomes": self._outcomes.copy(),
            "cache": {
                "completed_dispositions": self._cache.copy(),
                "eligible_completed_requests": eligible,
            },
            "rates": {
                "success_per_finished": _rate(
                    counts["successful_responses"], counts["requests_finished"]
                ),
                "http_errors_per_completed": _rate(
                    counts["http_error_responses"], counts["responses_completed"]
                ),
                "exact_hit_per_eligible_completed": _rate(self._cache["HIT"], eligible),
                "semantic_hit_per_eligible_completed": _rate(self._cache["SEMANTIC_HIT"], eligible),
                "any_hit_per_eligible_completed": _rate(
                    self._cache["HIT"] + self._cache["SEMANTIC_HIT"], eligible
                ),
            },
            "latency_ms": {
                label: histogram.snapshot() for label, histogram in self._histograms.items()
            },
            "usage": self._usage.copy(),
        }
