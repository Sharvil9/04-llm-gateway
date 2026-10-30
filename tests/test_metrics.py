"""Fixed-cardinality counts, completion semantics, and histogram boundaries."""

import asyncio
import json
import sys

import pytest

from llm_gateway.metrics import CACHE_DISPOSITIONS, LATENCY_LABELS, OUTCOMES, Metrics


def observe(metrics, *, status=200, complete=True, outcome="completed", durations=None, cache=None):
    metrics.start()
    metrics.finish(
        status=status,
        complete=complete,
        outcome=outcome,
        durations_ms=durations or {},
        cache_status=cache,
    )


async def test_concurrent_task_lifecycles_preserve_counts_and_in_flight_peak():
    metrics = Metrics()
    all_started, release = asyncio.Event(), asyncio.Event()
    arrived = 0
    statuses = [200] * 60 + [429] * 20 + [503] * 10 + [504] * 10

    async def request(status):
        nonlocal arrived
        metrics.start()
        arrived += 1
        if arrived == len(statuses):
            all_started.set()
        await release.wait()
        metrics.finish(
            status=status,
            complete=True,
            outcome="completed",
            durations_ms={"request_lifecycle": 5.0},
            cache_status="BYPASS",
        )

    tasks = [asyncio.create_task(request(status)) for status in statuses]
    try:
        await asyncio.wait_for(all_started.wait(), timeout=1)
        live = metrics.snapshot()
        assert live["counts"]["requests_started"] == live["gauges"]["requests_in_flight"] == 100
        assert live["gauges"]["peak_requests_in_flight"] == 100
        release.set()
        await asyncio.gather(*tasks)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
    snapshot = metrics.snapshot()
    assert (
        snapshot["counts"]["requests_finished"] == snapshot["counts"]["responses_completed"] == 100
    )
    assert snapshot["counts"]["successful_responses"] == 60
    assert snapshot["counts"]["http_error_responses"] == 40
    assert snapshot["counts"]["aborted_requests"] == 0
    assert snapshot["gauges"]["requests_in_flight"] == 0
    assert snapshot["status"]["by_family"]["2xx"] == 60
    assert snapshot["status"]["by_family"]["4xx"] == snapshot["status"]["by_family"]["5xx"] == 20
    assert snapshot["status"]["exact"] == {"429": 20, "503": 10, "504": 10}
    assert snapshot["latency_ms"]["request_lifecycle"]["count"] == 100


@pytest.mark.parametrize(
    ("status", "complete", "outcome", "family"),
    [
        (200, False, "cancelled", "2xx"),
        (200, True, "send_error", "2xx"),
        (200, False, "unhandled_error", "2xx"),
        (499, True, "disconnected", "4xx"),
        (None, False, "unhandled_error", "none"),
        (200, True, "private-unrecognized-outcome", "2xx"),
    ],
)
def test_observed_status_does_not_make_aborted_request_successful(
    status, complete, outcome, family
):
    metrics = Metrics()
    observe(
        metrics,
        status=status,
        complete=complete,
        outcome=outcome,
        cache="HIT",
        durations={"request_lifecycle": 10.0},
    )
    snapshot = metrics.snapshot()
    assert snapshot["counts"]["aborted_requests"] == 1
    assert snapshot["counts"]["responses_completed"] == 0
    assert (
        snapshot["counts"]["successful_responses"]
        == snapshot["counts"]["http_error_responses"]
        == 0
    )
    assert snapshot["status"]["by_family"][family] == 1
    assert sum(snapshot["cache"]["completed_dispositions"].values()) == 0
    assert snapshot["latency_ms"]["request_lifecycle"]["count"] == 1
    assert set(snapshot["outcomes"]) == set(OUTCOMES)
    assert "private-unrecognized-outcome" not in json.dumps(snapshot)


def test_histograms_have_inclusive_cumulative_fixed_buckets_and_no_raw_samples():
    metrics = Metrics()
    samples = [0.0, 0.1, 0.5, 1.0, 5.0, 300000.0, 300001.0]
    for value in samples:
        observe(metrics, durations={"adapter": value})
    histogram = metrics.snapshot()["latency_ms"]["adapter"]
    assert histogram["count"] == len(samples)
    assert histogram["sum"] == pytest.approx(sum(samples))
    assert histogram["max"] == max(samples)
    buckets = {bucket["upper_bound_ms"]: bucket["count"] for bucket in histogram["buckets"]}
    assert buckets[0.1] == 2
    assert buckets[0.5] == 3
    assert buckets[1.0] == 4
    assert buckets[2.5] == 4
    assert buckets[5.0] == 5
    assert buckets[300000.0] == 6
    assert buckets[None] == len(samples)
    assert [bucket["count"] for bucket in histogram["buckets"]] == sorted(buckets.values())
    assert set(histogram) == {"count", "sum", "max", "sum_saturated", "buckets"}
    empty = metrics.snapshot()["latency_ms"]["semantic"]
    assert empty["count"] == empty["sum"] == 0 and empty["max"] is None
    json.dumps(metrics.snapshot(), allow_nan=False)


def test_unknown_labels_and_invalid_samples_cannot_expand_or_leak_metric_dimensions():
    metrics = Metrics()
    durations = {f"private-client-prompt-{number}": float(number) for number in range(1000)}
    durations.update(
        adapter=-1,
        semantic=float("nan"),
        gateway_residual=float("inf"),
        request_lifecycle=True,
        gateway_to_start="25",
        exact_lookup=0.5,
    )
    observe(metrics, durations=durations, cache="private-api-key-cache-label")
    snapshot = metrics.snapshot()
    assert set(snapshot["latency_ms"]) == set(LATENCY_LABELS)
    assert snapshot["counts"]["ignored_duration_samples"] == 1005
    assert snapshot["counts"]["ignored_cache_dispositions"] == 1
    assert snapshot["latency_ms"]["exact_lookup"]["count"] == 1
    assert sum(histogram["count"] for histogram in snapshot["latency_ms"].values()) == 1
    assert set(snapshot["cache"]["completed_dispositions"]) == set(CACHE_DISPOSITIONS)
    serialized = json.dumps(snapshot, allow_nan=False)
    assert "private-client-prompt" not in serialized
    assert "private-api-key" not in serialized


def test_extreme_finite_samples_cannot_overflow_json_histogram_sums():
    metrics = Metrics()
    for _ in range(2):
        observe(metrics, durations={"adapter": sys.float_info.max})
    histogram = metrics.snapshot()["latency_ms"]["adapter"]
    assert histogram["sum"] == histogram["max"] == sys.float_info.max
    assert histogram["sum_saturated"] is True
    assert histogram["buckets"][-1] == {"upper_bound_ms": None, "count": 2}
    json.dumps(metrics.snapshot(), allow_nan=False)


def test_cache_rates_count_only_completed_dispositions_and_expose_denominators():
    metrics = Metrics()
    assert all(rate["value"] is None for rate in metrics.snapshot()["rates"].values())
    for status, disposition in [
        (200, "HIT"),
        (200, "HIT"),
        (200, "SEMANTIC_HIT"),
        (503, "MISS"),
        (200, "BYPASS"),
        (200, "BYPASS"),
        (401, None),
    ]:
        observe(metrics, status=status, cache=disposition)
    observe(metrics, status=200, complete=False, outcome="cancelled", cache="HIT")
    snapshot = metrics.snapshot()
    assert snapshot["cache"] == {
        "completed_dispositions": {
            "HIT": 2,
            "SEMANTIC_HIT": 1,
            "MISS": 1,
            "BYPASS": 2,
            "EARLY": 1,
        },
        "eligible_completed_requests": 4,
    }
    assert snapshot["rates"]["exact_hit_per_eligible_completed"] == {
        "numerator": 2,
        "denominator": 4,
        "value": 0.5,
    }
    assert snapshot["rates"]["semantic_hit_per_eligible_completed"] == {
        "numerator": 1,
        "denominator": 4,
        "value": 0.25,
    }
    assert snapshot["rates"]["any_hit_per_eligible_completed"] == {
        "numerator": 3,
        "denominator": 4,
        "value": 0.75,
    }
    assert snapshot["rates"]["success_per_finished"]["value"] == 5 / 8
    assert snapshot["rates"]["http_errors_per_completed"]["value"] == 2 / 7


def test_only_explicitly_recorded_new_provider_usage_changes_usage_totals():
    metrics = Metrics()
    metrics.record_upstream_usage(4, 3, 7)
    metrics.record_upstream_usage(0, 0, 0)
    before = metrics.snapshot()["usage"]
    assert before == {
        "responses_with_reported_usage": 2,
        "reported_prompt_tokens": 4,
        "reported_completion_tokens": 3,
        "reported_total_tokens": 7,
        "ignored_usage_reports": 0,
    }
    observe(metrics, cache="HIT")
    assert metrics.snapshot()["usage"] == before


def test_ignored_usage_reports_are_explicit_and_do_not_change_reported_totals():
    metrics = Metrics()
    assert metrics.snapshot()["usage"]["ignored_usage_reports"] == 0
    metrics.record_upstream_usage(2, 1, 3)
    before = metrics.snapshot()["usage"]
    with pytest.raises(ValueError):
        metrics.record_upstream_usage(2**63, 1, 2**63 + 1)
    assert metrics.snapshot()["usage"] == before
    metrics.ignore_upstream_usage()
    metrics.ignore_upstream_usage()
    assert metrics.snapshot()["usage"] == before | {"ignored_usage_reports": 2}


@pytest.mark.parametrize(
    "values", [(True, 0, 1), (-1, 1, 0), (1.0, 2, 3), (1, 2, 4), (2**63, 0, 2**63)]
)
def test_invalid_provider_usage_is_rejected_atomically(values):
    metrics = Metrics()
    metrics.record_upstream_usage(2, 1, 3)
    before = metrics.snapshot()
    with pytest.raises(ValueError):
        metrics.record_upstream_usage(*values)
    assert metrics.snapshot() == before


def test_snapshot_is_detached_and_arbitrary_statuses_do_not_create_labels():
    metrics = Metrics()
    for status in range(1000, 2000):
        observe(metrics, status=status, complete=False, outcome="aborted")
    snapshot = metrics.snapshot()
    assert snapshot["status"]["by_family"] == {
        "1xx": 0,
        "2xx": 0,
        "3xx": 0,
        "4xx": 0,
        "5xx": 0,
        "none": 0,
        "other": 1000,
    }
    assert snapshot["status"]["exact"] == {"429": 0, "503": 0, "504": 0}
    snapshot["counts"]["requests_started"] = -1
    snapshot["status"]["by_family"]["other"] = -1
    snapshot["latency_ms"]["adapter"]["buckets"][0]["count"] = -1
    fresh = metrics.snapshot()
    assert fresh["counts"]["requests_started"] == 1000
    assert fresh["status"]["by_family"]["other"] == 1000
    assert fresh["latency_ms"]["adapter"]["buckets"][0]["count"] == 0
    assert fresh["scope"] == "single_process_single_event_loop"


def test_finish_underflow_is_rejected_without_changing_state():
    metrics = Metrics()
    before = metrics.snapshot()
    with pytest.raises(RuntimeError, match="matching unfinished start"):
        metrics.finish(
            status=None,
            complete=False,
            outcome="unhandled_error",
            durations_ms={},
            cache_status=None,
        )
    assert metrics.snapshot() == before
    observe(metrics)
    before = metrics.snapshot()
    with pytest.raises(RuntimeError):
        metrics.finish(
            status=200, complete=True, outcome="completed", durations_ms={}, cache_status=None
        )
    assert metrics.snapshot() == before
