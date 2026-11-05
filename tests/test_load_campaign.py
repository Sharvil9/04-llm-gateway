"""Validate campaign scope/accounting without running load or inference."""

import pytest

from devtools.load_campaign import aggregate, workloads


def test_primary_is_repeated_randomized_and_keeps_reviewed_semantic_workload():
    name, options = workloads("primary")[0]
    assert name == "primary"
    assert options[options.index("--repeats") + 1] == "3"
    assert "--randomize-order" in options
    assert options[options.index("--concurrency") + 1 : options.index("--scenarios")] == [
        "1",
        "2",
        "4",
        "8",
        "16",
    ]
    assert "semantic" in options


def test_rate_and_sustained_work_are_bounded_and_real_model_scope_is_small():
    for stage in ("primary", "ramp", "sustained", "gemma"):
        for _, options in workloads(stage):
            count = int(options[options.index("--requests") + 1])
            assert 1 <= count <= 10000
            if stage == "gemma":
                assert count <= 30
    assert [name for name, _ in workloads("ramp")] == ["arrival-100", "arrival-300", "arrival-600"]
    with pytest.raises(ValueError):
        workloads("unknown")


def test_aggregation_separates_rejections_from_success_and_does_not_pool_tails():
    points = [
        {
            "scenario": "uncached",
            "concurrency": 16,
            "successful_rps": rate,
            "latency_successful": {"p50_ms": median},
            "status_counts": {"200": 50, "503": 50},
            "independent_upstream_calls": 50,
        }
        for rate, median in ((100.0, 10.0), (200.0, 30.0), (150.0, 20.0))
    ]
    result = aggregate(points)[0]
    assert result["successful_rps_median"] == 150
    assert result["successful_rps_min"] == 100
    assert result["successful_rps_max"] == 200
    assert result["http_503_total"] == 150
    assert result["actual_upstream_calls_total"] == 150
    assert result["successful_p50_ms_median"] == 20
