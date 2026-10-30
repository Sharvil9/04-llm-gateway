import asyncio
import hashlib
import json
import subprocess
import threading
from collections import Counter
from types import SimpleNamespace

import httpx
import pytest

import devtools.benchmark as benchmark
from devtools.benchmark import (
    make_schedule,
    metrics_delta,
    quantiles,
    run_load,
    verify_wire_replays,
)
from llm_gateway.metrics import Metrics


def completion(city: str = "Paris") -> dict:
    return {
        "id": "chatcmpl-benchmark-test",
        "object": "chat.completion",
        "created": 0,
        "model": "demo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": city},
                "finish_reason": "stop",
            }
        ],
    }


def payloads() -> dict[str, bytes]:
    return {
        kind: json.dumps({"probe_case": kind}).encode()
        for kind in ("direct", "uncached", "exact", "semantic")
    }


@pytest.mark.parametrize(
    ("count", "p50", "p95", "p99"),
    [(1, 1, None, None), (99, 50, None, None), (100, 50, 95, None), (1000, 500, 950, 990)],
)
def test_quantiles_use_nearest_rank_and_sample_size_gates(count, p50, p95, p99) -> None:
    observed = quantiles(list(reversed(range(1, count + 1))))
    assert observed["count"] == count
    assert observed["p50_ms"] == p50
    assert observed["p95_ms"] == p95
    assert observed["p99_ms"] == p99
    if count < 1000:
        assert observed["caveat"]


def test_empty_quantiles_have_no_invented_latency() -> None:
    observed = quantiles([])
    assert observed["count"] == 0
    assert observed["p50_ms"] is None
    assert observed["p95_ms"] is None
    assert observed["p99_ms"] is None


def test_mixed_schedule_is_reproducible_with_bounded_block_proportions() -> None:
    schedule = make_schedule("mixed", 1000, seed=42)
    assert schedule == make_schedule("mixed", 1000, seed=42)
    assert schedule != make_schedule("mixed", 1000, seed=43)
    assert Counter(schedule) == {"exact": 400, "semantic": 400, "uncached": 200}
    for start in range(0, len(schedule), 5):
        assert Counter(schedule[start : start + 5]) == {"exact": 2, "semantic": 2, "uncached": 1}
    assert len(make_schedule("mixed", 7)) == 7
    assert make_schedule("exact", 4) == ["exact"] * 4
    assert make_schedule("semantic", 4) == ["semantic"] * 4
    assert make_schedule("uncached", 4) == ["uncached"] * 4
    assert make_schedule("direct", 4) == ["direct"] * 4


async def test_load_uses_fixed_workers_and_completes_each_index_once() -> None:
    active = peak = 0
    worker_ids = set()
    requests = []
    wire_body = json.dumps(completion(), separators=(",", ":")).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        worker_ids.add(id(asyncio.current_task()))
        requests.append(request)
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.001)
            return httpx.Response(
                200,
                content=wire_body,
                headers={"X-Gateway-Cache": "HIT", "Server-Timing": "gateway;dur=0.2"},
            )
        finally:
            active -= 1

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_load(
            client,
            "http://gateway.invalid/v1/chat/completions",
            "unit-test-credential",
            payloads(),
            ["exact"] * 100,
            4,
        )
    records = result["raw_records"]
    assert len(records) == len(requests) == 100
    assert sorted(record["index"] for record in records) == list(range(100))
    assert peak == 4
    assert len(worker_ids) <= 4
    assert active == 0
    assert all(request.headers["X-Gateway-Cache"] == "exact" for request in requests)
    assert all(record["status"] == 200 and record["response_correct"] for record in records)
    assert all(record["cache"] == "HIT" for record in records)
    assert all(
        record["response_sha256"] == hashlib.sha256(wire_body).hexdigest() for record in records
    )
    assert all(record["latency_ms"] >= 0 for record in records)
    assert result["attempted"] == result["completed"] == result["successful"] == 100
    assert result["http_200"] == 100
    assert result["rejected"] == result["transport_failures"] == result["incorrect_responses"] == 0
    assert result["successful_rps"] > 0


async def test_load_reports_transport_rejection_and_wrong_answer_without_secrets() -> None:
    secret = "sensitive-transport-credential"
    uncached = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal uncached
        kind = json.loads(request.content)["probe_case"]
        if kind == "direct":
            raise httpx.ConnectError(f"network diagnostic includes {secret}", request=request)
        if kind == "uncached":
            uncached += 1
            if uncached == 1:
                return httpx.Response(503, json={"error": {"code": "gateway_overloaded"}})
            return httpx.Response(200, json=completion("Tokyo"))
        return httpx.Response(
            200,
            json=completion(),
            headers={
                "X-Gateway-Cache": "HIT" if kind == "exact" else "SEMANTIC_HIT",
                "Server-Timing": "gateway;dur=1.2, semantic;dur=0.8",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_load(
            client,
            "http://gateway.invalid/v1/chat/completions",
            secret,
            payloads(),
            ["exact", "semantic", "uncached", "direct", "uncached"],
            1,
        )
    records = sorted(result["raw_records"], key=lambda row: row["index"])
    assert [row["status"] for row in records] == [200, 200, 503, None, 200]
    assert records[0]["cache"] == "HIT"
    assert records[1]["cache"] == "SEMANTIC_HIT"
    assert records[0]["response_correct"] and records[1]["response_correct"]
    assert not records[2]["response_correct"]
    assert records[3]["transport_error"]
    assert not records[4]["response_correct"]
    assert records[1]["timing_ms"]["semantic"] == 0.8
    assert secret not in json.dumps(result)
    assert result["attempted"] == 5
    assert result["completed"] == 4
    assert result["successful"] == 2
    assert result["http_200"] == 3
    assert result["rejected"] == result["transport_failures"] == result["incorrect_responses"] == 1
    assert result["other_http_errors"] == 0
    assert result["status_counts"] == {"200": 3, "503": 1, "None": 1}
    assert result["cache_dispositions"] == {"HIT": 1, "SEMANTIC_HIT": 1, "UNLABELLED": 3}


async def test_cancelling_load_stops_fixed_workers_and_releases_requests() -> None:
    active = 0
    entered = asyncio.Event()
    block = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active
        active += 1
        if active == 2:
            entered.set()
        try:
            await block.wait()
            return httpx.Response(200, json=completion())
        finally:
            active -= 1

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(
            run_load(
                client,
                "http://gateway.invalid/v1/chat/completions",
                "unit-test-credential",
                payloads(),
                ["exact"] * 100,
                2,
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert active == 0


def test_metrics_delta_preserves_gauges_maxima_and_recalculates_interval_rates() -> None:
    metrics = Metrics()

    def finish(status: int, cache: str, duration: float) -> None:
        metrics.finish(
            status=status,
            complete=True,
            outcome="completed",
            durations_ms={"request_lifecycle": duration},
            cache_status=cache,
        )

    metrics.start()
    finish(200, "MISS", 1000.0)
    metrics.record_upstream_usage(100, 10, 110)
    metrics.start()  # One request spans the beginning of the measured window.
    before = metrics.snapshot()
    before["semantic_lookup"] = None
    before["cache_storage"] = {
        "hits": 10,
        "misses": 20,
        "upstream_calls": 20,
        "entries": 2,
        "bytes_used": 2048,
    }

    metrics.start()
    finish(200, "HIT", 2.0)
    finish(200, "MISS", 5.0)
    metrics.record_upstream_usage(3, 2, 5)
    metrics.start()
    finish(503, "BYPASS", 8.0)
    metrics.start()
    metrics.start()  # Two requests remain active at the end of the window.
    after = metrics.snapshot()
    after["semantic_lookup"] = None
    after["cache_storage"] = {
        "hits": 11,
        "misses": 21,
        "upstream_calls": 21,
        "entries": 3,
        "bytes_used": 3072,
    }
    for _ in range(2):
        metrics.finish(
            status=None, complete=False, outcome="aborted", durations_ms={}, cache_status=None
        )

    observed = metrics_delta(before, after)
    assert observed["counts"]["requests_started"] == 4
    assert observed["counts"]["requests_finished"] == 3
    assert observed["counts"]["successful_responses"] == 2
    assert observed["status"]["exact"]["503"] == 1
    expected_usage = {
        "responses_with_reported_usage": 1,
        "reported_prompt_tokens": 3,
        "reported_completion_tokens": 2,
        "reported_total_tokens": 5,
    }
    assert all(observed["usage"][name] == count for name, count in expected_usage.items())
    histogram = observed["latency_ms"]["request_lifecycle"]
    assert histogram["count"] == 3
    assert histogram["sum"] == 15.0
    assert histogram["lifetime_max_after"] == 1000.0
    assert histogram["sum_saturated_after"] is False
    buckets = {bucket["upper_bound_ms"]: bucket["count"] for bucket in histogram["buckets"]}
    assert buckets[5.0] == 2
    assert buckets[10.0] == buckets[None] == 3
    assert observed["rates"]["success_per_finished"] == {
        "numerator": 2,
        "denominator": 3,
        "value": 2 / 3,
    }
    assert observed["rates"]["exact_hit_per_eligible_completed"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert observed["gauges_before"] == {"requests_in_flight": 1, "peak_requests_in_flight": 1}
    assert observed["gauges_after"] == {"requests_in_flight": 2, "peak_requests_in_flight": 2}
    assert observed["semantic_lookup"] == {}
    assert observed["cache_storage_counters"] == {"hits": 1, "misses": 1, "upstream_calls": 1}
    assert observed["cache_occupancy_after"] == {"entries": 3, "bytes_used": 3072}


def test_wire_replay_checks_controlled_sources_without_claiming_concurrent_semantic_lineage() -> (
    None
):
    source, another = "a" * 64, "b" * 64
    warming = {"primed_source_sha256": source}
    semantic = {"kind": "semantic", "cache": "SEMANTIC_HIT", "response_sha256": another}
    with pytest.raises(RuntimeError, match="differs"):
        verify_wire_replays({"raw_records": [semantic]}, warming, "semantic", 1)
    concurrent = verify_wire_replays({"raw_records": [semantic]}, warming, "semantic", 4)
    assert concurrent["cache_wire_replay_verified"] is None
    assert concurrent["checked_cache_wire_replays"] == 0
    assert concurrent["total_cache_hits"] == 1

    exact = {"kind": "exact", "cache": "HIT", "response_sha256": source}
    mixed = verify_wire_replays({"raw_records": [exact, semantic]}, warming, "mixed", 4)
    assert mixed["cache_wire_replay_verified"] is None
    assert mixed["checked_cache_wire_replays"] == 1
    assert mixed["total_cache_hits"] == 2
    exact["response_sha256"] = another
    with pytest.raises(RuntimeError, match="differs"):
        verify_wire_replays({"raw_records": [exact, semantic]}, warming, "mixed", 4)

    semantic["response_sha256"] = source
    controlled = verify_wire_replays({"raw_records": [semantic]}, warming, "semantic", 1)
    assert controlled["cache_wire_replay_verified"] is True
    assert controlled["checked_cache_wire_replays"] == controlled["total_cache_hits"] == 1


def test_missing_process_observations_are_unavailable_instead_of_zero_usage(monkeypatch) -> None:
    sampler = benchmark.ProcessSampler({"gateway": 111222})
    sampler.available = True
    monkeypatch.setattr(benchmark, "_process", lambda _pid: None)
    sampler.sample()
    report = sampler.summary(1.0)
    assert report["available"] is True  # Sampling capability, not successful PID observation.
    gateway = report["groups"]["gateway"]
    assert gateway["available"] is False
    assert gateway["observed_process_identities"] == 0
    assert gateway["cpu_core_seconds"] is None
    assert gateway["mean_cpu_cores"] is None
    assert gateway["sampled_peak_rss_bytes"] is None
    assert gateway["reason"]


@pytest.mark.parametrize(
    ("changes", "expected", "fails"),
    [
        ([(0.0, 5)], 5, False),
        ([(0.0, 5), (0.075, 6), (0.15, 7)], 7, False),
        ([(0.0, 5)], 6, True),
    ],
)
async def test_backend_log_count_settles_after_changes_or_times_out(
    monkeypatch, changes, expected, fails
) -> None:
    clock = {"now": 0.0}
    observed = []

    async def advance(delay: float) -> None:
        clock["now"] += delay

    def count_posts(_log):
        count = next(count for changed, count in reversed(changes) if changed <= clock["now"])
        observed.append((clock["now"], count))
        return count

    monkeypatch.setattr(benchmark, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    monkeypatch.setattr(benchmark, "asyncio", SimpleNamespace(sleep=advance))
    monkeypatch.setattr(benchmark, "log_posts", count_posts)
    if fails:
        with pytest.raises(RuntimeError, match="within1s"):
            await benchmark.backend_calls(None, "unused", object(), expected)
        assert 1.0 <= clock["now"] <= 1.05
    else:
        assert await benchmark.backend_calls(None, "unused", object(), expected) == expected
        last_change = max(
            instant
            for index, (instant, value) in enumerate(observed)
            if index == 0 or observed[index - 1][1] != value
        )
        assert clock["now"] - last_change >= 0.1
        assert clock["now"] < 1.0


def test_source_provenance_guard_rejects_changed_added_and_removed_files(monkeypatch) -> None:
    initial = {"devtools/benchmark.py": "a" * 64, "src/llm_gateway/app.py": "b" * 64}
    monkeypatch.setattr(benchmark, "source_manifest", lambda: dict(initial))
    assert benchmark.verify_source_manifest(initial) is None
    variants = (
        (initial | {"devtools/benchmark.py": "c" * 64}, "devtools/benchmark.py"),
        (initial | {"src/llm_gateway/new.py": "d" * 64}, "src/llm_gateway/new.py"),
        ({"devtools/benchmark.py": initial["devtools/benchmark.py"]}, "src/llm_gateway/app.py"),
    )
    for current, changed_path in variants:
        monkeypatch.setattr(benchmark, "source_manifest", lambda current=current: current)
        with pytest.raises(RuntimeError) as error:
            benchmark.verify_source_manifest(initial)
        assert changed_path in str(error.value)


def unavailable_proc_tasks(monkeypatch) -> None:
    class MissingTasks:
        def iterdir(self):
            raise FileNotFoundError

    monkeypatch.setattr(benchmark, "Path", lambda _path: MissingTasks())


def test_owned_descendant_discovery_falls_back_when_proc_children_are_absent(monkeypatch):
    unavailable_proc_tasks(monkeypatch)
    calls = []
    tree = {111222: [111223], 111223: [111224], 111224: []}

    def ps(command, **kwargs):
        assert command[:4] == ["ps", "-o", "pid=", "--ppid"]
        assert kwargs["timeout"] == 0.25 and kwargs["check"] is False
        parent = int(command[4])
        calls.append(parent)
        children = tree[parent]
        return SimpleNamespace(
            returncode=0 if children else 1,
            stdout="\n".join(str(pid) for pid in children),
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", ps)
    pids, methods, complete = benchmark._descendants(111222)
    assert pids == {111222, 111223, 111224}
    assert calls == [111222, 111223, 111224]  # Only the owned tree is traversed.
    assert methods == {"ps_ppid"} and complete is True


@pytest.mark.parametrize("failure", ["timeout", "stderr", "invalid_pid"])
def test_owned_child_discovery_errors_are_explicitly_incomplete(monkeypatch, failure):
    unavailable_proc_tasks(monkeypatch)

    def ps(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(
            returncode=1 if failure == "stderr" else 0,
            stdout="" if failure == "stderr" else "not-a-pid",
            stderr="unsupported option" if failure == "stderr" else "",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", ps)
    assert benchmark._descendants(111222) == ({111222}, {"ps_ppid"}, False)


def test_owned_tree_limit_marks_omitted_descendants_incomplete(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_child_pids",
        lambda pid: (list(range(111223, 111255)) if pid == 111222 else [], "ps_ppid", True),
    )
    pids, methods, complete = benchmark._descendants(111222)
    assert len(pids) == 32 and 111222 in pids
    assert methods == {"ps_ppid"} and complete is False


def test_partial_owned_tree_withholds_wrapper_only_resource_totals(monkeypatch):
    sampler = benchmark.ProcessSampler(
        {"owned_ollama_tree": 111222}, descendants={"owned_ollama_tree"}
    )
    sampler.available = True
    monkeypatch.setattr(benchmark, "_descendants", lambda _pid: ({111222}, {"ps_ppid"}, False))
    monkeypatch.setattr(
        benchmark,
        "_process",
        lambda pid: {"identity": (pid, "started"), "cpu_ticks": 0, "rss_bytes": 942080},
    )
    sampler.sample()
    report = sampler.summary(1.0)["groups"]["owned_ollama_tree"]
    assert report["observed_process_identities"] == 1
    assert report["discovery_methods"] == ["ps_ppid"]
    assert report["descendant_discovery_complete"] is False
    assert report["available"] is False
    assert report["sampled_peak_rss_bytes"] is None
    assert report["mean_cpu_cores"] is None
    assert report["reason"]


async def test_sampler_io_runs_off_loop_and_cancellation_waits_for_its_one_job(monkeypatch):
    sampler = benchmark.ProcessSampler({"gateway": 111222})
    started, release = threading.Event(), threading.Event()
    jobs = []
    event_loop_thread = threading.get_ident()

    def blocked_sample():
        jobs.append(threading.get_ident())
        started.set()
        assert release.wait(timeout=2)

    monkeypatch.setattr(sampler, "sample", blocked_sample)
    stop = asyncio.Event()
    collector = asyncio.create_task(sampler.collect(stop))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        assert await asyncio.wait_for(asyncio.sleep(0, result=True), timeout=0.1)
        stop.set()
        collector.cancel()
        await asyncio.sleep(0)
        assert not collector.done()
        assert len(jobs) == 1 and jobs[0] != event_loop_thread
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await collector
        assert len(jobs) == 1
    finally:
        release.set()
        await asyncio.gather(collector, return_exceptions=True)
