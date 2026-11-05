"""Bounded closed-loop HTTP benchmarks; mock, cache and real inference stay separate.

Mock suite: .venv/bin/python -m devtools.benchmark --output artifacts/benchmark.json
Existing dedicated Ollama: add --ollama-log artifacts/.../ollama.log --requests 30
Run model workloads sequentially. No inference server/model download is started.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import secrets
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from itertools import islice
from pathlib import Path

import httpx
from fastapi import FastAPI

from devtools.semantic_eval import FROZEN_SHA256, load_dataset
from devtools.semantic_probe import config_text as semantic_config_text
from devtools.semantic_probe import log_posts, server_timings
from devtools.smoke import ROOT, free_port, wait_ready
from llm_gateway.embeddings import fingerprint_for_model
from llm_gateway.schemas import ChatRequest

fixture_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_fixture_stats = {"calls": 0, "active": 0, "peak": 0}


@fixture_app.get("/_benchmark/stats")
async def fixture_stats():
    return _fixture_stats


@fixture_app.post("/_benchmark/reset_peak")
async def fixture_reset_peak():
    if _fixture_stats["active"]:
        raise RuntimeError("Cannot reset an active benchmark window")
    _fixture_stats["peak"] = 0
    return _fixture_stats


@fixture_app.post("/v1/chat/completions")
async def fixture_complete(request: ChatRequest):
    _fixture_stats["calls"] += 1
    _fixture_stats["active"] += 1
    _fixture_stats["peak"] = max(_fixture_stats["peak"], _fixture_stats["active"])
    try:
        await asyncio.sleep(float(os.environ.get("GATEWAY_BENCH_DELAY_MS", "10")) / 1000)
        text = request.messages[-1].content
        city = "Tokyo" if "Japan" in text else "Berlin" if "Germany" in text else "Paris"
        return {
            "id": "chatcmpl-benchmark-fixture",
            "object": "chat.completion",
            "created": 0,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": city},
                    "finish_reason": "stop",
                }
            ],
        }
    finally:
        _fixture_stats["active"] -= 1


def quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    count = len(ordered)

    def rank(fraction: float) -> float | None:
        return ordered[math.ceil(count * fraction) - 1] if count else None

    return {
        "count": count,
        "p50_ms": rank(0.5),
        "p95_ms": rank(0.95) if count >= 100 else None,
        "p99_ms": rank(0.99) if count >= 1000 else None,
        "caveat": "nearest-rank samples; p95 withheld below100 and p99 below1000; "
        "no confidence interval or population-tail guarantee"
        if count < 1000
        else "nearest-rank empirical tails; no confidence interval or population-tail guarantee",
    }


def make_schedule(name: str, count: int, seed: int = 42) -> list[str]:
    if name not in {"direct", "uncached", "exact", "semantic", "mixed"} or count < 0:
        raise ValueError("Invalid benchmark schedule")
    if name != "mixed":
        return [name] * count
    generator, schedule = random.Random(seed), []
    while len(schedule) < count:
        block = ["exact", "exact", "semantic", "semantic", "uncached"]
        generator.shuffle(block)
        schedule.extend(block)
    return schedule[:count]


def point_order(
    concurrency: list[int], scenarios: list[str], repeats: int, seed: int, randomized: bool
) -> list[tuple[int, int, str]]:
    """Retain every point/repeat exactly once in a reproducible order."""
    if not 1 <= repeats <= 5:
        raise ValueError("Use repeats1..5")
    points = [
        (repeat, level, scenario)
        for repeat in range(repeats)
        for level in concurrency
        for scenario in scenarios
    ]
    if randomized:
        random.Random(seed).shuffle(points)
    return points


def cgroup_cpu_stat() -> dict[str, int] | None:
    try:
        return {
            key: int(value)
            for key, value in (
                line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines()
            )
        }
    except (OSError, ValueError):
        return None


def reconcile_upstream_counts(attempts: int, actual: int, forwarded_successes: int) -> dict:
    """An adapter/HTTP attempt need not reach the server after transport failure.

    Successful non-cache responses require observed backend calls. Extra calls or
    missing successful calls invalidate the experiment; failed attempts may be
    unobserved and remain explicit failures in client outcome records.
    """
    if not 0 <= forwarded_successes <= actual <= attempts:
        raise RuntimeError(
            f"Invalid upstream accounting: attempts={attempts}, observed={actual}, "
            f"successful_forwarded_responses={forwarded_successes}"
        )
    return {
        "upstream_attempts": attempts,
        "independent_upstream_calls": actual,
        "unconfirmed_upstream_attempts": attempts - actual,
    }


def reconcile_gateway_counts(attempted: int, completed: int, started: int, finished: int) -> dict:
    # An HTTP attempt can fail before ingress; a response can also be lost after
    # the gateway has completed its ASGI send. Neither is a successful client call.
    if not completed <= finished == started <= attempted:
        raise RuntimeError("Gateway/client counts violate completion or settled-ingress bounds")
    return {"http_attempts_not_observed_at_gateway": attempted - started}


def time_bins(records: list[dict]) -> list[dict]:
    """One-second start-cohort counts; last bin aggregates beyond one hour."""
    bins = {}
    for row in records:
        offset = row["start_offset_seconds"]
        if offset is None:
            offset = row["planned_offset_seconds"]
        index = min(int(offset), 3599)
        bucket = bins.setdefault(index, Counter())
        if row["generator_drop"]:
            bucket["generator_dropped"] += 1
        else:
            bucket["http_started"] += 1
            bucket["successful"] += row["status"] == 200 and row["response_correct"]
            bucket["http_429"] += row["status"] == 429
            bucket["http_503"] += row["status"] == 503
            bucket["http_504"] += row["status"] == 504
            bucket["transport_failed"] += row["status"] is None
    return [{"start_second": index, **dict(counts)} for index, counts in sorted(bins.items())]


async def run_load(
    client: httpx.AsyncClient,
    url: str,
    credential: str,
    payloads: dict[str, bytes],
    schedule: list[str],
    concurrency: int,
    expected_city: str = "Paris",
    *,
    arrival_rate: float | None = None,
) -> dict:
    """Bounded HTTP work: fixed closed-loop workers or a zero-queue arrival schedule.

    In open-loop mode a planned arrival is discarded locally when HTTP slots are
    full or the scheduler is over one period late. Local drops never count as
    attempted HTTP requests, provider failures, or gateway quota rejections.
    """
    if not 1 <= concurrency <= 64 or not 1 <= len(schedule) <= 10_000:
        raise ValueError("Use concurrency1..64 and requests1..10000")
    if arrival_rate is not None and (
        not math.isfinite(arrival_rate) or not 0.1 <= arrival_rate <= 10000
    ):
        raise ValueError("Use finite arrival-rate0.1..10000 or None")
    records, next_index = [None] * len(schedule), 0
    expected_models = {kind: json.loads(body).get("model") for kind, body in payloads.items()}
    started = time.perf_counter()
    active = peak = 0

    def row_for(index: int, planned: float | None) -> dict:
        return {
            "index": index,
            "kind": schedule[index],
            "status": None,
            "cache": None,
            "timing_ms": {},
            "response_correct": False,
            "response_sha256": None,
            "transport_error": None,
            "generator_drop": None,
            "planned_offset_seconds": planned,
            "start_offset_seconds": None,
            "scheduling_lag_ms": None,
            "latency_ms": None,
        }

    async def perform(index: int, planned: float | None):
        nonlocal active, peak
        kind = schedule[index]
        headers = {"Content-Type": "application/json"}
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
        if kind in {"exact", "semantic"}:
            headers["X-Gateway-Cache"] = kind
        elif kind == "uncached":
            headers["Cache-Control"] = "no-store"
        request_started = time.perf_counter()
        row = row_for(index, planned)
        row["start_offset_seconds"] = request_started - started
        row["scheduling_lag_ms"] = (
            max(0.0, (request_started - started - planned) * 1000) if planned is not None else None
        )
        active += 1
        peak = max(peak, active)
        try:
            response = await client.post(url, content=payloads[kind], headers=headers)
            row["latency_ms"] = (time.perf_counter() - request_started) * 1000
            row["status"] = response.status_code
            row["cache"] = response.headers.get("x-gateway-cache")
            if "server-timing" in response.headers:
                row["timing_ms"] = server_timings(response)
            if response.status_code == 200:
                row["response_sha256"] = hashlib.sha256(response.content).hexdigest()
                try:
                    result = response.json()
                    row["response_correct"] = result["choices"][0]["message"][
                        "content"
                    ].strip() == expected_city and (
                        expected_models[kind] is None or result["model"] == expected_models[kind]
                    )
                except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                    row["response_correct"] = False
        except httpx.HTTPError as exc:
            row["latency_ms"] = (time.perf_counter() - request_started) * 1000
            row["transport_error"] = type(exc).__name__
        finally:
            active -= 1
            records[index] = row

    async def worker():
        nonlocal next_index
        while next_index < len(schedule):
            index = next_index
            next_index += 1  # No await: assignment is atomic on this event loop.
            await perform(index, None)

    tasks = set()
    try:
        if arrival_rate is None:
            tasks = {asyncio.create_task(worker()) for _ in range(min(concurrency, len(schedule)))}
            await asyncio.gather(*tasks)
            offer_window = time.perf_counter() - started
        else:
            period = 1 / arrival_rate
            for index in range(len(schedule)):
                planned = index * period
                await asyncio.sleep(max(0.0, started + planned - time.perf_counter()))
                done = {task for task in tasks if task.done()}
                for task in done:
                    task.result()  # Unexpected driver errors must not disappear.
                tasks -= done
                lag = time.perf_counter() - started - planned
                reason = (
                    "scheduler_late"
                    if lag > period
                    else "http_slots_full"
                    if len(tasks) >= concurrency
                    else None
                )
                if reason:
                    row = row_for(index, planned)
                    row["generator_drop"] = reason
                    row["scheduling_lag_ms"] = max(0.0, lag * 1000)
                    records[index] = row
                else:
                    tasks.add(asyncio.create_task(perform(index, planned)))
            await asyncio.sleep(max(0.0, started + len(schedule) * period - time.perf_counter()))
            offer_window = time.perf_counter() - started
            await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    duration = time.perf_counter() - started
    attempts = [row for row in records if row["generator_drop"] is None]
    completed = sum(row["status"] is not None for row in attempts)
    successful = [row for row in attempts if row["status"] == 200 and row["response_correct"]]
    rejected = [row for row in attempts if row["status"] in {429, 503}]
    statuses = Counter(str(row["status"]) for row in attempts)
    return {
        "load_mode": "bounded_open_loop" if arrival_rate is not None else "closed_loop",
        "target_arrival_rate": arrival_rate,
        "planned_requests": len(schedule),
        "planned_offer_window_seconds": len(schedule) / arrival_rate if arrival_rate else None,
        "actual_offer_window_seconds": offer_window,
        "planned_offered_rps": arrival_rate,
        "offered_rps": len(attempts) / offer_window,
        "offered_rate_note": (
            "Actual HTTP starts / offer window; closed-loop starts depend on responses. "
            "Local drops are never HTTP attempts."
        ),
        "generator_dropped": len(records) - len(attempts),
        "generator_drop_reasons": dict(
            Counter(row["generator_drop"] for row in records if row["generator_drop"])
        ),
        "http_active_peak": peak,
        "duration_seconds": duration,
        "attempted": len(attempts),
        "completed": completed,
        "successful": len(successful),
        "http_200": sum(row["status"] == 200 for row in attempts),
        "http_429": statuses.get("429", 0),
        "http_503": statuses.get("503", 0),
        "http_504": statuses.get("504", 0),
        "unexpected_http_responses": sum(
            row["status"] is not None and row["status"] not in {200, 429, 503, 504}
            for row in attempts
        ),
        "rejected": len(rejected),
        "other_http_errors": sum(
            row["status"] is not None and row["status"] >= 400 and row["status"] not in {429, 503}
            for row in attempts
        ),
        "transport_failures": len(attempts) - completed,
        "incorrect_responses": sum(
            row["status"] == 200 and not row["response_correct"] for row in attempts
        ),
        "attempted_rps": len(attempts) / duration,
        "completed_rps": completed / duration,
        "successful_rps": len(successful) / duration,
        "status_counts": dict(statuses),
        "cache_dispositions": dict(Counter(row["cache"] or "UNLABELLED" for row in attempts)),
        "latency_all": quantiles([row["latency_ms"] for row in attempts]),
        "latency_successful": quantiles([row["latency_ms"] for row in successful]),
        "latency_rejected": quantiles([row["latency_ms"] for row in rejected]),
        "scheduling_lag": quantiles(
            [row["scheduling_lag_ms"] for row in records if row["scheduling_lag_ms"] is not None]
        ),
        "scheduling_lag_http_starts": quantiles(
            [row["scheduling_lag_ms"] for row in attempts if row["scheduling_lag_ms"] is not None]
        ),
        "server_timing": {
            name: quantiles(
                [row["timing_ms"][name] for row in attempts if name in row["timing_ms"]]
            )
            for name in ("gateway", "adapter", "exact", "semantic", "residual")
        },
        "raw_records": records,
        "start_cohort_one_second_bins": time_bins(records),
    }


def _process(pid: int) -> dict | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
        return {
            "identity": (pid, fields[19]),
            "cpu_ticks": int(fields[11]) + int(fields[12]),
            "rss_bytes": rss,
        }
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def _child_pids(parent: int) -> tuple[list[int], str, bool]:
    """Read children of one explicitly owned PID, never search unrelated trees."""
    try:
        tasks = list(islice(Path(f"/proc/{parent}/task").iterdir(), 257))
    except OSError:
        tasks = []
    children, readable, failed = set(), 0, len(tasks) > 256
    for task in tasks[:256]:
        try:
            children.update(int(value) for value in (task / "children").read_text().split())
            readable += 1
        except (OSError, ValueError):
            failed = True
    if readable:
        return sorted(children)[:32], "proc_children", not failed and len(children) <= 32

    # Some Linux kernels omit task/children entirely. GNU ps supplies the same
    # parent/child relation without requiring a scan of unrelated command lines.
    try:
        result = subprocess.run(
            ["ps", "-o", "pid=", "--ppid", str(parent)],
            capture_output=True,
            text=True,
            timeout=0.25,
            check=False,
        )
        # GNU ps returns 1 for a valid parent with no matching child processes.
        if result.returncode not in (0, 1) or result.stderr.strip():
            return [], "ps_ppid", False
        values = result.stdout.split()
        children = [int(value) for value in values[:32]]
        if any(child <= 0 for child in children):
            return [], "ps_ppid", False
        return children, "ps_ppid", len(values) <= 32
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return [], "ps_ppid", False


def _descendants(pid: int) -> tuple[set[int], set[str], bool]:
    found, pending, methods, complete = {pid}, [pid], set(), True
    while pending:
        parent = pending.pop()
        children, method, observed_all = _child_pids(parent)
        methods.add(method)
        complete = complete and observed_all
        for child in children:
            if child in found:
                continue
            if len(found) >= 32:
                complete = False
                continue
            found.add(child)
            pending.append(child)
    return found, methods, complete


class ProcessSampler:
    """Sample only owned/explicitly supplied PIDs; retain bounded aggregates."""

    def __init__(self, pids: dict[str, int], *, descendants: set[str] | None = None):
        self.pids, self.descendants = pids, descendants or set()
        self.records, self.group_peaks, self.samples = {}, Counter(), 0
        self.group_first_rss, self.group_last_rss = {}, {}
        self.discovery_methods = {name: set() for name in pids}
        self.discovery_complete = dict.fromkeys(pids, True)
        self.available = platform.system() == "Linux" and Path("/proc/self/stat").is_file()

    def sample(self):
        if not self.available:
            return
        self.samples += 1
        for group, parent in self.pids.items():
            if group in self.descendants:
                pids, methods, complete = _descendants(parent)
            else:
                pids, methods, complete = {parent}, {"supplied_pid"}, True
            self.discovery_methods[group].update(methods)
            self.discovery_complete[group] &= complete
            group_rss = 0
            for pid in pids:
                observed = _process(pid)
                if observed is None:
                    self.discovery_complete[group] = False
                    continue
                key = group, observed["identity"]
                if key not in self.records and len(self.records) >= 128:
                    self.discovery_complete[group] = False
                    continue
                record = self.records.setdefault(
                    key, {"first": observed["cpu_ticks"], "last": observed["cpu_ticks"]}
                )
                record["last"] = observed["cpu_ticks"]
                group_rss += observed["rss_bytes"]
            self.group_peaks[group] = max(self.group_peaks[group], group_rss)
            self.group_first_rss.setdefault(group, group_rss)
            self.group_last_rss[group] = group_rss

    async def collect(self, stop: asyncio.Event):
        while not stop.is_set():
            # One serial worker job: /proc I/O and bounded ps fallback must not
            # pause the HTTP driver's event loop and manufacture latency tails.
            job = asyncio.get_running_loop().run_in_executor(None, self.sample)
            cancelled = False
            while True:
                try:
                    await asyncio.shield(job)
                    break
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.1)
            except TimeoutError:
                pass

    def summary(self, duration: float) -> dict:
        if not self.available:
            return {
                "available": False,
                "reason": "Linux /proc sampling unavailable",
                "groups": None,
            }
        tick_hz = os.sysconf("SC_CLK_TCK")
        groups = {}
        for group in self.pids:
            observed = [row for (name, _), row in self.records.items() if name == group]
            cpu_seconds = sum(max(0, row["last"] - row["first"]) / tick_hz for row in observed)
            complete = self.discovery_complete[group]
            available = bool(observed) and complete
            groups[group] = {
                "available": available,
                "observed_process_identities": len(observed),
                "descendant_discovery_complete": complete,
                "discovery_methods": sorted(self.discovery_methods[group]),
                "cpu_core_seconds": cpu_seconds if available else None,
                "mean_cpu_cores": cpu_seconds / duration if available else None,
                "sampled_peak_rss_bytes": self.group_peaks[group] if available else None,
                "first_sampled_rss_bytes": self.group_first_rss.get(group) if available else None,
                "last_sampled_rss_bytes": self.group_last_rss.get(group) if available else None,
                "reason": (
                    "No successful observations of the supplied PID tree"
                    if not observed
                    else "Process discovery or observation incomplete; aggregate withheld"
                    if not complete
                    else None
                ),
            }
        return {
            "available": True,
            "sample_interval_seconds": 0.1,
            "samples": self.samples,
            "sampling_window_seconds": duration,
            "maximum_recorded_process_identities": 128,
            "groups": groups,
            "note": "CPU cores are CPUtime/walltime, not percent; RSS sums can double-count shared "
            "pages; 10Hz peaks can miss spikes. Process lifetime/model allocation differs. "
            "Owned-child discovery can use bounded GNU ps fallback; its subprocess CPU is not "
            "included in parent CPU ticks. Sampling I/O runs off the driver's event loop.",
        }


def _subtract(before: dict | None, after: dict | None) -> dict:
    before, after = before or {}, after or {}
    return {
        key: _subtract(before.get(key, {}), value)
        if isinstance(value, dict)
        else value - before.get(key, 0)
        for key, value in after.items()
        if isinstance(value, dict)
        or (isinstance(value, (int, float)) and not isinstance(value, bool))
    }


def metrics_delta(before: dict, after: dict) -> dict:
    """Subtract additive counters/buckets; preserve gauges and lifetime maxima."""
    delta = {
        name: _subtract(before.get(name, {}), after.get(name, {}))
        for name in ("counts", "status", "outcomes", "cache", "usage", "semantic_lookup")
    }
    delta["gauges_before"], delta["gauges_after"] = before["gauges"], after["gauges"]
    delta["latency_ms"] = {
        name: {
            "count": histogram["count"] - before["latency_ms"][name]["count"],
            "sum": histogram["sum"] - before["latency_ms"][name]["sum"],
            "lifetime_max_after": histogram["max"],
            "sum_saturated_after": histogram["sum_saturated"],
            "buckets": [
                {
                    "upper_bound_ms": bucket["upper_bound_ms"],
                    "count": bucket["count"] - old["count"],
                }
                for old, bucket in zip(
                    before["latency_ms"][name]["buckets"], histogram["buckets"], strict=True
                )
            ],
        }
        for name, histogram in after["latency_ms"].items()
    }
    delta["rates"] = {}
    for name, rate in after.get("rates", {}).items():
        numerator = rate["numerator"] - before["rates"][name]["numerator"]
        denominator = rate["denominator"] - before["rates"][name]["denominator"]
        delta["rates"][name] = {
            "numerator": numerator,
            "denominator": denominator,
            "value": numerator / denominator if denominator else None,
        }
    old, new = before.get("cache_storage", {}), after.get("cache_storage", {})
    delta["cache_storage_counters"] = _subtract(
        {key: value for key, value in old.items() if key not in {"entries", "bytes_used"}},
        {key: value for key, value in new.items() if key not in {"entries", "bytes_used"}},
    )
    delta["cache_occupancy_after"] = {key: new.get(key) for key in ("entries", "bytes_used")}
    return delta


async def snapshot(client: httpx.AsyncClient, gateway: str, credential: str) -> dict:
    for _ in range(100):
        response = await client.get(
            gateway + "/v1/metrics", headers={"Authorization": f"Bearer {credential}"}
        )
        if response.status_code != 200:
            raise RuntimeError(
                "Operator metrics endpoint unavailable; check gateway implementation"
            )
        observed = response.json()
        if observed["gauges"]["requests_in_flight"] == 0:
            return observed
        await asyncio.sleep(0.01)
    raise RuntimeError("Gateway completion bookkeeping did not settle before metrics snapshot")


def command(app: str, port: int, factory: bool = False) -> list[str]:
    result = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "uvicorn",
        app,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--no-access-log",
        "--log-level",
        "warning",
    ]
    return result + (["--factory"] if factory else [])


def stop_process(process: subprocess.Popen):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


@contextmanager
def backend_server(delay_ms: float, real: bool):
    if real:
        yield "http://127.0.0.1:11434", None
        return
    port = free_port()
    environment = os.environ.copy()
    environment["GATEWAY_BENCH_DELAY_MS"] = str(delay_ms)
    process = subprocess.Popen(
        command("devtools.benchmark:fixture_app", port),
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        url = f"http://127.0.0.1:{port}"
        with httpx.Client(trust_env=False, timeout=2.0) as client:
            wait_ready(client, process, url + "/_benchmark/stats", 200)
        yield url, process.pid
    finally:
        stop_process(process)


@contextmanager
def gateway_server(
    backend: str, scenario: str, concurrency: int, args, families: dict, threshold: float
):
    if scenario == "direct":
        yield None, None, None, None, None
        return
    real = args.ollama_log is not None
    recipe = semantic_config_text(backend, args.model_dir, threshold, families, real)
    original = tomllib.loads(recipe)
    capacity = 1 if real else 8
    recipe = recipe.replace(
        f"max_in_flight = {original['max_in_flight']}", f"max_in_flight = {capacity}", 1
    )
    if "max_connections" in original:
        recipe = recipe.replace(
            f"max_connections = {original['max_connections']}", "max_connections = 64", 1
        )
    else:
        recipe = "max_connections = 64\n" + recipe
    recipe = recipe.replace(f"burst = {original['rate_limit']['burst']}", "burst = 10000", 1)
    recipe = recipe.replace(
        f"requests_per_second = {original['rate_limit']['requests_per_second']}",
        "requests_per_second = 10000.0",
        1,
    )
    if "approved_prompts" not in original["exact_cache"]:
        recipe = recipe.replace(
            "[exact_cache]\n",
            "[exact_cache]\napproved_prompts = "
            + json.dumps([families["france-capital"]["canonical"]])
            + "\n",
            1,
        )
    if scenario == "uncached":
        recipe = recipe.replace(
            "[exact_cache]\nenabled = true", "[exact_cache]\nenabled = false", 1
        )
        # approved_prompts can precede enabled in the generated mock table.
        values = tomllib.loads(recipe)
        if values["exact_cache"]["enabled"]:
            recipe = recipe.replace("enabled = true", "enabled = false", 1)
    if scenario not in {"semantic", "mixed"}:
        recipe = recipe.replace(
            "[semantic_cache]\nenabled = true", "[semantic_cache]\nenabled = false", 1
        )
    if not real:
        recipe = recipe.replace('model = "deterministic-mock"', 'model = "demo"', 1)
    parsed = tomllib.loads(recipe)
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="benchmark-", dir=ROOT / "artifacts") as directory:
        config = Path(directory) / "gateway.toml"
        config.write_text(recipe)
        key, metrics_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        environment = os.environ.copy()
        environment.update(
            GATEWAY_CONFIG=str(config),
            GATEWAY_CLIENT_KEYS_JSON=json.dumps({"benchmark": key}),
            GATEWAY_METRICS_KEY=metrics_key,
        )
        process = subprocess.Popen(
            command("llm_gateway.app:create_app", port, True),
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            with httpx.Client(trust_env=False, timeout=2.0) as client:
                wait_ready(client, process, base + "/healthz", 200)
            yield base, key, metrics_key, process.pid, parsed
        finally:
            stop_process(process)


def payloads_for(families: dict, real: bool) -> dict[str, bytes]:
    canonical = {
        "model": "gemma" if real else "demo",
        "messages": [{"role": "user", "content": families["france-capital"]["canonical"]}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": 32,
    }
    semantic = canonical | {
        "messages": [
            {"role": "user", "content": families["france-capital"]["paraphrases"]["heldout"][0]}
        ]
    }
    direct = canonical | ({"model": "gemma3:1b"} if real else {})
    return {
        kind: json.dumps(payload, separators=(",", ":")).encode()
        for kind, payload in {
            "direct": direct,
            "uncached": canonical,
            "exact": canonical,
            "semantic": semantic,
        }.items()
    }


async def warmup(client, endpoint, key, scenario, payloads, count):
    seed = None
    if scenario in {"exact", "semantic", "mixed"}:
        kind = "exact" if scenario == "exact" else "semantic"
        seeded = payloads | {kind: payloads["exact"]}
        seed = await run_load(client, endpoint, key, seeded, [kind], 1)
        if seed["successful"] != 1 or seed["raw_records"][0]["cache"] != "MISS":
            raise RuntimeError("Fresh cache seed failed")
    schedule = make_schedule(scenario, count, seed=42)
    warmed = await run_load(client, endpoint, key, payloads, schedule, 1)
    if warmed["successful"] != count:
        raise RuntimeError("Benchmark warmup failed correctness/status checks")
    expected = "HIT" if scenario == "exact" else "SEMANTIC_HIT" if scenario == "semantic" else None
    return {
        "cache_requirement_met": not expected or warmed["cache_dispositions"] == {expected: count},
        "phase_records": (seed["raw_records"] if seed else []) + warmed["raw_records"],
        "requests": count + (1 if seed else 0),
        "seed_cache": seed["cache_dispositions"] if seed else None,
        "primed_source_sha256": seed["raw_records"][0]["response_sha256"] if seed else None,
        "measured_cache": warmed["cache_dispositions"],
        "upstream_calls_expected": sum(
            row["cache"] not in {"HIT", "SEMANTIC_HIT"}
            for phase in ([seed, warmed] if seed else [warmed])
            for row in phase["raw_records"]
        ),
        "excluded_from_measurement": True,
    }


def verify_wire_replays(measured: dict, warming: dict, scenario: str, concurrency: int) -> dict:
    """Attest only hits with a controlled source; concurrent fallback has new lineage."""
    hits = [row for row in measured["raw_records"] if row["cache"] in {"HIT", "SEMANTIC_HIT"}]
    source = warming["primed_source_sha256"]
    controlled = [
        row
        for row in hits
        if concurrency == 1
        or scenario == "exact"
        or (row["kind"] == "exact" and row["cache"] == "HIT")
    ]
    for row in controlled:
        if source is None or row["response_sha256"] != source:
            raise RuntimeError("Controlled cache replay differs from primed source bytes")
    return {
        "cache_wire_replay_verified": True if hits and len(controlled) == len(hits) else None,
        "checked_cache_wire_replays": len(controlled),
        "total_cache_hits": len(hits),
        "wire_replay_scope": "controlled canonical source; all single-client-worker hits and "
        "canonical exact hits checked. Concurrent semantic fallback lineage is not asserted.",
    }


async def real_identity(client: httpx.AsyncClient, backend: str, *, loaded=False) -> dict:
    config = tomllib.loads((ROOT / "config/ollama.toml").read_text())
    revision = config["backends"]["local"]["revision"]
    assert revision.startswith("ollama-0.40.11-")
    version = await client.get(backend + "/api/version")
    version.raise_for_status()
    assert version.json()["version"] == "0.40.11"
    response = await client.get(backend + ("/api/ps" if loaded else "/api/tags"))
    response.raise_for_status()
    model = next(row for row in response.json()["models"] if row["name"] == "gemma3:1b")
    digest = "sha256:" + model["digest"].removeprefix("sha256:")
    assert digest == config["models"]["gemma"]["revision"]
    if loaded:
        assert model["context_length"] == 2048
    return {
        "version": "0.40.11",
        "model_digest": digest,
        "backend_revision": revision,
        "observed_context_length": model.get("context_length"),
    }


async def backend_calls(client, backend, log, expected=None):
    if log is None:
        return (await client.get(backend + "/_benchmark/stats")).json()["calls"]
    # A successful HTTP body can reach the caller before GIN flushes its access
    # line. Settle outside the timed workload, both before and after measurement.
    observed = log_posts(log)
    started = changed = time.monotonic()
    while time.monotonic() - started < 1.0:
        if (expected is None or observed >= expected) and time.monotonic() - changed >= 0.1:
            return observed
        await asyncio.sleep(0.025)
        current = log_posts(log)
        if current != observed:
            observed, changed = current, time.monotonic()
    raise RuntimeError("Ollama log count did not reach its expected stable baseline within1s")


def hardware() -> dict:
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    packages = {}
    for name in (
        "fastapi",
        "starlette",
        "httpx",
        "pydantic",
        "uvicorn",
        "numpy",
        "onnxruntime",
        "tokenizers",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    cpu, memory = read("/sys/fs/cgroup/cpu.max"), read("/sys/fs/cgroup/memory.max")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "cgroup_cpu_max": cpu,
        "cgroup_memory_max": memory,
        "packages": packages,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


def source_manifest() -> dict[str, str]:
    """Freeze executable project sources and the lockfile, excluding editable docs."""
    paths = [
        Path(__file__),
        ROOT / "devtools/semantic_probe.py",
        ROOT / "devtools/semantic_eval.py",
        ROOT / "devtools/smoke.py",
        ROOT / "uv.lock",
        *(
            path
            for name in ("load_campaign.py", "load_reliability.py")
            if (path := ROOT / "devtools" / name).exists()
        ),
        *(ROOT / "src/llm_gateway").glob("*.py"),
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def verify_source_manifest(initial: dict[str, str]) -> None:
    current = source_manifest()
    changed = sorted(
        name for name in initial.keys() | current.keys() if initial.get(name) != current.get(name)
    )
    if changed:
        raise RuntimeError("Benchmark source changed during execution: " + ", ".join(changed))


async def run_suite(args) -> dict:
    suite_started_utc = datetime.now(UTC).isoformat()
    initial_sources = source_manifest()
    _, families, _ = load_dataset()
    evidence = json.loads(args.evaluation.read_text())
    fingerprint = fingerprint_for_model(args.model_dir)
    assert (
        evidence["dataset_sha256"] == FROZEN_SHA256 and evidence["model_fingerprint"] == fingerprint
    )
    selection = evidence["selected"]["guarded"]
    assert (
        selection["selection_split"] == "calibration"
        and selection["calibration_metrics"]["false_hits"] == 0
    )
    assert evidence["heldout_at_selected_thresholds"]["guarded"]["false_hits"] == 0
    threshold, real, results = selection["threshold"], args.ollama_log is not None, []
    unavailable_points = []
    payloads = payloads_for(families, real)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with backend_server(args.delay_ms, real) as (backend, backend_pid):
        points = point_order(
            args.concurrency, args.scenarios, args.repeats, args.seed, args.randomize_order
        )
        for run_order, (repeat, concurrency, scenario) in enumerate(points):
            setup_failures = []
            for setup_attempt in range(3):
                with gateway_server(backend, scenario, concurrency, args, families, threshold) as (
                    gateway,
                    key,
                    operator_key,
                    gateway_pid,
                    config,
                ):
                    async with httpx.AsyncClient(
                        trust_env=False,
                        timeout=70.0 if real else 15.0,
                        limits=httpx.Limits(max_connections=64, max_keepalive_connections=64),
                    ) as client:
                        endpoint = (gateway or backend) + "/v1/chat/completions"
                        if real:
                            await real_identity(client, backend)
                        before_warmup_calls = await backend_calls(client, backend, args.ollama_log)
                        warming = await warmup(
                            client, endpoint, key or "", scenario, payloads, args.warmup
                        )
                        identity = (
                            await real_identity(client, backend, loaded=True) if real else None
                        )
                        before = await snapshot(client, gateway, operator_key) if gateway else None
                        expected_warm_calls = (
                            before_warmup_calls + warming["upstream_calls_expected"]
                        )
                        initial_calls = await backend_calls(
                            client, backend, args.ollama_log, expected_warm_calls
                        )
                        if initial_calls != expected_warm_calls:
                            raise RuntimeError(
                                "Warmup backend count differs; require a dedicated backend/log"
                            )
                        if not warming["cache_requirement_met"]:
                            setup_failures.append(
                                {
                                    "attempt": setup_attempt + 1,
                                    "warmup": warming,
                                    "metrics_after_setup": before,
                                }
                            )
                            print(
                                json.dumps(
                                    {
                                        "setup_failed": run_order + 1,
                                        "attempt": setup_attempt + 1,
                                        "scenario": scenario,
                                        "concurrency": concurrency,
                                    }
                                ),
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                        fixture_before = None
                        if not real:
                            reset = await client.post(backend + "/_benchmark/reset_peak")
                            reset.raise_for_status()
                            fixture_before = reset.json()
                            if fixture_before["active"]:
                                raise RuntimeError("Mock backend not idle before measured window")
                        owned_pids = {"load_generator": os.getpid()}
                        if gateway_pid:
                            owned_pids["gateway"] = gateway_pid
                        if backend_pid:
                            owned_pids["mock_backend"] = backend_pid
                        if args.ollama_pid:
                            owned_pids["owned_ollama_tree"] = args.ollama_pid
                        sampler = ProcessSampler(owned_pids, descendants={"owned_ollama_tree"})
                        resource_started = time.perf_counter()
                        cgroup_before = cgroup_cpu_stat()
                        sampler.sample()
                        stop = asyncio.Event()
                        sampling = asyncio.create_task(sampler.collect(stop))
                        try:
                            measured_started_utc = datetime.now(UTC).isoformat()
                            measured = await run_load(
                                client,
                                endpoint,
                                key or "",
                                payloads,
                                make_schedule(scenario, args.requests, args.seed),
                                concurrency,
                                arrival_rate=args.arrival_rate,
                            )
                            measured_finished_utc = datetime.now(UTC).isoformat()
                        finally:
                            stop.set()
                            await sampling
                            sampler.sample()
                        resource_duration = time.perf_counter() - resource_started
                        cgroup_after = cgroup_cpu_stat()
                        after = await snapshot(client, gateway, operator_key) if gateway else None
                        delta = metrics_delta(before, after) if gateway else None
                        attempts = (
                            delta["counts"]["upstream_attempts"] if delta else measured["attempted"]
                        )
                        forwarded_successes = sum(
                            row["status"] == 200 and row["cache"] not in {"HIT", "SEMANTIC_HIT"}
                            for row in measured["raw_records"]
                        )
                        actual_calls = (
                            await backend_calls(
                                client,
                                backend,
                                args.ollama_log,
                                initial_calls + forwarded_successes,
                            )
                            - initial_calls
                        )
                        try:
                            count_evidence = reconcile_upstream_counts(
                                attempts, actual_calls, forwarded_successes
                            )
                        except RuntimeError:
                            print(
                                json.dumps(
                                    {
                                        "invalid_point": {
                                            "run_order": run_order,
                                            "scenario": scenario,
                                            "concurrency": concurrency,
                                        },
                                        "attempts": attempts,
                                        "observed_calls": actual_calls,
                                        "forwarded_successes": forwarded_successes,
                                        "status_counts": measured["status_counts"],
                                        "transport_failures": measured["transport_failures"],
                                    }
                                ),
                                file=sys.stderr,
                                flush=True,
                            )
                            raise
                        fixture_after = None
                        if not real:
                            fixture_after = (await client.get(backend + "/_benchmark/stats")).json()
                            if fixture_after["active"]:
                                raise RuntimeError("Mock backend retained active work after load")
                            if gateway and fixture_after["peak"] > config["max_in_flight"]:
                                raise RuntimeError(
                                    "Observed mock inference peak exceeds admission limit"
                                )
                        ingress_evidence = (
                            reconcile_gateway_counts(
                                measured["attempted"],
                                measured["completed"],
                                delta["counts"]["requests_started"],
                                delta["counts"]["requests_finished"],
                            )
                            if gateway
                            else None
                        )
                        successful_cache_dispositions = Counter(
                            row["cache"] for row in measured["raw_records"] if row["status"] == 200
                        )
                        if (
                            scenario == "exact"
                            and successful_cache_dispositions != {"HIT": measured["http_200"]}
                            and measured["http_200"]
                        ):
                            raise RuntimeError("Warm exact workload was not entirely cache hits")
                        if (
                            scenario == "semantic"
                            and concurrency == 1
                            and successful_cache_dispositions
                            != {"SEMANTIC_HIT": measured["http_200"]}
                            and measured["http_200"]
                        ):
                            raise RuntimeError(
                                "Single-worker semantic workload was not entirely semantic hits"
                            )
                        hits = sum(
                            measured["cache_dispositions"].get(name, 0)
                            for name in ("HIT", "SEMANTIC_HIT")
                        )
                        replay_checks = verify_wire_replays(
                            measured, warming, scenario, concurrency
                        )
                        # Probe after all measured snapshots/counts: not part of RPS,
                        # percentiles, resource window, or measured upstream calls.
                        recovery = await run_load(
                            client,
                            endpoint,
                            key or "",
                            payloads,
                            ["uncached" if gateway else "direct"],
                            1,
                        )
                        recovery_calls = (
                            await backend_calls(
                                client,
                                backend,
                                args.ollama_log,
                                initial_calls + actual_calls + recovery["http_200"],
                            )
                            - initial_calls
                            - actual_calls
                        )
                        reconcile_upstream_counts(1, recovery_calls, recovery["http_200"])
                        recovery_metrics = (
                            await snapshot(client, gateway, operator_key) if gateway else None
                        )
                        if recovery_metrics and recovery_metrics["gauges"]["inference_in_flight"]:
                            raise RuntimeError("Post-load recovery leaked inference capacity")
                        measured.update(
                            **replay_checks,
                            scenario=scenario,
                            repeat_index=repeat,
                            run_order=run_order,
                            process_ids=owned_pids,
                            separate_gateway_process=gateway_pid is None
                            or gateway_pid != os.getpid(),
                            cgroup_cpu_stat_before=cgroup_before,
                            cgroup_cpu_stat_after=cgroup_after,
                            cgroup_cpu_stat_delta=_subtract(cgroup_before, cgroup_after)
                            if cgroup_before is not None and cgroup_after is not None
                            else None,
                            fixture_stats_before=fixture_before,
                            fixture_stats_after=fixture_after,
                            observed_peak_inference_concurrency=fixture_after["peak"]
                            if fixture_after is not None
                            else None,
                            peak_inference_note=(
                                "Mock fixture window peak; real Ollama execution peak is unobserved"
                            ),
                            recovery={
                                "successful": recovery["successful"],
                                "status_counts": recovery["status_counts"],
                                "latency_successful": recovery["latency_successful"],
                                "independent_upstream_calls": recovery_calls,
                                "excluded_from_measurement": True,
                                "inference_in_flight_after": recovery_metrics["gauges"][
                                    "inference_in_flight"
                                ]
                                if recovery_metrics
                                else None,
                            },
                            started_utc=measured_started_utc,
                            finished_utc=measured_finished_utc,
                            concurrency=concurrency,
                            warmup=warming,
                            setup_failures=setup_failures,
                            model_identity=identity,
                            independent_upstream_calls=actual_calls,
                            ingress_count_evidence=ingress_evidence,
                            unconfirmed_upstream_attempts=count_evidence[
                                "unconfirmed_upstream_attempts"
                            ],
                            call_count_source="dedicated_ollama_gin_log"
                            if real
                            else "mock_fixture_stats",
                            inference_calls_avoided_by_hits=hits,
                            cache_avoidance_fraction_of_attempts=(
                                hits / measured["attempted"] if measured["attempted"] else None
                            ),
                            metrics_before=before,
                            metrics_after=after,
                            metrics_delta=delta,
                            resources=sampler.summary(resource_duration),
                            gateway_config=config,
                        )
                        results.append(measured)
                        print(
                            json.dumps(
                                {
                                    "point_completed": run_order + 1,
                                    "scenario": scenario,
                                    "concurrency": concurrency,
                                    "repeat_index": repeat,
                                    "successful_rps": measured["successful_rps"],
                                    "http_503": measured["http_503"],
                                }
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                    break
            else:
                unavailable_points.append(
                    {
                        "scenario": scenario,
                        "concurrency": concurrency,
                        "repeat_index": repeat,
                        "run_order": run_order,
                        "reason": "Warm-cache setup unavailable after3freshstarts",
                        "setup_failures": setup_failures,
                    }
                )
    verify_source_manifest(initial_sources)
    return {
        "schema_version": 1,
        "unavailable_points": unavailable_points,
        "setup_attempts_limit": 3,
        "code_manifest": initial_sources,
        "code_manifest_verified_unchanged_at_end": True,
        "suite_started_utc": suite_started_utc,
        "suite_finished_utc": datetime.now(UTC).isoformat(),
        "backend": "ollama" if real else "deterministic_mock",
        "real_model_inference": real,
        "mock_delay_ms": None if real else args.delay_ms,
        "requests_per_scenario": args.requests,
        "concurrency_levels": args.concurrency,
        "seed": args.seed,
        "repeats": args.repeats,
        "randomized_point_order": args.randomize_order,
        "target_arrival_rate": args.arrival_rate,
        "load_generator_pid": os.getpid(),
        "hardware_and_software": hardware(),
        "embedding_fingerprint": fingerprint,
        "dataset_sha256": FROZEN_SHA256,
        "calibration_selected_threshold": threshold,
        "uvicorn_workers": 1,
        "http_client_pool": {"max_connections": 64, "keepalive_connections": 64},
        "request_payloads": {
            kind: {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            for kind, body in payloads.items()
        },
        "generation_settings": {"temperature": 0.0, "top_p": 1.0, "seed": 42, "max_tokens": 32},
        "response_requirement": "Paris only for the frozen France-capital family",
        "warmup_concurrency": 1,
        "direct_and_uncached_body_identical": payloads["direct"] == payloads["uncached"],
        "real_body_difference": "public model alias vs same physical model identifier"
        if real
        else None,
        "results": results,
        "limitations": [
            "Closed-loop starts depend on responses; optional open-loop schedule drops local "
            "arrivals with no client queue.",
            "Cache RPS is not model generation speed; no token-rate or monetary savings claim.",
            "Header durations are walltime, not CPU; direct/gateway differences are not CPU costs.",
            "Concurrent semantic fallback can create later exact hits; dispositions show this.",
            "Sampler, client and servers share hardware; CPU ticks and RSS samples are coarse.",
            "Repeated real prompts can reuse backend KV caches independently of gateway caching.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests", type=int, help="Per-scenario requests1..10000; defaultmock1000/real30"
    )
    parser.add_argument("--concurrency", type=int, nargs="+", help="Defaultmock1,4,8,16; real1..4")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["direct", "uncached", "exact", "semantic", "mixed"],
        default=["direct", "uncached", "exact", "semantic", "mixed"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--randomize-order", action="store_true")
    parser.add_argument(
        "--arrival-rate", type=float, help="Optional planned requests/second; zero client queue"
    )
    parser.add_argument("--delay-ms", type=float, default=10.0)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/embeddings/l3"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation/l3_results.json"))
    parser.add_argument(
        "--ollama-log", type=Path, help="Existing dedicated log under artifacts/ enablesrealmode"
    )
    parser.add_argument(
        "--ollama-pid",
        type=int,
        help="Explicitly owned Ollama server/wrapper PID; only its descendants sampled",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Full JSON/rawrecords under artifacts/"
    )
    args = parser.parse_args()
    args.ollama_log = (ROOT / args.ollama_log).resolve() if args.ollama_log else None
    args.requests = (
        args.requests if args.requests is not None else (30 if args.ollama_log else 1000)
    )
    args.concurrency = args.concurrency or ([1] if args.ollama_log else [1, 4, 8, 16])
    if not 1 <= args.repeats <= 5:
        parser.error("Use repeats1..5")
    if args.arrival_rate is not None and (
        not math.isfinite(args.arrival_rate)
        or not 0.1 <= args.arrival_rate <= 10000
        or args.requests / args.arrival_rate > 600
    ):
        parser.error("Use finite arrival-rate0.1..10000 and at most600seconds planned window")
    if (
        not 1 <= args.requests <= 10000
        or not 1 <= args.warmup <= 20
        or not 0 <= args.delay_ms <= 1000
    ):
        parser.error("Use requests1..10000, warmup1..20, delay-ms0..1000")
    if (
        len(args.concurrency) > 8
        or len(set(args.concurrency)) != len(args.concurrency)
        or any(not 1 <= item <= 64 for item in args.concurrency)
    ):
        parser.error("Use at most8 distinct concurrency levels in1..64")
    if args.ollama_log and (
        not args.ollama_log.is_relative_to(ROOT / "artifacts") or not args.ollama_log.is_file()
    ):
        parser.error("--ollama-log must be an existing file under artifacts/")
    if args.ollama_log and any(level > 4 for level in args.concurrency):
        parser.error("Consumer-hardware real inference benchmark supports concurrency1..4")
    if args.ollama_pid is not None and (not args.ollama_log or args.ollama_pid < 1):
        parser.error("--ollama-pid requires realmode and a positive owned server PID")
    args.model_dir, args.evaluation, output = (
        (ROOT / args.model_dir).resolve(),
        (ROOT / args.evaluation).resolve(),
        (ROOT / args.output).resolve(),
    )
    if (
        not args.model_dir.is_relative_to(ROOT)
        or not args.model_dir.is_dir()
        or not args.evaluation.is_relative_to(ROOT)
        or not args.evaluation.is_file()
    ):
        parser.error("Model directory/evaluation must exist within this repository")
    if (
        not output.is_relative_to(ROOT / "artifacts")
        or output == ROOT / "artifacts"
        or output in {args.ollama_log, args.evaluation}
        or output.is_relative_to(args.model_dir)
    ):
        parser.error("--output must be an artifacts/ file, not an input log/evaluation/model file")
    report = asyncio.run(run_suite(args))
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    # Serialization can outlive the measurement; check again immediately before
    # publishing so imported old code cannot be attributed to edited new files.
    verify_source_manifest(report["code_manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized)
    print(
        json.dumps(
            {
                "backend": report["backend"],
                "results_file": str(output.relative_to(ROOT)),
                "points": [
                    {
                        key: result[key]
                        for key in (
                            "scenario",
                            "concurrency",
                            "successful_rps",
                            "latency_all",
                            "cache_dispositions",
                            "independent_upstream_calls",
                        )
                    }
                    for result in report["results"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
