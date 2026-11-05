"""Reproduce M08 with an idle controller and a separate HTTP load-driver process.

All processes still share this machine. This is not an independent-hardware test.
Run stages sequentially; Gemma requires an existing dedicated, quiet Ollama.
"""

import argparse
import gzip
import hashlib
import json
import os
import signal
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from devtools.benchmark import hardware, source_manifest, verify_source_manifest
from devtools.smoke import ROOT


def workloads(stage: str) -> list[tuple[str, list[str]]]:
    if stage == "primary":
        return [
            (
                "primary",
                [
                    "--requests",
                    "1000",
                    "--concurrency",
                    "1",
                    "2",
                    "4",
                    "8",
                    "16",
                    "--scenarios",
                    "direct",
                    "uncached",
                    "exact",
                    "semantic",
                    "mixed",
                    "--repeats",
                    "3",
                    "--randomize-order",
                ],
            )
        ]
    if stage == "ramp":
        return [
            (
                f"arrival-{rate}",
                [
                    "--requests",
                    "3000",
                    "--concurrency",
                    "16",
                    "--scenarios",
                    "uncached",
                    "--arrival-rate",
                    str(rate),
                ],
            )
            for rate in (100, 300, 600)
        ]
    if stage == "sustained":
        return [
            (
                "sustained-uncached",
                [
                    "--requests",
                    "10000",
                    "--concurrency",
                    "8",
                    "16",
                    "--scenarios",
                    "uncached",
                    "--randomize-order",
                ],
            ),
            (
                "sustained-exact",
                [
                    "--requests",
                    "6000",
                    "--concurrency",
                    "16",
                    "--scenarios",
                    "exact",
                    "--arrival-rate",
                    "200",
                ],
            ),
            (
                "sustained-semantic",
                [
                    "--requests",
                    "3000",
                    "--concurrency",
                    "1",
                    "--scenarios",
                    "semantic",
                    "--arrival-rate",
                    "100",
                ],
            ),
        ]
    if stage == "gemma":
        return [
            (
                "gemma-closed",
                [
                    "--requests",
                    "20",
                    "--concurrency",
                    "1",
                    "2",
                    "--scenarios",
                    "direct",
                    "uncached",
                    "--repeats",
                    "2",
                    "--randomize-order",
                ],
            ),
            (
                "gemma-arrival-1",
                [
                    "--requests",
                    "20",
                    "--concurrency",
                    "2",
                    "--scenarios",
                    "uncached",
                    "--arrival-rate",
                    "1",
                ],
            ),
            (
                "gemma-arrival-3",
                [
                    "--requests",
                    "30",
                    "--concurrency",
                    "2",
                    "--scenarios",
                    "uncached",
                    "--arrival-rate",
                    "3",
                ],
            ),
        ]
    raise ValueError("Unknown campaign stage")


def brief(point: dict) -> dict:
    keys = (
        "scenario",
        "concurrency",
        "repeat_index",
        "run_order",
        "duration_seconds",
        "planned_requests",
        "attempted",
        "completed",
        "successful",
        "generator_dropped",
        "offered_rps",
        "attempted_rps",
        "successful_rps",
        "target_arrival_rate",
        "status_counts",
        "cache_dispositions",
        "latency_successful",
        "latency_rejected",
        "latency_all",
        "scheduling_lag",
        "scheduling_lag_http_starts",
        "transport_failures",
        "incorrect_responses",
        "other_http_errors",
        "unexpected_http_responses",
        "independent_upstream_calls",
        "unconfirmed_upstream_attempts",
        "observed_peak_inference_concurrency",
        "http_active_peak",
        "process_ids",
        "recovery",
        "setup_failures",
        "cgroup_cpu_stat_delta",
        "resources",
        "start_cohort_one_second_bins",
    )
    return {key: point.get(key) for key in keys}


def aggregate(points: list[dict]) -> list[dict]:
    groups = {}
    for point in points:
        groups.setdefault((point["scenario"], point["concurrency"]), []).append(point)
    rows = []
    for (scenario, concurrency), trials in sorted(groups.items()):
        rps = [point["successful_rps"] for point in trials]
        rows.append(
            {
                "scenario": scenario,
                "concurrency": concurrency,
                "trials": len(trials),
                "successful_rps_median": statistics.median(rps),
                "successful_rps_min": min(rps),
                "successful_rps_max": max(rps),
                "successful_p50_ms_median": statistics.median(
                    p["latency_successful"]["p50_ms"]
                    for p in trials
                    if p["latency_successful"]["p50_ms"] is not None
                )
                if any(p["latency_successful"]["p50_ms"] is not None for p in trials)
                else None,
                "http_429_total": sum(p["status_counts"].get("429", 0) for p in trials),
                "http_503_total": sum(p["status_counts"].get("503", 0) for p in trials),
                "http_504_total": sum(p["status_counts"].get("504", 0) for p in trials),
                "actual_upstream_calls_total": sum(p["independent_upstream_calls"] for p in trials),
                "note": "Trial median/range; no pooled quantiles or confidence bounds",
            }
        )
    return rows


def run(args) -> dict:
    initial = source_manifest()
    started = datetime.now(UTC).isoformat()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials = []
    for name, options in workloads(args.stage):
        report_path = output / f"{name}.json"
        command = [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "devtools.benchmark",
            *options,
            "--seed",
            str(args.seed),
            "--warmup",
            "5",
            "--delay-ms",
            "10",
            "--output",
            str(report_path),
        ]
        if args.stage == "gemma":
            command += ["--ollama-log", str(args.ollama_log)]
            if args.ollama_pid:
                command += ["--ollama-pid", str(args.ollama_pid)]
        if args.dry_run:
            trials.append({"name": name, "command": command})
            continue
        with (output / f"{name}.stdout.log").open("wb") as log:
            # Parent is idle while child drives HTTP and measures owned servers.
            child = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = child.wait()
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=3)
                raise
        if code:
            raise RuntimeError(f"Benchmark child failed ({code}); inspect {name}.stdout.log")
        verify_source_manifest(initial)
        report = json.loads(report_path.read_text())
        if report["code_manifest"] != initial:
            raise RuntimeError("Child measured a different source manifest")
        raw = report_path.read_bytes()
        with (output / f"{name}.json.gz").open("wb") as compressed:
            with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as stream:
                stream.write(raw)
        trials.append(
            {
                "name": name,
                "command": command,
                "load_driver_pid": child.pid,
                "controller_pid": os.getpid(),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_gzip_file": str((output / f"{name}.json.gz").relative_to(ROOT)),
                "backend": report["backend"],
                "unavailable_points": report.get("unavailable_points", []),
                "points": [brief(point) for point in report["results"]],
                "aggregate": aggregate(report["results"]),
            }
        )
        print(
            json.dumps({"stage": args.stage, "completed": name, "points": len(report["results"])}),
            flush=True,
        )
    verify_source_manifest(initial)
    return {
        "schema_version": 1,
        "stage": args.stage,
        "source_manifest": initial,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "hardware_and_software": hardware(),
        "controller_pid": os.getpid(),
        "independent_hardware": False,
        "separate_load_driver_process": True,
        "conditions": "Loopback; all processes share the same cgroup CPU and memory budget",
        "seed": args.seed,
        "dry_run": args.dry_run,
        "trials": trials,
        "limitations": [
            "Closed-loop issue rate is achieved demand, not independently controlled arrivals.",
            "Open-loop local drops never reach gateway and are not HTTP rejections.",
            "Client, sampler, gateway and backend contend for two CPU cores; no remote hardware.",
            "RPS is verified successful responses; rejection throughput is reported separately.",
            "Concurrent reviewed semantic requests may become exact hits; inspect dispositions.",
            "Sampled RSS and short sustained windows cannot prove absence of memory leaks.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("primary", "ramp", "sustained", "gemma"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--ollama-log", type=Path)
    parser.add_argument("--ollama-pid", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir = (ROOT / args.output_dir).resolve()
    if (
        not args.output_dir.is_relative_to(ROOT / "artifacts")
        or args.output_dir == ROOT / "artifacts"
    ):
        parser.error("--output-dir must be below repository artifacts/")
    if args.stage == "gemma" and not args.ollama_log:
        parser.error("Gemma stage requires a dedicated existing --ollama-log")
    report = run(args)
    (args.output_dir / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
