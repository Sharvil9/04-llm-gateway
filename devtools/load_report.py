"""Package M08 reports; optional runtime capture sends metadata GETs, never inference."""

import argparse
import gzip
import hashlib
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path

import httpx

from devtools.benchmark import source_manifest
from devtools.semantic_probe import log_posts
from devtools.smoke import ROOT


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def aggregate(points: list[dict]) -> list[dict]:
    groups = {}
    for point in points:
        groups.setdefault((point["scenario"], point["concurrency"]), []).append(point)
    rows = []
    for (scenario, concurrency), trials in sorted(groups.items()):
        rates = [point["successful_rps"] for point in trials]
        latencies = {}
        for quantile in ("p50_ms", "p95_ms", "p99_ms"):
            values = [point["latency_successful"][quantile] for point in trials]
            latencies[quantile] = (
                statistics.median(values) if all(x is not None for x in values) else None
            )
        dispositions = Counter()
        for point in trials:
            dispositions.update(point["cache_dispositions"])
        rows.append(
            {
                "scenario": scenario,
                "concurrency": concurrency,
                "trials": len(trials),
                "successful_rps_median": statistics.median(rates),
                "successful_rps_range": [min(rates), max(rates)],
                "median_of_trial_successful_latency_quantiles": latencies,
                "quantile_scope": "Trial-quantile median; withheld if any trial is undersampled",
                "statuses": dict(
                    sum((Counter(point["status_counts"]) for point in trials), Counter())
                ),
                "actual_upstream_calls": sum(
                    point["independent_upstream_calls"] for point in trials
                ),
                "cache_dispositions": dict(dispositions),
                "pure_expected_hit_trials": sum(
                    point.get("cache_purity", {}).get("all_http_200_expected_type") is True
                    for point in trials
                    if point.get("cache_purity") is not None
                )
                if scenario in {"exact", "semantic"}
                else None,
            }
        )
    return rows


def build(inputs: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest()
    summary = {
        "schema_version": 1,
        "baseline_commit": "c389b6366a475d2ff5c8975666cc02ee19727176",
        "measurement_source_commit": "e7e7854c5e0109be3edef798a21045edc57ad234",
        "measurement_source_manifest": manifest,
        "analysis_source_sha256": digest(Path(__file__).read_bytes()),
        "independent_hardware": False,
        "semantic_policy_changed": False,
        "campaigns": {},
        "artifacts": {},
        "totals": Counter(),
    }
    for stage in ("primary", "ramp", "sustained", "gemma"):
        campaign = json.loads((inputs / stage / "campaign.json").read_text())
        if campaign["source_manifest"] != manifest:
            raise ValueError("Measurement sources differ from the recorded current sources")
        all_points, trials = [], []
        for trial in campaign["trials"]:
            path = ROOT / trial["raw_gzip_file"]
            with gzip.open(path, "rb") as compressed:
                raw = compressed.read()
            if digest(raw) != trial["raw_sha256"]:
                raise ValueError("Raw result digest mismatch")
            report = json.loads(raw)
            if report["code_manifest"] != manifest:
                raise ValueError("Child measurement source mismatch")
            if report.get("unavailable_points"):
                raise ValueError(
                    "Campaign has unavailable points; report them explicitly before packaging"
                )
            for point in report["results"]:
                if point["planned_requests"] != point["attempted"] + point["generator_dropped"]:
                    raise ValueError("Planned/attempted/local-drop mismatch")
                if len(point["raw_records"]) != point["planned_requests"]:
                    raise ValueError("Missing raw request records")
                if point["recovery"]["successful"] != 1:
                    raise ValueError("Recovery failed; do not publish a healthy release verdict")
                if point["process_ids"]["load_generator"] != trial["load_driver_pid"]:
                    raise ValueError("Driver PID provenance mismatch")
                for key in (
                    "planned_requests",
                    "attempted",
                    "completed",
                    "successful",
                    "generator_dropped",
                    "http_429",
                    "http_503",
                    "http_504",
                    "transport_failures",
                    "incorrect_responses",
                    "unexpected_http_responses",
                ):
                    summary["totals"][key] += point[key]
                all_points.append(point)
            destination = output / path.name
            shutil.copyfile(path, destination)
            summary["artifacts"][str(destination.relative_to(ROOT))] = {
                "gzip_sha256": digest(destination.read_bytes()),
                "raw_sha256": trial["raw_sha256"],
            }
            trials.append(
                {
                    "name": trial["name"],
                    "command": trial["command"],
                    "points": len(report["results"]),
                    "independent_upstream_calls": sum(
                        p["independent_upstream_calls"] for p in report["results"]
                    ),
                }
            )
        campaign_path = output / f"campaign-{stage}.json"
        campaign_path.write_text(json.dumps(campaign, indent=2) + "\n")
        summary["artifacts"][str(campaign_path.relative_to(ROOT))] = {
            "sha256": digest(campaign_path.read_bytes())
        }
        summary["campaigns"][stage] = {
            "started_utc": campaign["started_utc"],
            "finished_utc": campaign["finished_utc"],
            "points": len(all_points),
            "trials": trials,
            "hardware": campaign["hardware_and_software"],
            "comparison": aggregate(all_points),
        }
    for name in ("reliability.json", "model-runtime.json"):
        destination = output / name
        shutil.copyfile(inputs / name, destination)
        summary["artifacts"][str(destination.relative_to(ROOT))] = {
            "sha256": digest(destination.read_bytes())
        }
    expected_points = {"primary": 75, "ramp": 3, "sustained": 4, "gemma": 10}
    if any(summary["campaigns"][key]["points"] != value for key, value in expected_points.items()):
        raise ValueError("Final point count differs from the recorded M08 protocol")
    runtime = json.loads((inputs / "model-runtime.json").read_text())
    summary["real_model_measured_upstream_calls"] = sum(
        trial["independent_upstream_calls"] for trial in summary["campaigns"]["gemma"]["trials"]
    )
    summary["real_model_excluded_warmup_calls"] = 50
    summary["real_model_excluded_recovery_calls"] = 10
    if runtime["dedicated_log_chat_POSTs"] != summary["real_model_measured_upstream_calls"] + 60:
        raise ValueError("Whole Gemma log must reconcile measured, warm-up and recovery calls")
    try:
        summary["guest_cpu_model"] = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        summary["guest_cpu_model"] = None
    summary["totals"] = dict(summary["totals"])
    summary["limitations"] = [
        "Same host: shared two-core quota and 8 GiB; process separation is not CPU isolation.",
        "Closed-loop starts depend on responses; open-loop local drops never reached gateway.",
        "Semantic fallback can establish exact entries; inspect the actual hit mix.",
        "Gemma samples lack p95/p99 support; repeated prompts may reuse backend KV state.",
        "Incomplete resource windows have null CPU/RSS; brief samples cannot prove leak freedom.",
        "Trial ranges are descriptive, not confidence bounds or production capacity guarantees.",
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "artifacts/milestone08")
    parser.add_argument("--output-root", type=Path, default=ROOT / "benchmarks/milestone08")
    parser.add_argument("--capture-runtime", action="store_true")
    parser.add_argument("--ollama-log", type=Path)
    args = parser.parse_args()
    inputs, output = args.input_root.resolve(), args.output_root.resolve()
    if not inputs.is_relative_to(ROOT / "artifacts") or not output.is_relative_to(
        ROOT / "benchmarks"
    ):
        parser.error("Use project artifacts/ inputs and benchmarks/ outputs")
    if args.capture_runtime:
        log = args.ollama_log.resolve() if args.ollama_log else None
        if log is None or not log.is_relative_to(ROOT / "artifacts") or not log.is_file():
            parser.error("Runtime capture needs a dedicated existing --ollama-log under artifacts/")
        with httpx.Client(trust_env=False, timeout=2) as client:
            runtime = {}
            for name, route in (("version", "version"), ("tags", "tags"), ("loaded", "ps")):
                response = client.get("http://127.0.0.1:11434/api/" + route)
                response.raise_for_status()
                runtime[name] = response.json()
        runtime["dedicated_log_chat_POSTs"] = log_posts(log)
        inputs.mkdir(parents=True, exist_ok=True)
        (inputs / "model-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        print("Saved runtime identity and dedicated chat count; no inference requested.")
        return
    result = build(inputs, output)
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
