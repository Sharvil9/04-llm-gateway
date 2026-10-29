"""Evaluate frozen semantic retrieval cases with one local embedding model.

Validate without loading a model:
  .venv/bin/python -m devtools.semantic_eval --validate-only
Evaluate one already-downloaded model (run models sequentially):
  .venv/bin/python -m devtools.semantic_eval --model-dir artifacts/models/MODEL \
    --output artifacts/semantic-MODEL.json
"""

import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

from devtools.smoke import ROOT

DATASET = ROOT / "evaluation/semantic_cases.json"
FROZEN_SHA256 = "80e0ba7b0acaf85afaca2e9bd8fa468c426fe67d50acdfa07541bb98d1a04885"
THRESHOLDS = (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0)


def admitted_family(case: dict, registry: dict) -> dict | None:
    family = registry.get(case["text"])
    if family is None or any(
        case[field] != family[field] for field in ("scope", "response_requirement")
    ):
        return None
    return family


def load_dataset() -> tuple[dict, dict, dict]:
    content = DATASET.read_bytes()
    if hashlib.sha256(content).hexdigest() != FROZEN_SHA256:
        raise ValueError("Frozen semantic-cases-v1 hash mismatch; do not retune heldout cases")
    dataset = json.loads(content)
    if dataset["schema_version"] != 1 or dataset["dataset_version"] != "semantic-cases-v1":
        raise ValueError("Unsupported frozen semantic dataset version")
    families, registry = {}, {}
    for family in dataset["families"]:
        if family["id"] in families:
            raise ValueError("Duplicate family ID")
        families[family["id"]] = family
        phrases = [family["canonical"]] + sum(family["paraphrases"].values(), [])
        for phrase in phrases:
            if not phrase or phrase in registry:
                raise ValueError("Reviewed phrase membership must be unambiguous")
            registry[phrase] = family
    case_ids = set()
    for case in dataset["cases"]:
        if case["id"] in case_ids or case["split"] not in {"calibration", "heldout"}:
            raise ValueError("Case IDs/splits are invalid")
        case_ids.add(case["id"])
        family = admitted_family(case, registry)
        observed = family["id"] if family else None
        if observed != case["expected_family"]:
            raise ValueError("Labels conflict with the frozen finite admission policy")
        if case["expected_family"] and case["text"] == family["canonical"]:
            raise ValueError("Positive evaluation queries must be non-identical paraphrases")
    return dataset, families, registry


def metrics(rows: list[dict], threshold: float, mode: str) -> dict:
    positives = sum(row["expected_family"] is not None for row in rows)
    true_hits = false_hits = negative_hits = wrong_family_hits = bypasses = 0
    failures = Counter()
    for row in rows:
        candidate, score = row[f"{mode}_family"], row[f"{mode}_similarity"]
        if candidate is None:
            bypasses += 1
            continue
        if score < threshold:
            continue
        if candidate == row["expected_family"]:
            true_hits += 1
        else:
            false_hits += 1
            failures[row["kind"] + ":" + row["category"]] += 1
            if row["expected_family"] is None:
                negative_hits += 1
            else:
                wrong_family_hits += 1
    accepted = true_hits + false_hits
    misses = positives - true_hits
    negatives = len(rows) - positives
    return {
        "queries": len(rows),
        "positives": positives,
        "negatives": negatives,
        "true_hits": true_hits,
        "positive_misses": misses,
        "false_hits": false_hits,
        "negative_hits": negative_hits,
        "wrong_family_hits": wrong_family_hits,
        "admission_bypasses": bypasses,
        "accepted_reuses": accepted,
        "precision": true_hits / accepted if accepted else None,
        "recall": true_hits / positives if positives else None,
        "false_reuse_fraction": false_hits / accepted if accepted else None,
        "negative_acceptance_rate": negative_hits / negatives if negatives else None,
        "false_hits_per_all_queries": false_hits / len(rows) if rows else None,
        "f1": 2 * true_hits / (2 * true_hits + false_hits + misses)
        if 2 * true_hits + false_hits + misses
        else 0.0,
        "false_hits_by_kind_and_category": dict(failures),
    }


def select_threshold(calibration: list[dict], mode: str) -> dict:
    points = [(threshold, metrics(calibration, threshold, mode)) for threshold in THRESHOLDS]
    viable = [(threshold, point) for threshold, point in points if point["false_hits"] == 0]
    if mode == "raw":
        viable = [(threshold, point) for threshold, point in viable if point["true_hits"] > 0]
    if viable:
        threshold, point = max(viable, key=lambda item: (item[1]["recall"], item[0]))
        rule = "maximum calibration recall at zero false hits; highest-threshold tie break"
    else:
        threshold, point = max(
            points, key=lambda item: (item[1]["f1"], item[1]["precision"] or 0, item[0])
        )
        rule = "no nonzero-recall zero-false-hit point; best calibration F1/precision tradeoff"
    return {
        "threshold": threshold,
        "selection_split": "calibration",
        "selection_rule": rule,
        "calibration_metrics": point,
    }


def timing_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "total_ms": sum(values),
    }


def process_memory() -> dict:
    """Measure this process, including Python/libraries, rather than model allocation."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if platform.system() == "Darwin" else peak * 1024)
    current_bytes = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                current_bytes = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    return {"current_rss_bytes": current_bytes, "peak_rss_bytes": peak_bytes}


def evaluate(model_dir: Path, dataset: dict, families: dict, registry: dict) -> dict:
    # Import only here: --validate-only requires no ONNX/model runtime or weights.
    baseline_memory = process_memory()
    started = time.perf_counter()
    from llm_gateway.embeddings import LocalEmbedder

    embedder = LocalEmbedder(model_dir)
    load_ms = (time.perf_counter() - started) * 1000
    loaded_memory = process_memory()
    source_vectors, source_ms, dimensions = {}, [], set()

    def encode(text: str) -> tuple[list[float], float]:
        start = time.perf_counter()
        vector = embedder.encode(text)
        elapsed = (time.perf_counter() - start) * 1000
        if getattr(vector, "ndim", None) != 1:
            raise ValueError("LocalEmbedder must return a one-dimensional normalized vector")
        values = [float(value) for value in vector]
        dimensions.add(len(values))
        if (
            not values
            or any(not math.isfinite(value) for value in values)
            or abs(sum(value * value for value in values) - 1.0) > 0.001
        ):
            raise ValueError("LocalEmbedder returned an invalid/non-normalized vector")
        return values, elapsed

    for family_id, family in families.items():
        source_vectors[family_id], elapsed = encode(family["canonical"])
        source_ms.append(elapsed)

    rows, encode_ms, lookup_ms = [], [], []
    for case in dataset["cases"]:
        vector, elapsed = encode(case["text"])
        encode_ms.append(elapsed)
        start = time.perf_counter()
        similarities = {
            family_id: max(-1.0, min(1.0, sum(a * b for a, b in zip(vector, source, strict=True))))
            for family_id, source in source_vectors.items()
        }
        raw_family = max(similarities, key=similarities.get)
        family = admitted_family(case, registry)
        guarded_family = family["id"] if family else None
        lookup_ms.append((time.perf_counter() - start) * 1000)
        rows.append(
            {
                "id": case["id"],
                "split": case["split"],
                "expected_family": case["expected_family"],
                "kind": case["kind"],
                "category": case["category"],
                "raw_family": raw_family,
                "raw_similarity": similarities[raw_family],
                "guarded_family": guarded_family,
                "guarded_similarity": similarities[guarded_family] if guarded_family else None,
                "encode_ms": elapsed,
                "token_count": embedder.token_count(case["text"])
                if hasattr(embedder, "token_count")
                else None,
            }
        )
    if len(dimensions) != 1:
        raise ValueError("Embedding vector dimensions changed within the evaluation")
    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    selections = {mode: select_threshold(calibration, mode) for mode in ("raw", "guarded")}
    packages = {}
    for name in ("numpy", "onnxruntime", "tokenizers"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    final_memory = process_memory()
    return {
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": FROZEN_SHA256,
        "model_directory": str(model_dir.relative_to(ROOT)),
        "model_fingerprint": embedder.fingerprint,
        "max_tokens": embedder.max_tokens,
        "embedding_dimensions": next(iter(dimensions)),
        "source_entries": len(families),
        "source_policy": "one canonical cached source per family; no query-source expansion",
        "threshold_candidates": THRESHOLDS,
        "selected": selections,
        "heldout_at_selected_thresholds": {
            mode: metrics(heldout, selections[mode]["threshold"], mode) for mode in selections
        },
        "calibration_curve": [
            {
                "threshold": threshold,
                **{mode: metrics(calibration, threshold, mode) for mode in selections},
            }
            for threshold in THRESHOLDS
        ],
        "timing": {
            "model_load_ms": load_ms,
            "canonical_encoding": timing_summary(source_ms),
            "query_encoding": timing_summary(encode_ms),
            "similarity_and_guard_lookup": timing_summary(lookup_ms),
            "conditions": "sequential batch-size-one CPU calls; queries follow canonical encodings",
            "p95_definition": "nearest-rank p95",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": packages,
        },
        "process_memory": {
            "baseline_before_model_import": baseline_memory,
            "after_model_load": loaded_memory,
            "after_evaluation": final_memory,
            "load_rss_delta_bytes": loaded_memory["current_rss_bytes"]
            - baseline_memory["current_rss_bytes"]
            if loaded_memory["current_rss_bytes"] is not None
            and baseline_memory["current_rss_bytes"] is not None
            else None,
            "load_peak_rss_delta_bytes": loaded_memory["peak_rss_bytes"]
            - baseline_memory["peak_rss_bytes"],
            "final_peak_process_rss_bytes": final_memory["peak_rss_bytes"],
            "note": "Process RSS includes Python, libraries and model/runtime allocations; "
            "load deltas are not isolated model memory. Current RSS samples use Linux VmRSS; "
            "peak ru_maxrss is converted from Linux KiB or macOS bytes.",
        },
        "limitations": dataset["protocol"]["limitations"],
        "metric_note": "false_reuse_fraction divides false hits by accepted reuses; "
        "negative_acceptance_rate divides accepted negatives by all negatives; "
        "policy exclusions need not imply factually wrong answers",
        "scope_note": "Guard filters reviewed text by family, scope and response contract; "
        "raw retrieval ignores those boundaries. This component evaluation establishes neither "
        "client isolation nor answer quality.",
        "throughput_or_model_generation_claim": False,
        "case_scores": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, help="Existing model directory in this repository"
    )
    parser.add_argument(
        "--output", type=Path, help="JSON report under this repository's artifacts/"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Check frozen data without a model"
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve() if args.output else None
    if output and (not output.is_relative_to(ROOT / "artifacts") or output == ROOT / "artifacts"):
        parser.error("--output must name a file inside this repository's artifacts/")
    model_dir = (ROOT / args.model_dir).resolve() if args.model_dir else None
    if not args.validate_only and (
        model_dir is None or not model_dir.is_relative_to(ROOT) or not model_dir.is_dir()
    ):
        parser.error("--model-dir must name an existing directory inside this repository")
    dataset, families, registry = load_dataset()
    if args.validate_only:
        report = {
            "dataset_version": dataset["dataset_version"],
            "dataset_sha256": FROZEN_SHA256,
            "families": len(families),
            "cases": len(dataset["cases"]),
            "splits": dict(Counter(case["split"] for case in dataset["cases"])),
            "positives": sum(case["expected_family"] is not None for case in dataset["cases"]),
            "embedding_model_loaded": False,
        }
    else:
        report = evaluate(model_dir, dataset, families, registry)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"case_scores", "calibration_curve"}
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
