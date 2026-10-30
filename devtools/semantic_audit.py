"""Audit finite-membership claims and unseen wording without expanding cache eligibility.

Validate original reports and the new frozen set without loading a model:
  .venv/bin/python -m devtools.semantic_audit --validate-only
After committing the new set, score it once with the already-selected local L3:
  .venv/bin/python -m devtools.semantic_audit --output artifacts/semantic-audit.json
The threshold remains 0.75; these cases must never select or tune it.
"""

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from devtools.semantic_eval import FROZEN_SHA256, admitted_family, load_dataset, timing_summary
from devtools.smoke import ROOT

AUDIT_DATASET = ROOT / "evaluation/semantic_audit_cases.json"
AUDIT_SHA256 = "b819653ab102a268a1720db80e5f7719a56f9743b7eb305efe46bc68460d4661"
FIXED_THRESHOLD = 0.75
FIXED_GRID = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.0)


def load_audit_cases() -> dict:
    content = AUDIT_DATASET.read_bytes()
    if hashlib.sha256(content).hexdigest() != AUDIT_SHA256:
        raise ValueError("Frozen final audit hash mismatch; do not retune these cases")
    audit = json.loads(content)
    original, families, registry = load_dataset()
    if (
        audit["schema_version"] != 1
        or audit["dataset_version"] != "semantic-final-audit-v1"
        or audit["original_dataset_sha256"] != FROZEN_SHA256
        or audit["fixed_threshold"] != FIXED_THRESHOLD
    ):
        raise ValueError("Unexpected final audit protocol")
    previous = set(registry) | {case["text"] for case in original["cases"]}
    cases = audit["cases"]
    if (
        len(cases) != 18
        or len({case["id"] for case in cases}) != len(cases)
        or len({case["text"] for case in cases}) != len(cases)
        or any(not case["text"] or case["text"] in previous for case in cases)
    ):
        raise ValueError("Final audit requires unique wording absent from all original cases")
    for case in cases:
        if case["source_family"] not in families or case["expected_family"] not in {
            None,
            case["source_family"],
        }:
            raise ValueError("Invalid audit family label")
    if Counter(case["source_family"] for case in cases) != dict.fromkeys(families, 6):
        raise ValueError("Final audit family counts changed")
    if sum(case["expected_family"] is not None for case in cases) != 6:
        raise ValueError("Final audit positive count changed")
    return audit


def retrieval_metrics(rows: list[dict], threshold: float, mode: str) -> dict:
    """Independent count calculation for checking recorded selection arithmetic."""
    positives = sum(row["expected_family"] is not None for row in rows)
    true_hits = false_hits = negative_hits = bypasses = 0
    for row in rows:
        candidate = row[f"{mode}_family"]
        if candidate is None:
            bypasses += 1
        elif row[f"{mode}_similarity"] >= threshold:
            if candidate == row["expected_family"]:
                true_hits += 1
            else:
                false_hits += 1
                negative_hits += row["expected_family"] is None
    misses = positives - true_hits
    accepted = true_hits + false_hits
    return {
        "queries": len(rows),
        "positives": positives,
        "negatives": len(rows) - positives,
        "true_hits": true_hits,
        "positive_misses": misses,
        "false_hits": false_hits,
        "negative_hits": negative_hits,
        "admission_bypasses": bypasses,
        "accepted_reuses": accepted,
        "precision": true_hits / accepted if accepted else None,
        "recall": true_hits / positives if positives else None,
        "f1": 2 * true_hits / (2 * true_hits + false_hits + misses)
        if 2 * true_hits + false_hits + misses
        else 0.0,
    }


def recompute_calibration_selection(report: dict, mode: str) -> tuple[float, dict]:
    """Use only calibration rows, regardless of any held-out scores or labels."""
    rows = [row for row in report["case_scores"] if row["split"] == "calibration"]
    points = [(threshold, retrieval_metrics(rows, threshold, mode)) for threshold in FIXED_GRID]
    safe = [
        point
        for point in points
        if point[1]["false_hits"] == 0 and (mode != "raw" or point[1]["true_hits"] > 0)
    ]
    if safe:
        return max(safe, key=lambda point: (point[1]["recall"], point[0]))
    return max(points, key=lambda point: (point[1]["f1"], point[1]["precision"] or 0, point[0]))


def verify_original_report(report: dict, original: dict, families: dict, registry: dict) -> dict:
    if report["dataset_sha256"] != FROZEN_SHA256:
        raise ValueError("Original report dataset hash mismatch")
    if report["threshold_candidates"] != list(FIXED_GRID):
        raise ValueError("Original threshold grid changed")
    if report["source_entries"] != len(families):
        raise ValueError("Original source vector count changed")
    rows = {row["id"]: row for row in report["case_scores"]}
    if len(rows) != len(report["case_scores"]) or set(rows) != {
        case["id"] for case in original["cases"]
    }:
        raise ValueError("Original report case membership changed")
    for case in original["cases"]:
        row = rows[case["id"]]
        if any(
            row[field] != case[field] for field in ["split", "expected_family", "kind", "category"]
        ):
            raise ValueError("Original report labels/splits differ from frozen data")
        approved = admitted_family(case, registry)
        if row["guarded_family"] != (approved["id"] if approved else None):
            raise ValueError("Original report guard does not match reviewed membership")
        for mode in ["raw", "guarded"]:
            candidate, score = row[f"{mode}_family"], row[f"{mode}_similarity"]
            if candidate is None:
                if score is not None:
                    raise ValueError("A bypass cannot have a candidate score")
            elif candidate not in families or not math.isfinite(score) or not -1 <= score <= 1:
                raise ValueError("Invalid original candidate/score")
    selections = {}
    for mode in ["raw", "guarded"]:
        threshold, recomputed = recompute_calibration_selection(report, mode)
        recorded = report["selected"][mode]
        if recorded["selection_split"] != "calibration" or recorded["threshold"] != threshold:
            raise ValueError("Recorded threshold does not match calibration-only selection")
        if any(recorded["calibration_metrics"][key] != value for key, value in recomputed.items()):
            raise ValueError("Recorded calibration metrics do not match recalculation")
        selections[mode] = {"threshold": threshold, "calibration_only_recomputation_matches": True}
    return {"source_vectors": report["source_entries"], "selection": selections}


def metadata_context_messages(case: dict, registry: dict) -> list[dict]:
    """Make metadata-only context explicit; bare text cannot carry invisible fields."""
    family = registry.get(case["text"])
    messages = []
    if family and any(case[field] != family[field] for field in ["scope", "response_requirement"]):
        messages.append(
            {
                "role": "system",
                "content": (
                    f"Context: {case['scope']}. Required response: {case['response_requirement']}."
                ),
            }
        )
    return messages + [{"role": "user", "content": case["text"]}]


def audit_original_protocol() -> dict:
    original, families, registry = load_dataset()
    reports = {}
    for name in ["l3", "l6"]:
        report = json.loads((ROOT / f"evaluation/{name}_results.json").read_text())
        reports[name] = verify_original_report(report, original, families, registry)
    positives = [case for case in original["cases"] if case["expected_family"]]
    heldout = [case for case in positives if case["split"] == "heldout"]
    metadata_cases = [
        case
        for case in original["cases"]
        if case["text"] in registry and admitted_family(case, registry) is None
    ]
    oracle_rows = [
        {
            "expected_family": case["expected_family"],
            "oracle_family": family["id"] if (family := admitted_family(case, registry)) else None,
            "oracle_similarity": 1.0 if family else None,
        }
        for case in original["cases"]
    ]
    return {
        "dataset_sha256": FROZEN_SHA256,
        "report_checks": reports,
        "canonical_source_vectors": len(families),
        "approval_registry_phrases": len(registry),
        "positive_phrases_preapproved": sum(case["text"] in registry for case in positives),
        "positive_cases": len(positives),
        "heldout_positive_phrases_preapproved": sum(case["text"] in registry for case in heldout),
        "heldout_positive_cases": len(heldout),
        "membership_only_oracle_without_embeddings": retrieval_metrics(oracle_rows, 1.0, "oracle"),
        "metadata_only_negative_cases": [
            {
                "id": case["id"],
                "split": case["split"],
                "bare_text_family": registry[case["text"]]["id"],
                "component_policy_result": "BYPASS",
                "wire_mapping": "explicit system message, then original user text",
            }
            for case in metadata_cases
        ],
        "finding": "Threshold arithmetic reproduces calibration-only selection. However every "
        "held-out positive literal was already in the approval registry; guarded recall is not "
        "unseen-wording generalization. A membership-to-family lookup could reproduce the guarded "
        "admission outcomes without embeddings. Only three canonical source vectors were stored, "
        "so the raw cosine scores are genuine query-to-source retrieval, not cached query vectors.",
        "limits": "Recomputation cannot prove the absence of human inspection or pretrained-data "
        "overlap. The approval registry is not an independent semantic correctness oracle. "
        "Metadata-only component cases need explicit wire context; bare text alone stays eligible.",
    }


def evaluate_unseen(audit: dict, model_dir: Path, *, encoder=None) -> dict:
    """Score fixed new wording once; production membership remains unchanged."""
    _, families, registry = load_dataset()
    evidence = json.loads((ROOT / "evaluation/l3_results.json").read_text())
    actual_encoder = encoder is None
    if encoder is None:
        from llm_gateway.embeddings import LocalEmbedder

        encoder = LocalEmbedder(model_dir)
    if encoder.fingerprint != evidence["model_fingerprint"]:
        raise ValueError("Final audit requires the unchanged selected L3 pipeline fingerprint")

    def vector(text):
        values = [float(value) for value in encoder.encode(text)]
        if (
            len(values) != 384
            or any(not math.isfinite(value) for value in values)
            or abs(sum(value * value for value in values) - 1.0) > 0.001
        ):
            raise ValueError("Invalid audit embedding")
        return values

    sources = {family_id: vector(family["canonical"]) for family_id, family in families.items()}
    rows, elapsed_ms = [], []
    for case in audit["cases"]:
        started = time.perf_counter()
        query = vector(case["text"])
        elapsed = (time.perf_counter() - started) * 1000
        elapsed_ms.append(elapsed)
        scores = {
            family_id: max(-1.0, min(1.0, sum(a * b for a, b in zip(query, source, strict=True))))
            for family_id, source in sources.items()
        }
        best = max(scores, key=scores.get)
        if case["text"] in registry:
            raise ValueError("Audit wording was unexpectedly added to the approval registry")
        rows.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "category": case["category"],
                "expected_family": case["expected_family"],
                "raw_family": best,
                "raw_similarity": scores[best],
                "production_family": None,
                "production_similarity": None,
                "production_disposition": "BYPASS",
                "encode_ms": elapsed,
            }
        )
    encoding_timing = timing_summary(elapsed_ms)
    encoding_timing["p95_ms"] = None
    encoding_timing["note"] = (
        "Only 18 diagnostic query encodings; p95 is withheld below 100 observations. "
        "These samples measure neither HTTP latency nor throughput."
    )
    return {
        "dataset_version": audit["dataset_version"],
        "dataset_sha256": AUDIT_SHA256,
        "fixed_threshold": FIXED_THRESHOLD,
        "threshold_retuned": False,
        "actual_local_embeddings": actual_encoder,
        "model_fingerprint": encoder.fingerprint,
        "source_vectors": len(sources),
        "query_source_expansion": False,
        "raw_retrieval": retrieval_metrics(rows, FIXED_THRESHOLD, "raw"),
        "unchanged_production_guard": retrieval_metrics(rows, FIXED_THRESHOLD, "production"),
        "query_encoding_timing": encoding_timing,
        "case_scores": rows,
        "finding": "Production bypasses all unseen wording, including every genuine equivalent. "
        "Its zero false hits therefore do not demonstrate useful unseen-paraphrase recall. "
        "Raw retrieval is diagnostic only and cannot authorize cache serving.",
        "limitations": audit["protocol"]["limitations"],
        "answer_quality_evaluated": False,
        "throughput_benchmark": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/embeddings/l3"))
    parser.add_argument("--output", type=Path, help="Report file under repository artifacts/")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve() if args.output else None
    if output and (not output.is_relative_to(ROOT / "artifacts") or output == ROOT / "artifacts"):
        parser.error("--output must name a file under repository artifacts/")
    model_dir = (ROOT / args.model_dir).resolve()
    if not args.validate_only and (not model_dir.is_relative_to(ROOT) or not model_dir.is_dir()):
        parser.error("--model-dir must name an existing repository directory")
    audit = load_audit_cases()
    report = {"original_protocol_audit": audit_original_protocol()}
    if args.validate_only:
        report["new_audit_set"] = {
            "sha256": AUDIT_SHA256,
            "cases": len(audit["cases"]),
            "positives": sum(case["expected_family"] is not None for case in audit["cases"]),
            "all_phrases_unseen_by_original_registry": True,
            "embedding_model_loaded": False,
        }
    else:
        report["new_wording_audit"] = evaluate_unseen(audit, model_dir)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
