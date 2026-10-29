"""Frozen dataset and evaluation-metric checks; no embedding model is loaded."""

from collections import Counter
from types import SimpleNamespace

import pytest

from devtools import semantic_eval


def score_row(expected, candidate, score, *, kind="positive", category="fixture"):
    return {
        "expected_family": expected,
        "raw_family": candidate,
        "raw_similarity": score,
        "guarded_family": candidate,
        "guarded_similarity": score,
        "kind": kind,
        "category": category,
    }


def test_frozen_dataset_has_balanced_splits_and_rejects_any_byte_change(monkeypatch):
    dataset, families, registry = semantic_eval.load_dataset()
    assert len(dataset["cases"]) == 54
    assert len(families) == 3 and len(registry) == 21
    assert Counter(case["split"] for case in dataset["cases"]) == {"calibration": 27, "heldout": 27}
    assert Counter(case["split"] for case in dataset["cases"] if case["expected_family"]) == {
        "calibration": 9,
        "heldout": 9,
    }
    for case in dataset["cases"]:
        if case["expected_family"]:
            assert case["text"] != families[case["expected_family"]]["canonical"]
    changed = semantic_eval.DATASET.read_bytes() + b"\n"
    monkeypatch.setattr(semantic_eval, "DATASET", SimpleNamespace(read_bytes=lambda: changed))
    with pytest.raises(ValueError, match="hash mismatch"):
        semantic_eval.load_dataset()


def test_finite_guard_rejects_unknown_text_and_mismatched_scope_or_response_contract():
    dataset, _, registry = semantic_eval.load_dataset()
    positive = next(case for case in dataset["cases"] if case["expected_family"])
    assert semantic_eval.admitted_family(positive, registry)["id"] == positive["expected_family"]
    for changes in [
        {"text": "An unreviewed question with seemingly similar wording"},
        {"scope": "private-session-v1"},
        {"response_requirement": "long-explanation-v1"},
    ]:
        assert semantic_eval.admitted_family(positive | changes, registry) is None


def test_calibration_threshold_selection_uses_highest_threshold_for_equal_safe_recall():
    calibration = [
        score_row("a", "a", 0.96),
        score_row("b", "b", 0.99),
        score_row(None, "a", 0.64, kind="negative"),
    ]
    selected = semantic_eval.select_threshold(calibration, "guarded")
    assert selected["selection_split"] == "calibration"
    assert selected["threshold"] == 0.95
    assert selected["calibration_metrics"]["queries"] == len(calibration)
    assert selected["calibration_metrics"]["recall"] == 1
    assert selected["calibration_metrics"]["false_hits"] == 0
    heldout = [score_row(None, "a", 0.99, kind="negative")]
    assert semantic_eval.metrics(heldout, selected["threshold"], "guarded")["false_hits"] == 1
    assert selected["threshold"] == 0.95  # Held-out failures do not retune the selected threshold.


def test_false_hit_metrics_use_explicit_distinct_denominators():
    rows = [
        score_row("a", "a", 0.99),
        score_row("b", "a", 0.99, category="wrong-family"),
        score_row(None, "a", 0.99, kind="negative", category="incorrect-admission"),
        score_row(None, None, None, kind="negative", category="bypass"),
        score_row("b", "b", 0.5, category="below-threshold"),
    ]
    result = semantic_eval.metrics(rows, 0.9, "guarded")
    assert (result["queries"], result["positives"], result["negatives"]) == (5, 3, 2)
    assert (result["true_hits"], result["false_hits"], result["positive_misses"]) == (1, 2, 2)
    assert (
        result["wrong_family_hits"] == result["negative_hits"] == result["admission_bypasses"] == 1
    )
    assert result["accepted_reuses"] == 3
    assert result["precision"] == result["recall"] == pytest.approx(1 / 3)
    assert result["false_reuse_fraction"] == pytest.approx(2 / 3)
    assert result["negative_acceptance_rate"] == pytest.approx(1 / 2)
    assert result["false_hits_per_all_queries"] == pytest.approx(2 / 5)
    empty = semantic_eval.metrics([], 0.9, "guarded")
    assert empty["precision"] is empty["recall"] is empty["false_reuse_fraction"] is None
    assert empty["negative_acceptance_rate"] is empty["false_hits_per_all_queries"] is None
