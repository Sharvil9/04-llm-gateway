"""Audit provenance and abstention claims without loading embedding models."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_gateway import make_settings

from devtools import semantic_audit
from devtools.semantic_eval import admitted_family, load_dataset
from devtools.smoke import ROOT
from llm_gateway.cache import ExactResponseCache
from llm_gateway.schemas import ChatRequest


def report(name="l3"):
    return json.loads((ROOT / f"evaluation/{name}_results.json").read_text())


def test_new_audit_is_frozen_and_every_phrase_is_absent_from_original_data(monkeypatch):
    audit = semantic_audit.load_audit_cases()
    original, _, registry = load_dataset()
    texts = {case["text"] for case in audit["cases"]}
    assert len(texts) == len(audit["cases"]) == 18
    assert not texts.intersection(registry)
    assert not texts.intersection(case["text"] for case in original["cases"])
    assert sum(case["expected_family"] is not None for case in audit["cases"]) == 6
    content = semantic_audit.AUDIT_DATASET.read_bytes()
    assert hashlib.sha256(content).hexdigest() == semantic_audit.AUDIT_SHA256
    monkeypatch.setattr(
        semantic_audit, "AUDIT_DATASET", SimpleNamespace(read_bytes=lambda: content + b"\n")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        semantic_audit.load_audit_cases()


def test_original_reports_reproduce_calibration_selection_but_not_unseen_admission():
    result = semantic_audit.audit_original_protocol()
    assert result["canonical_source_vectors"] == 3
    assert result["approval_registry_phrases"] == 21
    assert result["heldout_positive_phrases_preapproved"] == result["heldout_positive_cases"] == 9
    assert result["positive_phrases_preapproved"] == result["positive_cases"] == 18
    oracle = result["membership_only_oracle_without_embeddings"]
    assert oracle["true_hits"] == 18 and oracle["false_hits"] == 0
    assert oracle["precision"] == oracle["recall"] == 1
    assert len(result["metadata_only_negative_cases"]) == 5
    for model in ["l3", "l6"]:
        assert result["report_checks"][model]["selection"]["guarded"]["threshold"] == 0.75


def test_independent_threshold_recalculation_does_not_use_heldout_scores():
    original = report()
    modified = copy.deepcopy(original)
    for row in modified["case_scores"]:
        if row["split"] == "heldout":
            row["expected_family"] = None
            row["raw_similarity"] = 1.0
            row["guarded_similarity"] = 1.0
            row["guarded_family"] = "france-capital"
    for mode in ["raw", "guarded"]:
        assert semantic_audit.recompute_calibration_selection(original, mode) == (
            semantic_audit.recompute_calibration_selection(modified, mode)
        )


@pytest.mark.parametrize("mutation", ["threshold", "grid", "split", "label", "guard", "sources"])
def test_original_report_tampering_is_rejected(mutation):
    original, families, registry = load_dataset()
    changed = report()
    if mutation == "threshold":
        changed["selected"]["guarded"]["threshold"] = 0.90
    elif mutation == "grid":
        changed["threshold_candidates"].append(0.77)
    elif mutation == "split":
        changed["case_scores"][0]["split"] = "heldout"
    elif mutation == "label":
        changed["case_scores"][0]["expected_family"] = None
    elif mutation == "guard":
        changed["case_scores"][0]["guarded_family"] = None
    else:
        changed["source_entries"] = 21
    with pytest.raises(ValueError):
        semantic_audit.verify_original_report(changed, original, families, registry)


@pytest.fixture
def approved_cache():
    _, families, _ = load_dataset()
    settings = make_settings(
        backends={"local": {"base_url": "http://backend.invalid/v1", "revision": "backend-r1"}},
        models={"demo": {"backend": "local", "model": "physical-model", "revision": "weights-r1"}},
        exact_cache={"enabled": True},
        semantic_cache={
            "enabled": True,
            "families": [
                {
                    "id": family["id"],
                    "domain": family["scope"],
                    "answer_contract": family["response_requirement"],
                    "prompts": [family["canonical"]] + sum(family["paraphrases"].values(), []),
                }
                for family in families.values()
            ],
        },
    )
    return ExactResponseCache(settings)


def test_metadata_only_component_cases_need_explicit_system_context_at_runtime(approved_cache):
    original, _, registry = load_dataset()
    metadata_cases = [
        case
        for case in original["cases"]
        if case["text"] in registry and admitted_family(case, registry) is None
    ]
    assert len(metadata_cases) == 5
    for case in metadata_cases:
        bare = ChatRequest(
            model="demo", messages=[{"role": "user", "content": case["text"]}], seed=7
        )
        assert approved_cache.key_for(bare, "client-a", True, semantic=True) is not None
        mapped = bare.model_copy(
            update={
                "messages": ChatRequest(
                    model="demo", messages=semantic_audit.metadata_context_messages(case, registry)
                ).messages
            }
        )
        assert approved_cache.key_for(mapped, "client-a", True, semantic=True) is None


def test_real_production_eligibility_bypasses_every_new_audit_phrase_including_positives(
    approved_cache,
):
    audit = semantic_audit.load_audit_cases()
    equivalents_bypassed = 0
    for case in audit["cases"]:
        request = ChatRequest(
            model="demo", messages=[{"role": "user", "content": case["text"]}], seed=7
        )
        assert approved_cache.key_for(request, "client-a", True, semantic=True) is None
        equivalents_bypassed += case["expected_family"] is not None
    assert equivalents_bypassed == 6
    assert approved_cache.snapshot()["bypasses"] == 18


def test_raw_audit_metrics_expose_wrong_family_and_negative_acceptance():
    rows = [
        {"expected_family": "a", "raw_family": "a", "raw_similarity": 0.80},
        {"expected_family": "a", "raw_family": "b", "raw_similarity": 0.90},
        {"expected_family": None, "raw_family": "a", "raw_similarity": 0.95},
        {"expected_family": None, "raw_family": "a", "raw_similarity": 0.70},
    ]
    result = semantic_audit.retrieval_metrics(rows, 0.75, "raw")
    assert (result["true_hits"], result["positive_misses"], result["false_hits"]) == (1, 1, 2)
    assert result["negative_hits"] == 1
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
