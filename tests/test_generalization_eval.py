"""Evaluator checks independent of the held-out case wording and model files."""

import copy
import json
from pathlib import Path

import pytest

from devtools import generalization_eval as evaluation
from devtools.semantic_eval import load_dataset
from llm_gateway.schemas import ChatRequest


def response(city="Paris", finish_reason="stop"):
    return {
        "id": "fixture-source",
        "object": "chat.completion",
        "created": 1,
        "model": "demo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": city},
                "finish_reason": finish_reason,
            }
        ],
    }


@pytest.mark.parametrize(
    ("content", "finish", "expected"),
    [
        ("Paris", "stop", True),
        (" Paris\n", "stop", True),
        ("Paris.", "stop", False),
        ('"Paris"', "stop", False),
        ("The answer is Paris", "stop", False),
        ("Tokyo", "stop", False),
        ("Paris", "length", False),
    ],
)
def test_suitability_checks_factual_content_and_output_contract(content, finish, expected):
    assert evaluation.suitable_city(response(content, finish), "Paris") is expected


def case():
    _, families, _ = load_dataset()
    return {
        "id": "test-development-fixture",
        "kind": "equivalent",
        "category": "test-only",
        "expected_family": "france-capital",
        "request": evaluation.canonical_request(families["france-capital"]),
        "headers": {"X-Gateway-Cache": "semantic"},
        "client": "client-a",
        "response_suitability": {"type": "city-name-only", "expected_city": "Paris"},
    }


def test_loader_accepts_wire_context_but_rejects_mismatched_oracle_and_final_hash(tmp_path):
    path = tmp_path / "cases.json"
    item = case()
    path.write_text(json.dumps({"cases": [item]}))
    dataset, digest = evaluation.load_cases(path)
    assert dataset["cases"] == [item]
    assert len(digest) == 64
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluation.load_cases(path, heldout=True)
    item["response_suitability"]["expected_city"] = "Tokyo"
    path.write_text(json.dumps({"cases": [item]}))
    with pytest.raises(ValueError, match="labels disagree"):
        evaluation.load_cases(path)


def test_negative_semantic_hit_is_false_even_if_city_matches():
    rows = [
        {
            "expected_family": "france-capital",
            "selected_source_family": "france-capital",
            "cache": "SEMANTIC_HIT",
            "response_suitable": True,
            "status": 200,
            "upstream_attempts": 0,
        },
        {
            "expected_family": None,
            "selected_source_family": "france-capital",
            "cache": "SEMANTIC_HIT",
            "response_suitable": None,
            "status": 200,
            "upstream_attempts": 0,
        },
    ]
    summary = evaluation.summarize(rows)
    assert summary["true_semantic_hits"] == summary["false_semantic_hits"] == 1
    assert summary["semantic_hit_precision"] == 0.5
    assert summary["valid_paraphrase_coverage"] == 1
    assert summary["negative_semantic_hits"] == 1


def test_genuine_sources_require_correct_requests_answers_and_independent_calls(tmp_path):
    _, families, _ = load_dataset()
    evidence = {
        "actual_upstream_calls": 2,
        "call_count_source": "test-owned-http-backend",
        "sources": {
            name: {
                "request": evaluation.canonical_request(families[name]),
                "response": response(city),
            }
            for name, city in evaluation.CITIES.items()
        },
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(evidence))
    loaded, _ = evaluation.load_gemma_sources(path, families)
    assert loaded == evidence
    wrong = copy.deepcopy(evidence)
    wrong["sources"]["france-capital"]["response"] = response("Paris.")
    path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="violates"):
        evaluation.load_gemma_sources(path, families)
    wrong = copy.deepcopy(evidence)
    wrong["actual_upstream_calls"] = 1
    path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="independent evidence"):
        evaluation.load_gemma_sources(path, families)


async def test_production_pipeline_uses_fixed_seeds_and_counts_cold_attempts():
    np = pytest.importorskip("numpy")

    class Encoder:
        fingerprint = "unit-test-only"
        max_tokens = 128

        def encode(self, text):
            vector = np.zeros(384, dtype=np.float32)
            vector[1 if "Japan" in text else 0] = 1
            return vector

    _, families, _ = load_dataset()
    first = case()
    first["request"]["messages"][0]["content"] = (
        "What is the capital city of France? Reply with only the city name."
    )
    second = copy.deepcopy(first)
    second.update(id="second-cold-fixture", client="client-b", expected_family=None)
    result = await evaluation.evaluate_phase(
        {"cases": [first, second]},
        Path("does-not-load-encoder-artifacts"),
        families,
        candidate=True,
        encoder=Encoder(),
    )
    assert result["seed_calls"] == 2
    assert result["case_results"][0]["cache"] == "SEMANTIC_HIT"
    assert result["case_results"][0]["production_lookup_observation"][0]["similarity"] == 1.0
    assert all(row["seed_candidate_entries"] == 2 for row in result["case_results"])
    assert result["case_results"][1]["upstream_attempts"] == 1
    assert result["case_results"][1]["cache"] == "MISS"
    assert result["total_fixture_backend_calls"] == result["total_gateway_upstream_attempts"]
    ChatRequest.model_validate(first["request"])


async def test_unknown_wire_alias_fails_instead_of_producing_all_404_evidence():
    _, families, _ = load_dataset()
    item = case()
    item["request"]["model"] = "not-configured"
    with pytest.raises(ValueError, match="wire model"):
        await evaluation.evaluate_phase(
            {"cases": [item]},
            Path("unused-model-directory"),
            families,
            candidate=True,
            encoder=None,
        )
