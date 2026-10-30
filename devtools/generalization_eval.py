"""Evaluate the production decision pipeline on separately frozen public fixtures.

The final holdout is never a tuning input. Each case starts with only the two
canonical source entries; evaluation queries cannot become later candidates.
ASGI transport exercises the real gateway boundary without claiming an external
HTTP benchmark. Optional captured Gemma source responses test actual answer
suitability on replay, not fresh model inference for every case.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import platform
import secrets
import time
import tomllib
from collections import Counter, OrderedDict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from devtools.semantic_eval import load_dataset
from devtools.smoke import ROOT
from llm_gateway.app import create_app
from llm_gateway.config import GatewayConfig, Settings
from llm_gateway.embeddings import LocalEmbedder, fingerprint_for_model
from llm_gateway.schemas import ChatRequest, ChatResponse

FIXED_THRESHOLD = 0.75
HOLDOUT_SHA256 = "ad05be1634d5a890e62a74eceedcff4e1c6b2cc5ec218fcff1230077f5684845"
CITIES = {"france-capital": "Paris", "japan-capital": "Tokyo"}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_manifest() -> dict[str, str]:
    files = [
        Path(__file__),
        ROOT / "uv.lock",
        ROOT / "config/ollama.toml",
        ROOT / "devtools/semantic_probe.py",
        *sorted((ROOT / "src/llm_gateway").glob("*.py")),
    ]
    live_probe = ROOT / "devtools/generalization_live.py"
    if live_probe.exists():
        files.append(live_probe)
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in files}


def load_cases(path: Path, *, heldout: bool = False) -> tuple[dict, str]:
    content = path.read_bytes()
    digest = sha256(content)
    if heldout and digest != HOLDOUT_SHA256:
        raise ValueError("Frozen final holdout hash mismatch")
    dataset = json.loads(content)
    cases = dataset["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 256:
        raise ValueError("Expected a bounded evaluation case list")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique")
    for case in cases:
        ChatRequest.model_validate(case["request"])
        if case["expected_family"] not in {None, *CITIES}:
            raise ValueError("Unexpected equivalence label")
        if case["client"] not in {"client-a", "client-b"}:
            raise ValueError("Unexpected fixture client")
        if not isinstance(case["headers"], (dict, list)):
            raise ValueError("Headers must represent real wire inputs")
        family = case["expected_family"]
        if family is not None and case["response_suitability"]["expected_city"] != CITIES[family]:
            raise ValueError("Equivalence and answer-suitability labels disagree")
    return dataset, digest


def canonical_request(family: dict) -> dict:
    return {
        "model": "gemma",
        "messages": [{"role": "user", "content": family["canonical"]}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32,
        "seed": 42,
        "stream": False,
    }


def suitable_city(response: dict, city: str) -> bool:
    """Independent factual/output oracle; replay byte equality alone is insufficient."""
    try:
        normalized = ChatResponse.model_validate(response)
    except ValueError:
        return False
    choice = normalized.choices[0]
    return choice.finish_reason == "stop" and choice.message.content.strip() == city


def load_gemma_sources(path: Path, families: dict) -> tuple[dict, str]:
    content = path.read_bytes()
    evidence = json.loads(content)
    if evidence["actual_upstream_calls"] != 2 or not evidence.get("call_count_source"):
        raise ValueError("Require independent evidence of two genuine source calls")
    for name, city in CITIES.items():
        item = evidence["sources"][name]
        expected = canonical_request(families[name])
        source_request = ChatRequest.model_validate(item["request"]).model_dump(mode="json")
        # Public alias may differ in the separately recorded genuine HTTP run.
        source_request["model"] = "gemma"
        if source_request != expected or not suitable_city(item["response"], city):
            raise ValueError("Genuine source request or answer violates the fixed contract")
    return evidence, sha256(content)


def settings_for(
    model_dir: Path, families: dict, *, candidate: bool, real_replay: bool
) -> Settings:
    if real_replay:
        values = tomllib.loads((ROOT / "config/ollama.toml").read_text())
        values["models"] = {"gemma": values["models"]["gemma"]}
    else:
        values = {
            "backends": {
                "local": {"base_url": "http://backend.invalid/v1", "revision": "fixture-v1"}
            },
            "models": {"gemma": {"backend": "local", "model": "fixture", "revision": "fixture-v1"}},
        }
    values.update(
        rate_limit={"burst": 10000, "requests_per_second": 10000.0},
        max_in_flight=1,
        exact_cache={
            "enabled": True,
            "ttl_seconds": 3600.0,
            "max_entries": 16,
            "max_bytes": 1048576,
            "namespace": "m07-generalization-evaluation-v1",
        },
        semantic_cache={
            "enabled": True,
            "model_dir": str(model_dir),
            "threshold": FIXED_THRESHOLD,
            "timeout_seconds": 0.25,
            "max_entries": 16,
            "capital_entities": ["France", "Japan"] if candidate else [],
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
    return Settings(
        config=GatewayConfig.model_validate(values),
        client_key_hashes={
            hashlib.sha256(key.encode()).digest(): name for name, key in KEYS.items()
        },
    )


# Generated per evaluator process, never emitted or committed as credentials.
KEYS = {name: secrets.token_urlsafe(32) for name in ("client-a", "client-b")}


class ReplayBackend:
    """Count fixture forwards; only canonical responses can become source entries."""

    def __init__(self, families: dict, gemma: dict | None = None):
        self.families = families
        self.gemma = gemma
        self.calls = 0
        self.source_calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        incoming = json.loads(request.content)
        source = next(
            (
                name
                for name in CITIES
                if incoming["messages"]
                == [{"role": "user", "content": self.families[name]["canonical"]}]
            ),
            None,
        )
        if source:
            self.source_calls += 1
            if self.gemma:
                response = copy.deepcopy(self.gemma["sources"][source]["response"])
                response["model"] = incoming["model"]
                return httpx.Response(200, json=response)
        return httpx.Response(
            200,
            headers={} if source else {"Cache-Control": "no-store"},
            json={
                "id": f"evaluation-fixture-{self.calls}",
                "object": "chat.completion",
                "created": 1,
                "model": incoming["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": CITIES[source] if source else "Uncached evaluation fixture",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )


def summarize(rows: list[dict]) -> dict:
    positives = [row for row in rows if row["expected_family"] is not None]
    semantic = [row for row in rows if row["cache"] == "SEMANTIC_HIT"]
    true = [
        row
        for row in semantic
        if row["expected_family"] == row["selected_source_family"]
        and row["response_suitable"] is True
    ]
    false = len(semantic) - len(true)
    return {
        "queries": len(rows),
        "equivalents": len(positives),
        "negatives": len(rows) - len(positives),
        "semantic_hits": len(semantic),
        "true_semantic_hits": len(true),
        "false_semantic_hits": false,
        "semantic_hit_precision": len(true) / len(semantic) if semantic else None,
        "valid_paraphrase_coverage": len(true) / len(positives) if positives else None,
        "equivalent_misses_or_bypasses": len(positives) - len(true),
        "equivalent_bypasses": sum(row["cache"] == "BYPASS" for row in positives),
        "negative_semantic_hits": sum(row["expected_family"] is None for row in semantic),
        "bypasses": sum(row["cache"] == "BYPASS" for row in rows),
        "http_rejections": sum(row["status"] != 200 for row in rows),
        "cache_dispositions": dict(Counter(row["cache"] for row in rows)),
        "http_statuses": dict(Counter(str(row["status"]) for row in rows)),
        "fixture_upstream_attempts": sum(row["upstream_attempts"] for row in rows),
        "unsuitable_semantic_answers": sum(row["response_suitable"] is False for row in semantic),
        "all_semantic_answers_suitable": all(row["response_suitable"] is True for row in semantic),
        "precision_denominator": "All SEMANTIC_HIT responses, including policy-negative hits",
        "coverage_denominator": (
            "All labeled equivalent wire requests, including conservative bypasses"
        ),
    }


async def evaluate_phase(
    dataset: dict,
    model_dir: Path,
    families: dict,
    *,
    candidate: bool,
    encoder,
    gemma: dict | None = None,
) -> dict:
    settings = settings_for(model_dir, families, candidate=candidate, real_replay=gemma is not None)
    unknown_models = {case["request"]["model"] for case in dataset["cases"]} - set(
        settings.config.models
    )
    if unknown_models:
        raise ValueError("Fixture wire model is absent from the evaluator configuration")
    backend = ReplayBackend(families, gemma)
    app = create_app(settings, transport=httpx.MockTransport(backend), embedding_encoder=encoder)
    rows = []
    async with app.router.lifespan_context(app):
        lookup_trace = []
        original_match = app.state.semantic_cache.embedder.match

        async def observed_match(text, candidates, threshold):
            # Observe the real bounded worker's return without changing it or
            # feeding evaluation labels into its decision.
            result = await original_match(text, candidates, threshold)
            lookup_trace.append(
                {
                    "compatible_candidates": len(candidates),
                    "similarity": result[2] if result is not None else None,
                    "above_threshold_candidate": result[1] is not None
                    if result is not None
                    else None,
                    "worker_busy": result is None,
                }
            )
            return result

        app.state.semantic_cache.embedder.match = observed_match
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as client:
            seed_hashes, seed_rows = {}, []
            for name in CITIES:
                response = await client.post(
                    "/v1/chat/completions",
                    json=canonical_request(families[name]),
                    headers={
                        "Authorization": f"Bearer {KEYS['client-a']}",
                        "X-Gateway-Cache": "semantic",
                    },
                )
                response.raise_for_status()
                if response.headers.get("x-gateway-cache") != "MISS":
                    raise ValueError("Canonical seeds must start cold")
                if not suitable_city(response.json(), CITIES[name]):
                    raise ValueError("Seed source answer violates the factual/output oracle")
                seed_hashes[sha256(response.content)] = name
                seed_rows.append({"family": name, "response": response.json(), "cache": "MISS"})
            cache = app.state.response_cache
            if len(cache._entries) != 2 or any(
                entry.semantic is None for entry in cache._entries.values()
            ):
                raise ValueError("Expected exactly two successfully tagged canonical sources")
            source_entries = OrderedDict(cache._entries)
            source_bytes = cache._bytes_used
            seed_calls = backend.calls
            for case in dataset["cases"]:
                # Testfixture-only reset: no final query becomes a later source.
                # Immutable entry bytes, fixed HMAC secret and insertion TTL are retained.
                cache._entries = OrderedDict(source_entries)
                cache._bytes_used = source_bytes
                if len(cache._entries) != 2:
                    raise ValueError("Canonical candidate snapshot changed")
                request = copy.deepcopy(case["request"])
                headers = (
                    list(case["headers"].items())
                    if isinstance(case["headers"], dict)
                    else list(case["headers"])
                )
                if any(name.lower() == "authorization" for name, _ in headers):
                    raise ValueError("Evaluation fixtures must not contain raw credentials")
                headers.append(("Authorization", f"Bearer {KEYS[case['client']]}"))
                lookup_trace.clear()
                before_calls = backend.calls
                before_attempts = cache.stats.upstream_calls
                before_semantic = app.state.semantic_cache.snapshot()
                started = time.perf_counter()
                response = await client.post("/v1/chat/completions", json=request, headers=headers)
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code != 200:
                    raise ValueError(
                        f"Unexpected evaluation HTTP {response.status_code} for case {case['id']}"
                    )
                attempts = cache.stats.upstream_calls - before_attempts
                if attempts != backend.calls - before_calls:
                    raise ValueError(
                        "Gateway attempts and independently counted fixture calls disagree"
                    )
                family = cache.family_for(ChatRequest.model_validate(request))
                disposition = response.headers.get("x-gateway-cache", "EARLY")
                body = response.json()
                selected = (
                    seed_hashes.get(sha256(response.content))
                    if disposition in {"HIT", "SEMANTIC_HIT"}
                    else None
                )
                expected = case["expected_family"]
                suitable = (
                    suitable_city(body, CITIES[expected])
                    if expected and disposition in {"HIT", "SEMANTIC_HIT"}
                    else None
                )
                rows.append(
                    {
                        "id": case["id"],
                        "kind": case["kind"],
                        "category": case["category"],
                        "expected_family": expected,
                        "status": response.status_code,
                        "cache": disposition,
                        "parsed_or_reviewed_family": family.id if family else None,
                        "production_lookup_observation": copy.deepcopy(lookup_trace),
                        "selected_source_family": selected,
                        "response_suitable": suitable,
                        "assistant_text": body.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content"),
                        "upstream_attempts": attempts,
                        "fixture_backend_calls": backend.calls - before_calls,
                        "semantic_stage_delta": {
                            key: value - before_semantic[key]
                            for key, value in app.state.semantic_cache.snapshot().items()
                        },
                        "client_asgi_latency_ms": elapsed,
                        "server_timing": response.headers.get("server-timing"),
                        "seed_candidate_entries": 2,
                    }
                )
            if app.state.admission.active:
                raise ValueError("Evaluation leaked an inference capacity permit")
            total_attempts = cache.stats.upstream_calls
            if total_attempts != backend.calls:
                raise ValueError("Total gateway attempts and fixture calls disagree")
    return {
        "policy": "capital-grammar-candidate" if candidate else "reviewed-phrase-baseline",
        "configuration": settings.config.model_dump(mode="json"),
        "summary": summarize(rows),
        "seed_calls": seed_calls,
        "total_fixture_backend_calls": backend.calls,
        "total_gateway_upstream_attempts": total_attempts,
        "seed_sources": seed_rows,
        "case_results": rows,
    }


async def run(path: Path, model_dir: Path, *, heldout: bool, gemma_path: Path | None) -> dict:
    manifest = await asyncio.to_thread(source_manifest)
    dataset, dataset_digest = await asyncio.to_thread(load_cases, path, heldout=heldout)
    _, families, _ = load_dataset()
    gemma, gemma_digest = (
        await asyncio.to_thread(load_gemma_sources, gemma_path, families)
        if gemma_path
        else (None, None)
    )
    fingerprint = await asyncio.to_thread(fingerprint_for_model, model_dir)
    encoder = await asyncio.to_thread(LocalEmbedder, model_dir)
    before = datetime.now(UTC).isoformat()
    phases = {}
    for candidate in (False, True):
        name = "candidate" if candidate else "baseline"
        phases[name] = await evaluate_phase(
            dataset, model_dir, families, candidate=candidate, encoder=encoder, gemma=gemma
        )
    current_manifest = await asyncio.to_thread(source_manifest)
    current_content = await asyncio.to_thread(path.read_bytes)
    if current_manifest != manifest or sha256(current_content) != dataset_digest:
        raise ValueError("Evaluation source/dataset changed during scoring")
    return {
        "dataset": str(path.relative_to(ROOT)),
        "dataset_sha256": dataset_digest,
        "heldout": heldout,
        "fixed_threshold": FIXED_THRESHOLD,
        "threshold_tuned": False,
        "source_manifest": manifest,
        "started_at_utc": before,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "hardware": {"platform": platform.platform(), "python": platform.python_version()},
        "transport": "production ASGI boundary; counted MockTransport backend",
        "actual_local_embeddings": True,
        "embedding_fingerprint": fingerprint,
        "query_source_expansion": False,
        "source_vectors_per_case": 2,
        "source_reset": "Restore only original canonical immutable entries before each case",
        "fresh_model_inference_during_scoring": False,
        "genuine_model_source_replay": gemma is not None,
        "genuine_source_sha256": gemma_digest,
        "genuine_source_evidence": gemma,
        "noncanonical_inference": (
            "No-store deterministic fixture; cold answers are not model-quality evidence"
        ),
        "suitability_scope": (
            "Cached equivalent answers only: exact Paris/Tokyo after surrounding whitespace trim"
        ),
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/embeddings/l3")
    parser.add_argument("--gemma-sources", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = ROOT / f"evaluation/generalization_{args.dataset}.json"
    result = asyncio.run(
        run(
            path,
            args.model_dir.resolve(),
            heldout=args.dataset == "holdout",
            gemma_path=args.gemma_sources,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({name: phase["summary"] for name, phase in result["phases"].items()}, indent=2)
    )


if __name__ == "__main__":
    main()
