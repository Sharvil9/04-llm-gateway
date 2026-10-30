"""Capture genuine Gemma sources and verify unseen replay over loopback HTTP.

Requires the dedicated quiet Ollama from README; never installs a model. Public
fixture prompts/answers are saved intentionally, outside production logging.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx

from devtools.semantic_eval import load_dataset
from devtools.semantic_probe import log_posts, server_timings, servers
from devtools.smoke import ROOT


def run(model_dir: Path, log: Path) -> dict:
    _, families, _ = load_dataset()
    with servers(model_dir, 0.75, families, True, capital_entities=("France", "Japan")) as (
        backend,
        gateway,
        keys,
        config,
    ):
        with httpx.Client(trust_env=False, timeout=70) as client:
            version = client.get(backend + "/api/version").json()["version"]
            assert version == "0.40.11"
            tags = client.get(backend + "/api/tags").json()["models"]
            model = next(row for row in tags if row["name"] == "gemma3:1b")
            digest = "sha256:" + model["digest"].removeprefix("sha256:")
            assert digest == config["models"]["gemma"]["revision"]
            before = log_posts(log)
            rows, sources = [], {}

            def request(text, expected, *, city=None, owner="client-a"):
                body = {
                    "model": "gemma",
                    "messages": [{"role": "user", "content": text}],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 42,
                    "max_tokens": 32,
                    "stream": False,
                }
                response = client.post(
                    gateway + "/v1/chat/completions",
                    json=body,
                    headers={
                        "Authorization": "Bearer " + keys[owner],
                        "X-Gateway-Cache": "semantic",
                    },
                )
                assert response.status_code == 200
                assert response.headers["x-gateway-cache"] == expected
                answer = response.json()["choices"][0]["message"]["content"]
                if city:
                    assert answer.strip() == city
                rows.append(
                    {
                        "request": body,
                        "client": owner,
                        "cache": expected,
                        "status": response.status_code,
                        "assistant_text": answer,
                        "expected_city": city,
                        "answer_suitable": answer.strip() == city if city else None,
                        "server_timing_ms": server_timings(response),
                        "response_sha256": hashlib.sha256(response.content).hexdigest(),
                    }
                )
                return body, response.json()

            for family, city in (("france-capital", "Paris"), ("japan-capital", "Tokyo")):
                body, response = request(families[family]["canonical"], "MISS", city=city)
                sources[family] = {"request": body, "response": response}
            # Wait for independent access logging, rather than trusting gateway counters.
            for _ in range(10):
                if log_posts(log) - before >= 2:
                    break
                time.sleep(0.1)
            assert log_posts(log) - before == 2
            source_calls = log_posts(log) - before
            loaded = next(
                row
                for row in client.get(backend + "/api/ps").json()["models"]
                if row["name"] == "gemma3:1b"
            )
            assert loaded["context_length"] == 2048
            assert "sha256:" + loaded["digest"].removeprefix("sha256:") == digest

            # Development wording only; never choose probes from final-case scores.
            for entity, city in (("France", "Paris"), ("Japan", "Tokyo")):
                request(
                    f"Please state the capital city of {entity}. Return just the city name.",
                    "SEMANTIC_HIT",
                    city=city,
                )
            request(
                "What is the capital of Germany? Answer with only the city name.",
                "BYPASS",
                city="Berlin",
            )
            request(families["france-capital"]["canonical"], "MISS", city="Paris", owner="client-b")
            for _ in range(10):
                if log_posts(log) - before >= 4:
                    break
                time.sleep(0.1)
            assert log_posts(log) - before == 4
            return {
                "real_model_inference": True,
                "transport": "loopback_http",
                "model_identity": {
                    "backend_version": version,
                    "model": "gemma3:1b",
                    "model_digest": digest,
                    "context_length": 2048,
                    "hardware": "CPU-only Linux, 8 GiB, two-core cgroup quota",
                },
                "actual_upstream_calls": source_calls,
                "call_count_source": "dedicated_quiet_ollama_gin_log",
                "live_probe_total_upstream_calls": 4,
                "live_probe_requests": len(rows),
                "sources": sources,
                "live_probe_rows": rows,
                "answer_oracle": "Exact city name; trim surrounding whitespace only",
                "limitations": (
                    "Two static facts; no general answer-quality or language-equivalence guarantee."
                ),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/embeddings/l3")
    parser.add_argument("--ollama-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model_dir, args.ollama_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "live_requests": result["live_probe_requests"],
                "actual_upstream_calls": result["live_probe_total_upstream_calls"],
            }
        )
    )


if __name__ == "__main__":
    main()
