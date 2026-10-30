"""Verify guarded semantic reuse over loopback HTTP with one gateway encoder.

Run against a disposable deterministic backend:
  .venv/bin/python -m devtools.semantic_probe --output artifacts/semantic-probe.json
For genuine inference, supply --ollama-log artifacts/.../ollama.log and an
existing dedicated, quiet Ollama server at 127.0.0.1:11434. No model is downloaded
or inference server launched. Run these probes and other model work sequentially.
"""

import argparse
import json
import os
import secrets
import subprocess
import tempfile
import time
import tomllib
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import httpx

from devtools.semantic_eval import FROZEN_SHA256, load_dataset
from devtools.smoke import ROOT, free_port, wait_ready
from llm_gateway.embeddings import fingerprint_for_model


def config_text(
    backend: str,
    model_dir: Path,
    threshold: float,
    families: dict,
    real: bool,
    *,
    capital_entities: tuple[str, ...] = (),
) -> str:
    if real:
        recipe = (ROOT / "config/ollama.toml").read_text()
        parsed = tomllib.loads(recipe)
        assert parsed["backends"]["local"]["base_url"] == backend + "/v1"
        assert not parsed["exact_cache"]["enabled"]
        recipe = recipe.replace("enabled = false", "enabled = true", 1)
    else:
        recipe = (
            "request_timeout_seconds = 10.0\nmax_in_flight = 2\n"
            "[rate_limit]\nburst = 100\nrequests_per_second = 1.0\n"
            "[exact_cache]\nenabled = true\nttl_seconds = 60.0\n"
            "max_entries = 16\nmax_bytes = 1048576\n"
            f'[backends.local]\nbase_url = "{backend}/v1"\nrevision = "fixture-v1"\n'
            '[models.demo]\nbackend = "local"\nmodel = "deterministic-mock"\n'
            'revision = "fixture-model-v1"\n'
        )
    recipe += (
        "\n[semantic_cache]\nenabled = true\n"
        f"model_dir = {json.dumps(str(model_dir))}\nthreshold = {threshold}\n"
        "timeout_seconds = 0.25\nmax_entries = 16\n"
        f"capital_entities = {json.dumps(list(capital_entities))}\n"
    )
    for family in families.values():
        prompts = [family["canonical"]] + sum(family["paraphrases"].values(), [])
        recipe += (
            "\n[[semantic_cache.families]]\n"
            + "\n".join(
                f"{key} = {json.dumps(value)}"
                for key, value in {
                    "id": family["id"],
                    "domain": family["scope"],
                    "answer_contract": family["response_requirement"],
                    "prompts": prompts,
                }.items()
            )
            + "\n"
        )
    return recipe


@contextmanager
def servers(
    model_dir: Path,
    threshold: float,
    families: dict,
    real: bool,
    *,
    capital_entities: tuple[str, ...] = (),
):
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="semantic-probe-", dir=artifacts) as directory:
        backend_port, gateway_port = (11434 if real else free_port()), free_port()
        while gateway_port == backend_port:
            gateway_port = free_port()
        backend = f"http://127.0.0.1:{backend_port}"
        gateway = f"http://127.0.0.1:{gateway_port}"
        config = Path(directory) / "gateway.toml"
        config.write_text(
            config_text(
                backend, model_dir, threshold, families, real, capital_entities=capital_entities
            )
        )
        keys = {name: secrets.token_urlsafe(32) for name in ("client-a", "client-b")}
        environment = os.environ.copy()
        environment.update(GATEWAY_CONFIG=str(config), GATEWAY_CLIENT_KEYS_JSON=json.dumps(keys))
        apps = [] if real else [("devtools.traffic_probe:backend_app", backend_port, False)]
        apps.append(("llm_gateway.app:create_app", gateway_port, True))
        processes = []
        try:
            for app, port, factory in apps:
                command = [
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
                if factory:
                    command.append("--factory")
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                )
            with httpx.Client(trust_env=False, timeout=2.0) as client:
                if not real:
                    wait_ready(client, processes[0], f"{backend}/_probe/stats", 200)
                wait_ready(client, processes[-1], f"{gateway}/healthz", 200)
            yield backend, gateway, keys, tomllib.loads(config.read_text())
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


def log_posts(log: Path) -> int:
    return sum(
        "[GIN]" in line and "POST" in line and '"/v1/chat/completions"' in line
        for line in log.read_text(errors="replace").splitlines()
    )


def server_timings(response: httpx.Response) -> dict[str, float]:
    return {
        field.split(";", 1)[0].strip(): float(field.split("dur=", 1)[1])
        for field in response.headers["server-timing"].split(",")
        if "dur=" in field
    }


def run(model_dir: Path, evaluation: Path, log: Path | None) -> dict:
    _, families, _ = load_dataset()
    evidence = json.loads(evaluation.read_text())
    fingerprint = fingerprint_for_model(model_dir)  # Verifies files; does not load an encoder.
    assert evidence["dataset_sha256"] == FROZEN_SHA256
    assert evidence["model_fingerprint"] == fingerprint
    selection = evidence["selected"]["guarded"]
    assert selection["selection_split"] == "calibration"
    assert selection["calibration_metrics"]["false_hits"] == 0
    assert evidence["heldout_at_selected_thresholds"]["guarded"]["false_hits"] == 0
    threshold = selection["threshold"]
    case_scores = {case["id"]: case for case in evidence["case_scores"]}
    real = log is not None
    with servers(model_dir, threshold, families, real) as (backend, gateway, keys, config):
        with httpx.Client(trust_env=False, timeout=70.0 if real else 5.0) as client:
            identity = {}
            if real:
                version = client.get(backend + "/api/version")
                version.raise_for_status()
                assert version.json()["version"] == "0.40.11"
                route, configured = config["models"]["gemma"], config["backends"]["local"]
                assert route["model"] == "gemma3:1b"
                assert configured["revision"].startswith("ollama-0.40.11-")
                tags = client.get(backend + "/api/tags")
                tags.raise_for_status()
                model = next(row for row in tags.json()["models"] if row["name"] == route["model"])
                digest = "sha256:" + model["digest"].removeprefix("sha256:")
                assert digest == route["revision"]
                identity = {"backend_version": "0.40.11", "model_digest": digest}
                before = log_posts(log)
            else:
                reset, release = (
                    client.post(backend + "/_probe/reset"),
                    client.post(backend + "/_probe/release"),
                )
                assert reset.status_code == release.status_code == 200
                before = client.get(backend + "/_probe/stats").json()["calls"]

            rows, responses = [], {}

            def complete(name, text, expected, *, city=None, owner="client-a", sampled=False):
                start = time.perf_counter()
                response = client.post(
                    gateway + "/v1/chat/completions",
                    json={
                        "model": "gemma" if real else "demo",
                        "messages": [{"role": "user", "content": text}],
                        "temperature": 0.7 if sampled else 0.0,
                        "top_p": 1.0,
                        "seed": 42,
                        "max_tokens": 32,
                    },
                    headers={
                        "Authorization": f"Bearer {keys[owner]}",
                        "X-Gateway-Cache": "semantic",
                    },
                )
                elapsed = (time.perf_counter() - start) * 1000
                assert response.status_code == 200
                disposition = response.headers["x-gateway-cache"]
                assert disposition == expected, f"{name}: expected {expected}, got {disposition}"
                timings = server_timings(response)
                if disposition in {"HIT", "SEMANTIC_HIT"}:
                    assert "adapter" not in timings
                else:
                    assert "adapter" in timings
                if disposition == "SEMANTIC_HIT":
                    assert "semantic" in timings
                if real and city:
                    assert response.json()["choices"][0]["message"]["content"].strip() == city
                rows.append(
                    {
                        "case": name,
                        "cache": disposition,
                        "status": response.status_code,
                        # Fixed public fixture inputs only; this is evidence,
                        # never a production request/response logging path.
                        "assistant_text": response.json()["choices"][0]["message"]["content"],
                        "client_latency_ms": elapsed,
                        "server_timing_ms": timings,
                    }
                )
                responses[name] = response
                return response

            france, japan = families["france-capital"], families["japan-capital"]
            complete("france_canonical", france["canonical"], "MISS", city="Paris")
            if real:
                loaded = client.get(backend + "/api/ps")
                loaded.raise_for_status()
                running = next(row for row in loaded.json()["models"] if row["name"] == "gemma3:1b")
                assert (
                    "sha256:" + running["digest"].removeprefix("sha256:")
                    == identity["model_digest"]
                )
                assert running["context_length"] == 2048
                identity["observed_context_length"] = running["context_length"]
            exact = complete("france_exact", france["canonical"], "HIT", city="Paris")
            assert exact.content == responses["france_canonical"].content
            for family, city in ((france, "Paris"), (japan, "Tokyo")):
                if family is japan:
                    complete("japan_canonical", family["canonical"], "MISS", city=city)
                score = case_scores[f"heldout-{family['id']}-positive-1"]["guarded_similarity"]
                expected = "SEMANTIC_HIT" if score >= threshold else "MISS"
                name = family["id"].split("-", 1)[0]
                result = complete(
                    name + "_paraphrase", family["paraphrases"]["heldout"][0], expected, city=city
                )
                if expected == "SEMANTIC_HIT":
                    assert result.content == responses[name + "_canonical"].content
            complete(
                "germany_negative",
                france["canonical"].replace("France", "Germany"),
                "BYPASS",
                city="Berlin",
            )
            complete("france_client_b", france["canonical"], "MISS", city="Paris", owner="client-b")
            complete("france_sampled", france["canonical"], "BYPASS", city="Paris", sampled=True)
            expected_calls = sum(row["cache"] in {"MISS", "BYPASS"} for row in rows)
            if real:
                for _ in range(10):
                    time.sleep(0.1)
                    calls = log_posts(log) - before
                    if calls >= expected_calls:
                        break
                source = "dedicated_quiet_ollama_gin_log"
            else:
                observed = client.get(backend + "/_probe/stats").json()
                assert observed["active"] == 0
                calls, source = (
                    observed["calls"] - before,
                    "disposable_backend_fixture/_probe/stats",
                )
            assert calls == expected_calls, (
                "Independent backend calls differ; require a quiet backend"
            )
            return {
                "backend": "ollama" if real else "deterministic_mock",
                "transport": "loopback_http",
                "real_model_inference": real,
                "actual_local_embeddings": True,
                "throughput_benchmark": False,
                "answer_correctness_evaluated": real,
                "dataset_sha256": FROZEN_SHA256,
                "embedding_fingerprint": fingerprint,
                "embedding_model_directory": str(model_dir.relative_to(ROOT)),
                "calibration_selected_threshold": threshold,
                "model_identity": identity,
                "requests": len(rows),
                "cache_dispositions": dict(Counter(row["cache"] for row in rows)),
                "actual_upstream_calls": calls,
                "call_count_source": source,
                "uncached_reference_calls": len(rows),
                "upstream_calls_avoided": len(rows) - calls,
                "relative_upstream_call_reduction": 1 - calls / len(rows),
                "client_isolation_verified": True,
                "negative_and_sampling_bypass_verified": True,
                "first_response": responses["france_canonical"].json(),
                "request_observations": rows,
                "timing_note": "HTTP and Server-Timing boundaries differ; semantic includes worker "
                "scheduling/encoding/search; adapter covers upstream I/O and response processing.",
                "scope_note": "Reviewed phrase families authorize reuse; cosine alone does not. "
                "Cached IDs, timestamps and usage are original, not new inference usage.",
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/embeddings/l3"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation/l3_results.json"))
    parser.add_argument(
        "--ollama-log", type=Path, help="Existing dedicated Ollama log under artifacts/"
    )
    parser.add_argument("--output", type=Path, help="JSON output inside artifacts/")
    args = parser.parse_args()
    model_dir, evaluation = (ROOT / args.model_dir).resolve(), (ROOT / args.evaluation).resolve()
    log = (ROOT / args.ollama_log).resolve() if args.ollama_log else None
    output = (ROOT / args.output).resolve() if args.output else None
    if not model_dir.is_relative_to(ROOT) or not model_dir.is_dir():
        parser.error("--model-dir must be an existing repository directory")
    if not evaluation.is_relative_to(ROOT) or not evaluation.is_file():
        parser.error("--evaluation must be an existing repository file")
    if log and (not log.is_relative_to(ROOT / "artifacts") or not log.is_file()):
        parser.error("--ollama-log must be an existing file under repository artifacts/")
    if output and (
        not output.is_relative_to(ROOT / "artifacts")
        or output == ROOT / "artifacts"
        or output == log
    ):
        parser.error(
            "--output must be a file under artifacts/ and must not overwrite the Ollama log"
        )
    report = json.dumps(run(model_dir, evaluation, log), indent=2, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
