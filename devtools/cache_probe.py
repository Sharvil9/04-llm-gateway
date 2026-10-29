"""Verify exact-cache call reduction over actual loopback HTTP.

Run: .venv/bin/python -m devtools.cache_probe --output artifacts/cache-probe.json
Use --ollama-log artifacts/.../ollama.log for an existing dedicated, quiet Ollama
server at 127.0.0.1:11434. This mode starts only a gateway, never installs or pulls.
This is a correctness demonstration, not a model inference/throughput benchmark.
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

from devtools.smoke import ROOT, free_port, wait_ready

PUBLIC_PROMPT = "What is the capital of France?"
QUESTION = {
    "model": "demo",
    "messages": [{"role": "user", "content": PUBLIC_PROMPT}],
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
}


@contextmanager
def servers():
    """Launch the existing traffic fixture and an isolated cache-enabled gateway."""
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cache-probe-", dir=artifacts) as directory:
        backend_port, gateway_port = free_port(), free_port()
        while gateway_port == backend_port:
            gateway_port = free_port()
        backend = f"http://127.0.0.1:{backend_port}"
        gateway = f"http://127.0.0.1:{gateway_port}"
        config = Path(directory) / "gateway.toml"
        config.write_text(
            "request_timeout_seconds = 10.0\nmax_in_flight = 2\n"
            "[rate_limit]\nburst = 100\nrequests_per_second = 1.0\n"
            "[exact_cache]\nenabled = true\nttl_seconds = 60.0\n"
            "max_entries = 4\nmax_bytes = 1048576\n"
            f"approved_prompts = [{json.dumps(PUBLIC_PROMPT)}]\n"
            f'[backends.local]\nbase_url = "{backend}/v1"\nrevision = "fixture-v1"\n'
            '[models.demo]\nbackend = "local"\nmodel = "deterministic-mock"\n'
            'revision = "fixture-model-v1"\n'
        )
        keys = {name: secrets.token_urlsafe(32) for name in ("client-a", "client-b")}
        environment = os.environ.copy()
        environment.update(GATEWAY_CONFIG=str(config), GATEWAY_CLIENT_KEYS_JSON=json.dumps(keys))
        processes = []
        try:
            for app, port, factory in (
                ("devtools.traffic_probe:backend_app", backend_port, False),
                ("llm_gateway.app:create_app", gateway_port, True),
            ):
                command = [
                    str(ROOT / ".venv/bin/python"),
                    "-m",
                    "uvicorn",
                    app,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
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
                wait_ready(client, processes[0], f"{backend}/_probe/stats", 200)
                wait_ready(client, processes[1], f"{gateway}/healthz", 200)
            yield backend, gateway, keys
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


def run() -> dict:
    with servers() as (backend, gateway, keys):
        with httpx.Client(trust_env=False, timeout=3.0) as client:
            endpoint = f"{gateway}/v1/chat/completions"
            reset = client.post(f"{backend}/_probe/reset")
            release = client.post(f"{backend}/_probe/release")
            assert reset.status_code == release.status_code == 200

            def complete(owner: str, *, sampled: bool = False) -> httpx.Response:
                response = client.post(
                    endpoint,
                    json=QUESTION | ({"temperature": 0.7} if sampled else {}),
                    headers={
                        "Authorization": f"Bearer {keys[owner]}",
                        "X-Gateway-Cache": "exact",
                    },
                )
                assert response.status_code == 200
                assert response.json()["model"] == "demo"
                assert response.headers["cache-control"] == "no-store"
                return response

            first_client = [complete("client-a") for _ in range(3)]
            first_statuses = [response.headers["x-gateway-cache"] for response in first_client]
            assert first_statuses == ["MISS", "HIT", "HIT"]
            assert len({response.content for response in first_client}) == 1
            assert client.get(f"{backend}/_probe/stats").json()["calls"] == 1

            second_client = [complete("client-b") for _ in range(2)]
            second_statuses = [response.headers["x-gateway-cache"] for response in second_client]
            assert second_statuses == ["MISS", "HIT"]
            assert second_client[0].content == second_client[1].content
            assert client.get(f"{backend}/_probe/stats").json()["calls"] == 2

            sampled = complete("client-a", sampled=True)
            assert sampled.headers["x-gateway-cache"] == "BYPASS"
            observed = client.get(f"{backend}/_probe/stats").json()
            assert observed == {"calls": 3, "active": 0, "peak": 1}

            unauthorized = client.post(
                endpoint, json=QUESTION, headers={"X-Gateway-Cache": "exact"}
            )
            assert unauthorized.status_code == 401
            assert "x-gateway-cache" not in unauthorized.headers
            assert unauthorized.content != first_client[0].content
            assert client.get(f"{backend}/_probe/stats").json() == observed

            dispositions = first_statuses + second_statuses + ["BYPASS"]
            successful_requests = len(dispositions)
            calls = observed["calls"]
            return {
                "backend": "deterministic_mock",
                "transport": "loopback_http",
                "real_model_inference": False,
                "throughput_benchmark": False,
                "answer_correctness_evaluated": False,
                "authenticated_successful_requests": successful_requests,
                "unauthenticated_checks": 1,
                "client_cache_sequences": {
                    "client-a": first_statuses,
                    "client-b": second_statuses,
                    "sampled_client-a": "BYPASS",
                },
                "cache_dispositions": dict(Counter(dispositions)),
                "actual_upstream_calls": calls,
                "call_count_source": "disposable_backend_fixture/_probe/stats",
                "uncached_reference_calls": successful_requests,
                "uncached_reference_basis": "one upstream call per successful request",
                "upstream_calls_avoided": successful_requests - calls,
                "relative_upstream_call_reduction": 1 - calls / successful_requests,
                "byte_identical_cache_replays": True,
                "client_isolation_verified": True,
                "sampling_bypass_verified": True,
                "unauthorized_status": unauthorized.status_code,
                "unauthorized_backend_calls": 0,
                "backend_active_at_completion": observed["active"],
            }


def run_ollama(log: Path) -> dict:
    """Verify pinned local inference and count completed upstream POSTs in its log."""
    recipe = (ROOT / "config/ollama.toml").read_text()
    config = tomllib.loads(recipe)
    route, backend = config["models"]["gemma"], config["backends"]["local"]
    assert route["model"] == "gemma3:1b" and route["backend"] == "local"
    assert backend["base_url"] == "http://127.0.0.1:11434/v1"
    assert backend["revision"].startswith("ollama-0.40.11-")
    assert config["exact_cache"]["enabled"] is False
    recipe = recipe.replace("enabled = false", "enabled = true", 1)
    assert tomllib.loads(recipe)["exact_cache"]["enabled"] is True
    prompt = config["exact_cache"]["approved_prompts"][0]
    assert prompt == "What is the capital of France? Answer with only the city name."

    def posts() -> int:
        return sum(
            "[GIN]" in line and "POST" in line and '"/v1/chat/completions"' in line
            for line in log.read_text(errors="replace").splitlines()
        )

    with httpx.Client(trust_env=False, timeout=70.0) as client:
        version = client.get("http://127.0.0.1:11434/api/version")
        version.raise_for_status()
        assert version.json()["version"] == "0.40.11"
        tags = client.get("http://127.0.0.1:11434/api/tags")
        tags.raise_for_status()
        model = next(item for item in tags.json()["models"] if item["name"] == route["model"])
        digest = "sha256:" + model["digest"].removeprefix("sha256:")
        assert digest == route["revision"]
        with tempfile.TemporaryDirectory(
            prefix="cache-probe-", dir=ROOT / "artifacts"
        ) as directory:
            temporary_config = Path(directory) / "gateway.toml"
            temporary_config.write_text(recipe)
            credential = secrets.token_urlsafe(32)
            environment = os.environ.copy()
            environment.update(
                GATEWAY_CONFIG=str(temporary_config),
                GATEWAY_CLIENT_KEYS_JSON=json.dumps({"cache-probe": credential}),
            )
            gateway = f"http://127.0.0.1:{(port := free_port())}"
            process = subprocess.Popen(
                [
                    str(ROOT / ".venv/bin/python"),
                    "-m",
                    "uvicorn",
                    "llm_gateway.app:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--workers",
                    "1",
                    "--no-access-log",
                    "--log-level",
                    "warning",
                ],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                wait_ready(client, process, f"{gateway}/healthz", 200)
                before, responses, context = posts(), [], None
                for index in range(3):
                    response = client.post(
                        f"{gateway}/v1/chat/completions",
                        json=QUESTION
                        | {
                            "model": "gemma",
                            "max_tokens": 32,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                        headers={
                            "Authorization": f"Bearer {credential}",
                            "X-Gateway-Cache": "exact",
                        },
                    )
                    assert response.status_code == 200
                    responses.append(response)
                    if index == 0:
                        loaded = client.get("http://127.0.0.1:11434/api/ps")
                        loaded.raise_for_status()
                        running = next(
                            item
                            for item in loaded.json()["models"]
                            if item["name"] == route["model"]
                        )
                        assert "sha256:" + running["digest"].removeprefix("sha256:") == digest
                        context = running["context_length"]
                        assert context == 2048
                dispositions = [item.headers["x-gateway-cache"] for item in responses]
                assert dispositions == ["MISS", "HIT", "HIT"]
                assert len({item.content for item in responses}) == 1
                original = responses[0].json()
                assert original["choices"][0]["message"]["content"].strip() == "Paris"
                for _ in range(10):
                    time.sleep(0.1)
                    after = posts()
                    if after > before:
                        break
                assert after - before == 1, "Expected one POST in a dedicated, quiet Ollama log"
                return {
                    "backend": "ollama",
                    "transport": "loopback_http",
                    "real_model_inference": True,
                    "throughput_benchmark": False,
                    "backend_version": version.json()["version"],
                    "configured_backend_revision": backend["revision"],
                    "model": route["model"],
                    "model_digest": digest,
                    "observed_context_length": context,
                    "authenticated_successful_requests": 3,
                    "cache_sequence": dispositions,
                    "cache_dispositions": {"MISS": 1, "HIT": 2},
                    "actual_upstream_calls": after - before,
                    "call_count_source": "ollama_gin_log",
                    "uncached_reference_calls": 3,
                    "upstream_calls_avoided": 2,
                    "uncached_reference_basis": "one upstream call per successful request",
                    "relative_upstream_call_reduction": 2 / 3,
                    "byte_identical_cache_replays": True,
                    "dedicated_quiet_server_required": True,
                    "first_response": original,
                    "usage_note": "Replays retain original response usage; no new inference usage",
                }
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Optional JSON path under this repository's artifacts/"
    )
    parser.add_argument(
        "--ollama-log",
        type=Path,
        help="Existing dedicated Ollama GIN log under artifacts/; use genuine model mode",
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve() if args.output else None
    if output and (not output.is_relative_to(ROOT / "artifacts") or output == ROOT / "artifacts"):
        parser.error("--output must name a file inside this repository's artifacts/ directory")
    log = (ROOT / args.ollama_log).resolve() if args.ollama_log else None
    if log and (not log.is_relative_to(ROOT / "artifacts") or not log.is_file()):
        parser.error("--ollama-log must be an existing file inside this repository's artifacts/")
    if log and log == output:
        parser.error("--output must not overwrite --ollama-log")
    report = json.dumps(run_ollama(log) if log else run(), indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
