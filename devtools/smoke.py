"""Run the foundation demonstration across two actual loopback HTTP servers."""

import json
import os
import secrets
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_ready(client: httpx.Client, process: subprocess.Popen, url: str, status: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Demo server exited before readiness")
        try:
            if client.get(url).status_code == status:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise RuntimeError("Demo server did not become ready within 10 seconds")


def main() -> None:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    processes = []
    try:
        with tempfile.TemporaryDirectory(prefix="smoke-", dir=artifacts) as directory:
            backend_port, gateway_port = free_port(), free_port()
            while gateway_port == backend_port:
                gateway_port = free_port()
            config = Path(directory) / "mock.toml"
            config.write_text(
                f'[backends.local]\nbase_url = "http://127.0.0.1:{backend_port}/v1"\n'
                '[models.demo]\nbackend = "local"\nmodel = "deterministic-mock"\n'
            )
            credential = secrets.token_urlsafe(32)
            environment = os.environ.copy()
            environment["GATEWAY_CONFIG"] = str(config)
            environment["GATEWAY_CLIENT_KEYS_JSON"] = json.dumps({"smoke-client": credential})
            for app, port, factory in [
                ("devtools.mock_backend:app", backend_port, False),
                ("llm_gateway.app:create_app", gateway_port, True),
            ]:
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
                base = f"http://127.0.0.1:{gateway_port}"
                wait_ready(client, processes[0], f"http://127.0.0.1:{backend_port}/", 404)
                wait_ready(client, processes[1], f"{base}/healthz", 200)
                endpoint = f"{base}/v1/chat/completions"
                payload = {"model": "demo", "messages": [{"role": "user", "content": "Hello"}]}
                headers = {"Authorization": f"Bearer {credential}"}
                unauthorized = client.post(endpoint, json=payload)
                invalid = client.post(endpoint, json=payload | {"stream": True}, headers=headers)
                unknown = client.post(endpoint, json=payload | {"model": "absent"}, headers=headers)
                models = client.get(f"{base}/v1/models", headers=headers)
                valid = client.post(endpoint, json=payload, headers=headers)
                assert unauthorized.status_code == 401
                assert invalid.status_code == 422
                assert unknown.status_code == 404
                assert (
                    models.status_code == 200
                    and {"id": "demo", "object": "model"} in models.json()["data"]
                )
                assert valid.status_code == 200
                assert valid.json()["model"] == "demo"
                assert (
                    valid.json()["choices"][0]["message"]["content"]
                    == "Deterministic mock response."
                )
                print(
                    json.dumps(
                        {
                            "backend": "deterministic_mock",
                            "real_model_inference": False,
                            "checks": {
                                "unauthorized": 401,
                                "invalid_stream": 422,
                                "unknown_model": 404,
                                "model_listing": 200,
                                "completion": 200,
                            },
                        }
                    )
                )
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


if __name__ == "__main__":
    main()
