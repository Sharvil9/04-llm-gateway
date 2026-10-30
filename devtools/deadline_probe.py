"""Measure deadline scopes over real loopback HTTP; no LLM or throughput claim."""

import argparse
import json
import os
import secrets
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx

from devtools.smoke import ROOT, free_port, wait_ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve() if args.output else None
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    if output and not output.is_relative_to(artifacts):
        parser.error("output must be inside this repository's artifacts directory")

    chunks = []

    class Trickle(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for _ in range(60):
                    self.wfile.write(b"1\r\n \r\n")
                    self.wfile.flush()
                    chunks.append(time.perf_counter())
                    time.sleep(0.05)
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    backend = HTTPServer(("127.0.0.1", 0), Trickle)
    worker = threading.Thread(target=backend.serve_forever, daemon=True)
    worker.start()
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="deadline-", dir=artifacts) as directory:
            config = Path(directory) / "gateway.toml"
            config.write_text(
                "request_timeout_seconds = 1.0\nmax_in_flight = 1\n"
                f'[backends.local]\nbase_url = "http://127.0.0.1:{backend.server_port}/v1"\n'
                '[models.demo]\nbackend = "local"\nmodel = "deterministic-mock"\n'
            )
            port = free_port()
            key = secrets.token_urlsafe(32)
            operator_key = secrets.token_urlsafe(32)
            environment = os.environ.copy()
            environment["GATEWAY_CONFIG"] = str(config)
            environment["GATEWAY_CLIENT_KEYS_JSON"] = json.dumps({"probe": key})
            environment["GATEWAY_METRICS_KEY"] = operator_key
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
                ],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with httpx.Client(trust_env=False, timeout=10) as client:
                base = f"http://127.0.0.1:{port}"
                wait_ready(client, process, f"{base}/healthz", 200)
                started = time.perf_counter()
                response = client.post(
                    f"{base}/v1/chat/completions",
                    json={"model": "demo", "messages": [{"role": "user", "content": "probe"}]},
                    headers={"Authorization": f"Bearer {key}"},
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                measured = {
                    name: float(value)
                    for item in response.headers["server-timing"].split(", ")
                    for name, value in [item.split(";dur=")]
                }
                assert response.status_code == 504 and len(chunks) >= 2
                observed = client.get(
                    f"{base}/v1/metrics",
                    headers={"Authorization": f"Bearer {operator_key}"},
                )
                observed.raise_for_status()
                metrics = observed.json()
                assert metrics["status"]["exact"]["504"] == 1
                assert metrics["counts"]["upstream_attempts"] == 1
                assert metrics["gauges"]["inference_in_flight"] == 0
                result = {
                    "backend": "deterministic_http_trickle",
                    "real_model_inference": False,
                    "status": response.status_code,
                    "configured_upstream_deadline_ms": 1000,
                    "fixture_chunk_interval_ms": 50,
                    "fixture_chunks_sent": len(chunks),
                    "client_full_response_ms": round(elapsed_ms, 3),
                    "gateway_to_response_start_ms": measured["gateway"],
                    "adapter_roundtrip_ms": measured["adapter"],
                    "gateway_metrics": metrics,
                    "interpretation": (
                        "Scope differences include scheduling and transport; not CPU overhead"
                    ),
                }
                if output:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result, indent=2))
    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        backend.shutdown()
        backend.server_close()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()
