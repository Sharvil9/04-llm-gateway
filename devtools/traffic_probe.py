"""Verify quotas/admission over loopback HTTP; this is not a throughput benchmark.

The backend and its control endpoints are disposable fixtures bound to loopback.
Run: .venv/bin/python -m devtools.traffic_probe --output artifacts/traffic-probe.json
"""

import argparse
import asyncio
import json
import os
import secrets
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException

from devtools.smoke import ROOT, free_port, wait_ready
from llm_gateway.schemas import ChatRequest

backend_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_gate = asyncio.Event()
_stats = {"calls": 0, "active": 0, "peak": 0}
QUESTION = {"model": "demo", "messages": [{"role": "user", "content": "Probe request"}]}


@backend_app.post("/_probe/reset")
async def reset():
    global _gate
    if _stats["active"]:
        raise HTTPException(409, "Cannot reset active fixture requests")
    _gate = asyncio.Event()
    _stats.update(calls=0, active=0, peak=0)
    return _stats


@backend_app.get("/_probe/stats")
async def stats():
    return _stats


@backend_app.post("/_probe/release")
async def release():
    _gate.set()
    return {"released": True}


@backend_app.post("/v1/chat/completions")
async def complete(request: ChatRequest):
    _stats["calls"] += 1
    _stats["active"] += 1
    _stats["peak"] = max(_stats["peak"], _stats["active"])
    try:
        await _gate.wait()
        return {
            "id": "chatcmpl-traffic-probe",
            "object": "chat.completion",
            "created": 0,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Fixture response."},
                    "finish_reason": "stop",
                }
            ],
        }
    finally:
        _stats["active"] -= 1


@contextmanager
def servers(policy: str, *, metrics_key: str | None = None):
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="traffic-probe-", dir=artifacts) as directory:
        backend_port, gateway_port = free_port(), free_port()
        while gateway_port == backend_port:
            gateway_port = free_port()
        backend, gateway = f"http://127.0.0.1:{backend_port}", f"http://127.0.0.1:{gateway_port}"
        config = Path(directory) / "gateway.toml"
        config.write_text(
            "request_timeout_seconds = 10.0\n"
            + policy
            + f'\n[backends.local]\nbase_url = "{backend}/v1"\n'
            '[models.demo]\nbackend = "local"\nmodel = "deterministic-mock"\n'
        )
        keys = {name: secrets.token_urlsafe(32) for name in ["client-a", "client-b"]}
        environment = os.environ.copy()
        environment.update(GATEWAY_CONFIG=str(config), GATEWAY_CLIENT_KEYS_JSON=json.dumps(keys))
        if metrics_key:
            environment["GATEWAY_METRICS_KEY"] = metrics_key
        processes = []
        try:
            for app, port, factory in [
                ("devtools.traffic_probe:backend_app", backend_port, False),
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
            with httpx.Client(trust_env=False, timeout=1.0) as client:
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


async def post_chat(client: httpx.AsyncClient, gateway: str, key: str) -> httpx.Response:
    return await client.post(
        f"{gateway}/v1/chat/completions", json=QUESTION, headers={"Authorization": f"Bearer {key}"}
    )


async def clean_requests(client: httpx.AsyncClient, backend: str, tasks: list) -> None:
    try:
        await client.post(f"{backend}/_probe/release")
    except httpx.HTTPError:
        pass
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def metrics_snapshot(client: httpx.AsyncClient, gateway: str, key: str) -> dict:
    for _ in range(100):
        response = await client.get(
            f"{gateway}/v1/metrics", headers={"Authorization": f"Bearer {key}"}
        )
        assert response.status_code == 200
        result = response.json()
        if result["gauges"]["requests_in_flight"] == 0:
            return result
        await asyncio.sleep(0.01)
    raise RuntimeError("Completion metrics did not settle")


async def quota_phase(client: httpx.AsyncClient) -> dict:
    policy = (
        "max_in_flight = 8\n[rate_limit]\nburst = 3\nrequests_per_second = 0.000001\n"
        "[client_rate_limits.client-b]\nburst = 2\nrequests_per_second = 0.000001\n"
    )
    operator_key = secrets.token_urlsafe(32)
    with servers(policy, metrics_key=operator_key) as (backend, gateway, keys):
        await client.post(f"{backend}/_probe/reset")
        tagged = [
            (name, asyncio.create_task(post_chat(client, gateway, key)))
            for name, key in keys.items()
            for _ in range(12)
        ]
        tasks = [task for _, task in tagged]
        try:
            async with asyncio.timeout(5):
                while True:
                    observed = (await client.get(f"{backend}/_probe/stats")).json()
                    finished = [task.result() for task in tasks if task.done()]
                    assert all(response.status_code == 429 for response in finished)
                    if observed["active"] == 5 and len(finished) == 19:
                        break
                    await asyncio.sleep(0.01)
            assert observed == {"calls": 5, "active": 5, "peak": 5}
            assert all(int(response.headers["Retry-After"]) > 0 for response in finished)
            assert all(
                response.json()["error"]["code"] == "rate_limit_exceeded" for response in finished
            )
            await client.post(f"{backend}/_probe/release")
            await asyncio.gather(*tasks)
            counts = {
                name: dict(
                    Counter(task.result().status_code for owner, task in tagged if owner == name)
                )
                for name in keys
            }
            assert counts == {"client-a": {200: 3, 429: 9}, "client-b": {200: 2, 429: 10}}
            assert (await client.get(f"{backend}/_probe/stats")).json()["active"] == 0
            metrics = await metrics_snapshot(client, gateway, operator_key)
            assert metrics["counts"]["responses_completed"] == 24
            assert metrics["status"]["exact"]["429"] == 19
            assert metrics["counts"]["upstream_attempts"] == 5
            return {
                "requests_per_client": 12,
                "client_status_counts": counts,
                "backend_calls": 5,
                "backend_peak_in_flight": 5,
                "positive_retry_after": True,
                "client_isolation_verified": True,
                "gateway_metrics": metrics,
            }
        finally:
            await clean_requests(client, backend, tasks)


async def capacity_phase(client: httpx.AsyncClient) -> dict:
    policy = "max_in_flight = 2\n[rate_limit]\nburst = 100\nrequests_per_second = 0.000001\n"
    operator_key = secrets.token_urlsafe(32)
    with servers(policy, metrics_key=operator_key) as (backend, gateway, keys):
        await client.post(f"{backend}/_probe/reset")
        admitted = [
            asyncio.create_task(post_chat(client, gateway, keys["client-a"])) for _ in range(2)
        ]
        try:
            async with asyncio.timeout(5):
                while True:
                    observed = (await client.get(f"{backend}/_probe/stats")).json()
                    if observed["active"] == 2:
                        break
                    await asyncio.sleep(0.01)
            async with asyncio.timeout(5):
                excess = await asyncio.gather(
                    *[post_chat(client, gateway, keys["client-a"]) for _ in range(20)]
                )
            assert all(response.status_code == 503 for response in excess)
            assert all(
                response.json()["error"]["code"] == "gateway_overloaded" for response in excess
            )
            assert all("Retry-After" not in response.headers for response in excess)
            assert (await client.get(f"{backend}/_probe/stats")).json() == {
                "calls": 2,
                "active": 2,
                "peak": 2,
            }
            await client.post(f"{backend}/_probe/release")
            assert all(result.status_code == 200 for result in await asyncio.gather(*admitted))
            recovery = await post_chat(client, gateway, keys["client-a"])
            assert recovery.status_code == 200
            assert (await client.get(f"{backend}/_probe/stats")).json() == {
                "calls": 3,
                "active": 0,
                "peak": 2,
            }
            metrics = await metrics_snapshot(client, gateway, operator_key)
            assert metrics["counts"]["responses_completed"] == 23
            assert metrics["status"]["exact"]["503"] == 20
            assert metrics["counts"]["upstream_attempts"] == 3
            assert metrics["gauges"]["inference_in_flight"] == 0
            return {
                "configured_capacity": 2,
                "admitted": 2,
                "excess_requests": 20,
                "overload_rejections": 20,
                "backend_calls_before_release": 2,
                "backend_peak_in_flight": 2,
                "retry_after_omitted": True,
                "recovery_status": recovery.status_code,
                "backend_calls_after_recovery": 3,
                "gateway_metrics": metrics,
            }
        finally:
            await clean_requests(client, backend, admitted)


async def run() -> dict:
    async with httpx.AsyncClient(
        trust_env=False, timeout=8.0, limits=httpx.Limits(max_connections=64)
    ) as client:
        quotas = await quota_phase(client)
        capacity = await capacity_phase(client)
    return {
        "backend": "deterministic_mock",
        "transport": "loopback_http",
        "real_model_inference": False,
        "process_local_limits": True,
        "quota_scenario": quotas,
        "capacity_scenario": capacity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Optional JSON path under this repository's artifacts/"
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve() if args.output else None
    if output and (not output.is_relative_to(ROOT / "artifacts") or output == ROOT / "artifacts"):
        parser.error("--output must name a file inside this repository's artifacts/ directory")
    report = json.dumps(asyncio.run(run()), indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
