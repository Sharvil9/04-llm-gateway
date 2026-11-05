"""Bounded repeated HTTP cancellation/timeout recovery, separate from throughput.

The disposable held backend deliberately does not stop computation on socket
disconnect. Gateway permit release and abandoned backend work are separate facts.
No model, production control endpoint or gateway code change is involved.
"""

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

import httpx

from devtools.smoke import ROOT, free_port, wait_ready
from devtools.traffic_probe import capacity_phase, post_chat, quota_phase


def source_manifest() -> dict[str, str]:
    paths = [
        Path(__file__),
        ROOT / "devtools/traffic_probe.py",
        ROOT / "devtools/smoke.py",
        ROOT / "uv.lock",
        *sorted((ROOT / "src/llm_gateway").glob("*.py")),
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def owned_process_sample(pid: int) -> dict:
    """Read RSS/fd count only for our explicitly launched gateway PID."""
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
        descriptors = list(islice(Path(f"/proc/{pid}/fd").iterdir(), 513))
        return {
            "available": True,
            "rss_bytes": rss,
            "open_fds": len(descriptors) if len(descriptors) <= 512 else None,
            "fd_count_truncated": len(descriptors) > 512,
        }
    except (OSError, ValueError, StopIteration):
        return {"available": False, "rss_bytes": None, "open_fds": None}


@contextmanager
def held_servers(*, timeout: float, capacity: int) -> Iterator[tuple[str, str, str, str, int]]:
    """Use the established fixture/production gateway with disposable config."""
    (ROOT / "artifacts").mkdir(exist_ok=True)
    backend_port, gateway_port = free_port(), free_port()
    while gateway_port == backend_port:
        gateway_port = free_port()
    backend, gateway = f"http://127.0.0.1:{backend_port}", f"http://127.0.0.1:{gateway_port}"
    client_key, operator_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory(prefix="load-reliability-", dir=ROOT / "artifacts") as tmp:
        config = Path(tmp) / "gateway.toml"
        config.write_text(
            f"request_timeout_seconds = {timeout}\nmax_in_flight = {capacity}\n"
            "[rate_limit]\nburst = 10000\nrequests_per_second = 10000.0\n"
            f'[backends.local]\nbase_url = "{backend}/v1"\n'
            '[models.demo]\nbackend = "local"\nmodel = "deterministic-mock"\n'
        )
        environment = os.environ.copy()
        environment.update(
            GATEWAY_CONFIG=str(config),
            GATEWAY_CLIENT_KEYS_JSON=json.dumps({"reliability-client": client_key}),
            GATEWAY_METRICS_KEY=operator_key,
        )
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
            with httpx.Client(trust_env=False, timeout=1.0) as client:
                wait_ready(client, processes[0], backend + "/_probe/stats", 200)
                wait_ready(client, processes[1], gateway + "/healthz", 200)
            yield backend, gateway, client_key, operator_key, processes[1].pid
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


async def wait_json(
    client: httpx.AsyncClient,
    url: str,
    predicate: Callable[[dict], bool],
    *,
    headers: dict[str, str] | None = None,
) -> dict:
    async with asyncio.timeout(5.0):
        while True:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if predicate(data):
                return data
            await asyncio.sleep(0.01)


async def gateway_idle(client: httpx.AsyncClient, gateway: str, operator_key: str) -> dict:
    return await wait_json(
        client,
        gateway + "/v1/metrics",
        lambda value: (
            value["gauges"]["inference_in_flight"] == 0
            and value["gauges"]["requests_in_flight"] == 0
        ),
        headers={"Authorization": "Bearer " + operator_key},
    )


async def cancel_requests(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def cancellation_phase(client: httpx.AsyncClient, *, cycles: int = 5) -> dict:
    if not 1 <= cycles <= 10:
        raise ValueError("Use cancellation cycles 1..10")
    with held_servers(timeout=5.0, capacity=2) as (backend, gateway, key, operator, pid):
        rows, resources = [], [owned_process_sample(pid)]
        for index in range(cycles):
            reset = await client.post(backend + "/_probe/reset")
            reset.raise_for_status()
            before = await gateway_idle(client, gateway, operator)
            tasks = [asyncio.create_task(post_chat(client, gateway, key)) for _ in range(2)]
            try:
                await wait_json(client, backend + "/_probe/stats", lambda x: x["active"] == 2)
                observed = await wait_json(
                    client,
                    gateway + "/v1/metrics",
                    lambda x: x["gauges"]["inference_in_flight"] == 2,
                    headers={"Authorization": "Bearer " + operator},
                )
                excess = await asyncio.gather(*[post_chat(client, gateway, key) for _ in range(20)])
                assert all(response.status_code == 503 for response in excess)
                assert all("Retry-After" not in response.headers for response in excess)
                assert all(
                    response.json()["error"]["code"] == "gateway_overloaded" for response in excess
                )
                tasks[0].cancel()
                await asyncio.gather(tasks[0], return_exceptions=True)
                released_one = await wait_json(
                    client,
                    gateway + "/v1/metrics",
                    lambda x: x["gauges"]["inference_in_flight"] == 1,
                    headers={"Authorization": "Bearer " + operator},
                )
                tasks.append(asyncio.create_task(post_chat(client, gateway, key)))
                lingering = await wait_json(
                    client,
                    backend + "/_probe/stats",
                    lambda x: x["calls"] == 3,
                )
                assert lingering["active"] == 3
                await cancel_requests(tasks)
                idle = await gateway_idle(client, gateway, operator)
                # The fixture deliberately keeps its native application jobs
                # alive after socket closure; releasing it proves this caveat.
                await client.post(backend + "/_probe/release")
                await wait_json(client, backend + "/_probe/stats", lambda x: x["active"] == 0)
                recovery = await post_chat(client, gateway, key)
                assert recovery.status_code == 200
                after = await gateway_idle(client, gateway, operator)
                final_backend = (await client.get(backend + "/_probe/stats")).json()
                assert final_backend == {"calls": 4, "active": 0, "peak": 3}
                assert (
                    after["counts"]["upstream_attempts"] - before["counts"]["upstream_attempts"]
                    == 4
                )
                assert after["status"]["exact"]["503"] - before["status"]["exact"]["503"] == 20
                assert after["outcomes"]["disconnected"] - before["outcomes"]["disconnected"] == 3
                rows.append(
                    {
                        "cycle": index + 1,
                        "capacity": 2,
                        "observed_gateway_inference_peak": 2,
                        "held_gateway_inference": observed["gauges"]["inference_in_flight"],
                        "after_one_disconnect": released_one["gauges"]["inference_in_flight"],
                        "after_all_disconnects": idle["gauges"]["inference_in_flight"],
                        "overload_503": len(excess),
                        "cancelled_client_requests": 3,
                        "uncooperative_backend_jobs_before_release": lingering["active"],
                        "actual_upstream_calls": final_backend["calls"],
                        "recovery_status": 200,
                        "final_metrics": after,
                    }
                )
                resources.append(owned_process_sample(pid))
            finally:
                await client.post(backend + "/_probe/release")
                await cancel_requests(tasks)
        return {
            "cycles": cycles,
            "rows": rows,
            "resource_samples": resources,
            "gateway_capacity": 2,
            "waiting_slots": 0,
            "network_cancellation": "Cancelling HTTPX tasks closes actual client sockets",
            "backend_caveat": (
                "Fixture jobs intentionally ignore disconnects and remain active until released. "
                "Gateway capacity limits admitted operations, not abandoned remote computation."
            ),
            "resource_scope": (
                "Gateway RSS/fd samples at start and after each idle recovery; not peak RSS"
            ),
        }


async def timeout_phase(client: httpx.AsyncClient, *, cycles: int = 3) -> dict:
    if not 1 <= cycles <= 10:
        raise ValueError("Use timeout cycles 1..10")
    with held_servers(timeout=0.15, capacity=1) as (backend, gateway, key, operator, pid):
        rows, resources = [], [owned_process_sample(pid)]
        for index in range(cycles):
            (await client.post(backend + "/_probe/reset")).raise_for_status()
            before = await gateway_idle(client, gateway, operator)
            started = time.perf_counter()
            response = await post_chat(client, gateway, key)
            elapsed = (time.perf_counter() - started) * 1000
            assert response.status_code == 504
            idle = await gateway_idle(client, gateway, operator)
            lingering = (await client.get(backend + "/_probe/stats")).json()
            assert lingering == {"calls": 1, "active": 1, "peak": 1}
            await client.post(backend + "/_probe/release")
            await wait_json(client, backend + "/_probe/stats", lambda x: x["active"] == 0)
            recovery = await post_chat(client, gateway, key)
            assert recovery.status_code == 200
            after = await gateway_idle(client, gateway, operator)
            assert after["status"]["exact"]["504"] - before["status"]["exact"]["504"] == 1
            assert after["counts"]["upstream_attempts"] - before["counts"]["upstream_attempts"] == 2
            rows.append(
                {
                    "cycle": index + 1,
                    "configured_upstream_deadline_ms": 150,
                    "status": 504,
                    "client_full_response_ms": elapsed,
                    "gateway_inference_after_timeout": idle["gauges"]["inference_in_flight"],
                    "uncooperative_backend_jobs_after_timeout": lingering["active"],
                    "recovery_status": recovery.status_code,
                    "actual_upstream_calls": 2,
                    "final_metrics": after,
                }
            )
            resources.append(owned_process_sample(pid))
        return {
            "cycles": cycles,
            "rows": rows,
            "resource_samples": resources,
            "deadline_scope": (
                "Cooperative upstream deadline; cleanup/scheduling can extend elapsed time"
            ),
        }


async def run(*, cancellation_cycles: int = 5, timeout_cycles: int = 3) -> dict:
    manifest = source_manifest()
    started = datetime.now(UTC).isoformat()
    async with httpx.AsyncClient(
        trust_env=False,
        timeout=8.0,
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=64),
    ) as client:
        quota = await quota_phase(client)
        capacity = await capacity_phase(client)
        cancellations = await cancellation_phase(client, cycles=cancellation_cycles)
        timeouts = await timeout_phase(client, cycles=timeout_cycles)
    if source_manifest() != manifest:
        raise RuntimeError("Reliability source changed during execution")
    return {
        "schema_version": 1,
        "backend": "disposable deterministic HTTP fixtures",
        "real_model_inference": False,
        "transport": "loopback_http",
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_manifest": manifest,
        "source_manifest_unchanged": True,
        "quota": quota,
        "capacity": capacity,
        "cancellation": cancellations,
        "timeout": timeouts,
        "limitations": [
            "One process-local gateway, one event loop; no distributed correctness claim.",
            "Driver and server processes share hardware; this is correctness evidence, not RPS.",
            "Brief RSS/fd samples and finite cycles cannot prove absence of long-term leaks.",
            "Fixture ignores socket disconnects; gateway cancellation cannot stop remote work.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancellation-cycles", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--timeout-cycles", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output == ROOT / "artifacts":
        parser.error("--output must be a file inside artifacts/")
    report = asyncio.run(
        run(cancellation_cycles=args.cancellation_cycles, timeout_cycles=args.timeout_cycles)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "quota_rejections": report["quota"]["gateway_metrics"]["status"]["exact"]["429"],
                "cancellation_cycles": args.cancellation_cycles,
                "timeout_cycles": args.timeout_cycles,
            }
        )
    )


if __name__ == "__main__":
    main()
