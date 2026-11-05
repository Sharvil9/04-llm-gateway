"""Reporting must preserve undersampled tails and avoid inference during capture."""

import json
import sys
import tempfile
from pathlib import Path

import httpx

from devtools import load_report
from devtools.smoke import ROOT


def test_trial_quantiles_are_withheld_when_any_trial_is_undersampled():
    points = [
        {
            "scenario": "uncached",
            "concurrency": 16,
            "successful_rps": rate,
            "latency_successful": {"p50_ms": latency, "p95_ms": latency * 2, "p99_ms": tail},
            "status_counts": {"200": successes, "503": 1000 - successes},
            "independent_upstream_calls": successes,
            "cache_dispositions": {"BYPASS": 1000},
        }
        for rate, latency, tail, successes in [(200, 10, 50, 1000), (100, 20, None, 500)]
    ]
    result = load_report.aggregate(points)[0]
    assert result["successful_rps_median"] == 150
    assert result["median_of_trial_successful_latency_quantiles"]["p99_ms"] is None
    assert result["median_of_trial_successful_latency_quantiles"]["p50_ms"] == 15
    assert result["statuses"] == {"200": 1500, "503": 500}


def test_runtime_capture_sends_metadata_gets_and_does_not_add_chat_calls(monkeypatch):
    seen = []

    def backend(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"version": "fixture-metadata"})

    real_client = httpx.Client

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        return real_client(transport=httpx.MockTransport(backend), **kwargs)

    monkeypatch.setattr(load_report.httpx, "Client", client)
    with tempfile.TemporaryDirectory(prefix="report-capture-", dir=ROOT / "artifacts") as tmp:
        directory = Path(tmp)
        log = directory / "ollama.log"
        log.write_text('[GIN] POST "/v1/chat/completions"\n')
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "load_report",
                "--capture-runtime",
                "--input-root",
                str(directory),
                "--ollama-log",
                str(log),
            ],
        )
        load_report.main()
        result = json.loads((directory / "model-runtime.json").read_text())
        assert result["dedicated_log_chat_POSTs"] == 1
    assert seen == [("GET", "/api/version"), ("GET", "/api/tags"), ("GET", "/api/ps")]
