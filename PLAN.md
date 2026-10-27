# Milestone plan

Implement and verify one milestone at a time; checkboxes require evidence.

## 0 — Isolated foundation

- [x] Confirm `/workspace` is not a Git repository; no `/AGENTS.md` or
  `/workspace/AGENTS.md`; create only `/workspace/04-llm-gateway` and initialize
  independent Git on `main`.
- [x] Inspect hardware/tools/network. Linux x86_64, Python 3.12.14, 8 GiB cgroup
  memory, 2 CPU cores' quota (3 visible logical CPUs), uv/Git/curl available;
  Ollama/GPU tools absent. This is not the target 24 GB macOS machine.
- [x] Establish operating instructions and architecture; verify essential choices
  against authoritative docs before finalizing the foundation.

## 1 — Authenticated HTTP gateway (complete)

- [x] Strict supported chat schema, configured model aliases, fail-closed bearer
  authentication and input-size limits.
- [x] Shared async HTTP adapter, deterministic HTTP mock, normalized responses,
  bounded timeouts/responses, sanitized errors and cancellation cleanup.
- [x] Focused automated contract/failure tests, lint, and a real loopback HTTP
  smoke demonstration. Lock dependencies and record exact verification commands.
- [x] README installation/configuration/examples and honest current limitations.

Exit: tests and lint pass; unauthorized/invalid requests are rejected; an
authenticated request crosses the gateway and mock HTTP services successfully.
No cache, rate-limit, or model-throughput claims at this gate.

Verified 2026-10-27 in the environment recorded above:

- `uv sync --locked --cache-dir .cache/uv`: isolated environment installed from
  `uv.lock`; FastAPI 0.141.1, Starlette 1.10.0, Pydantic 2.14.1, HTTPX 0.28.1,
  Uvicorn 0.52.4. Dev tools: pytest 9.1.1, pytest-asyncio 1.4.1, Ruff 0.17.3.
- `.venv/bin/pytest`: **58 passed**. Includes schema/auth/input-size checks,
  backend credential separation, fixed routing/no redirects, normalized usage,
  backend errors/invalid bodies, total deadline/no retry, cancellation cleanup,
  six concurrent distinct requests and lifespan shutdown. Added regression checks
  for invalid backend URLs and boolean response indices after focused review.
- `.venv/bin/ruff check .`: all checks passed;
  `.venv/bin/ruff format --check .`: 13 files already formatted.
- `.venv/bin/python -m devtools.smoke`: two actual loopback Uvicorn processes;
  unauthorized **401**, unsupported streaming **422**, unknown model **404**,
  model listing **200**, normalized deterministic completion **200**.
  `real_model_inference=false`; both processes and temporary config cleaned up.
- `git diff --check`: no whitespace errors. No large models downloaded and no
  performance experiment performed. Socket-disconnect detection is still pending;
  the cancellation test explicitly cancels the request task.

## 2 — Genuine local LLM integration (next)

- [ ] Install/run one local inference backend sequentially; use Ollama Gemma 3 1B
  if feasible, otherwise document the actual access/resource blocker.
- [ ] Configure model alias and demonstrate genuine inference through the adapter.
- [ ] Record model/server versions, hardware, command, sanitized output and latency.

Exit: real inference demonstrated independently of deterministic mock tests.

## 3 — Exact cache

- [ ] Conservative eligibility, client/context/model/settings/policy keys,
  TTL/capacity, and in-flight coalescing.
- [ ] Test hits, bypasses, isolation, expiry, eviction and concurrent cancellation.

Exit: upstream call reduction measured with no cross-boundary reuse.

## 4 — Conservative semantic cache

- [ ] Small CPU embedding model, bounded embedding execution and vector lookup.
- [ ] Explicit opt-in/domain constraints and equivalence safeguards.
- [ ] Curated calibration/held-out positives and adversarial negatives; measured
  threshold/precision/recall/false hits; keep disabled if safety gate fails.

Exit: explainable eligible hits and zero held-out false hits with limitations.

## 5 — Traffic control, failure handling, metrics

- [ ] Atomic request token buckets, client in-flight limits, global upstream
  capacity and bounded queueing; correct 429/503 and retry guidance.
- [ ] Test saturation, races, timeouts, cancellation and permit cleanup.
- [ ] Protected metrics, sanitized structured logs and usage provenance.

Exit: concurrent tests demonstrate quotas and capacity bounds, with metrics.

## 6 — Reproducible performance evidence

- [ ] Lightweight HTTP benchmark harness; fixed seeds, warmup, hardware/version
  capture and raw JSON results for scenarios A–F.
- [ ] Separate mock overhead/cache measurements from real-model inference results.
- [ ] Report RPS, median/p95, errors, hit/false-hit rates, quota correctness,
  upstream-call reduction; optimize only observed bottlenecks.
- [ ] Final portfolio demonstration and concise reproducible documentation.

Exit: completion gates in `SPEC.md` all have evidence; full project is complete
only here.
