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

## 2 — Genuine local LLM integration (complete)

- [x] Install/run one local inference backend sequentially; use Ollama Gemma 3 1B
  if feasible, otherwise document the actual access/resource blocker.
- [x] Configure model alias and demonstrate genuine inference through the adapter.
- [x] Record model/server versions, hardware, command, sanitized output and latency.

Exit: real inference demonstrated independently of deterministic mock tests.

Verified 2026-10-27, preserving all source, configurations, dependencies, and tests
from foundation commit `971202f`; only README/PLAN changed:

- Official Ollama **0.40.11** Linux archive SHA-256 verified as
  `7bedab90dd122661be78798e9f0edf37fccf0cccf2699d2c58ea1a3afdc04b6d`.
  Extracted into ignored `artifacts/ollama-runtime`, no system service/install.
  Bubblewrap **0.12.0** provided a read-only host, sandbox devices, ephemeral home,
  and repository-backed key/model/temp state without changing `HOME` or proxy/CA
  variables. Native startup would otherwise create `~/.ollama` outside the project.
- Genuine `gemma3:1b` pull returned success: 999.89M parameters, GGUF `Q4_K_M`.
  Manifest digest `97558784bdc9175075a72655bf360b15da17df6b3b9e5182af9f8c0c8e58ba7c`;
  806,058,400-byte weight layer independently hashed as
  `2f8c3d137f2496dcb4019e17a50f2974b89c1be1a742db3c80cc86f91cafeeef`.
- CPU only, one loaded model and one inference at a time, 2,048-token runtime
  context. `/api/ps`: 881,076,469 bytes loaded; VRAM 0. Sampled cgroup peak
  **5.65 GiB** including file cache; peak anonymous memory **0.964 GiB**.
  `memory.events` max/oom/oom_kill all zero; no installation/access/OOM blocker.
  Feasibility applies to this short sequential workload, not full model context
  or concurrent inference. macOS/Metal remains untested.
- Direct physical-model request and authenticated gateway alias `gemma` request
  both returned **200, `Paris`**, stop, reported usage **23/2/25**; request IDs
  `chatcmpl-325` / `chatcmpl-613`. Elapsed samples **0.962 / 0.555 s** are not
  a performance benchmark. Model alias/response filtering matched the contract.
  Sampling parameters **temperature 0.2, top_p 0.8, seed 42, cap 32, stream false**
  were accepted; runner logs confirmed temperature/top_p; existing tests establish
  exact parameter forwarding. A cap-1 request reported **1 completion token**, length;
  system/user role handling returned **`Tokyo`** for the capital of Japan.
- Real-endpoint checks: missing/invalid auth **401**; model listing only `gemma`;
  unsupported physical-name alias **404**; streaming **422**. Stopping actual
  Ollama produced **503 backend_unavailable**, while liveness stayed **200**.
  Restarting the same backend restored **200 `Paris`** through the same gateway.
- Temporary, ignored TOML copies tested 50 ms and 1 s deadlines through a separate
  HTTP gateway. Results **504 in 0.082 s / 1.567 s**. The 50 ms probe did not
  establish inference start; the 1 s warmed-model probe did: real runner task,
  upstream cancellation, and successful subsequent direct completion. Observed
  client wall-time overshoot means this cooperative adapter deadline is not a
  hard end-to-end wall-clock bound.
  No retries or fallback; all verification servers cleaned up. A later direct
  cold/reload request took **52.14 s**; no model-performance guarantee is inferred.
- Quality failure retained: **17 + 25 → `32`**, HTTP 200 with usage 24/4/28.
  Real inference and transport correctness do not imply answer correctness.
- Reproduction/install/request/fault commands and sample outputs are in README.
  Raw local JSON/logs and verification drivers are ignored under
  `artifacts/milestone02/`; the original Ollama TOML comment predates verification.

Final regressions: `.venv/bin/pytest` **58 passed in 1.04 s**;
`.venv/bin/ruff check .` **all checks passed**;
`.venv/bin/ruff format --check .` **13 files already formatted**;
`git diff --check` **clean**. Ports 11434/8000/8001 closed after cleanup.
No caching, quotas, providers, dashboards, or optimizations added.

## 3 — Exact cache (next; not started)

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
