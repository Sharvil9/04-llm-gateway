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

## 3 — Bounded admission and client quotas (complete)

- [x] Configurable per-client monotonic token buckets, fixed authenticated IDs,
  default policy and client overrides; accurate integer Retry-After for 429.
- [x] Configurable global active-operation limit, immediate 503 on saturation,
  zero admission waiters; no guessed capacity Retry-After.
- [x] Release slots after success, errors, timeouts, caller cancellation and ASGI
  disconnect, including cancellation/deadline overlap during cleanup.
- [x] Reproducible actual HTTP enforcement and timeout-scope probes; investigate
  prior external-latency discrepancy and retain scope limitations.
- [x] Preserve the API/model-routing contract and genuine Gemma integration.

Verified 2026-10-28 on the same Linux/8 GiB/two-CPU-quota runner, no new packages:

- `devtools.traffic_probe` over real loopback HTTP, deterministic held fixture:
  **24 simultaneous client requests**; A burst 3 → **3 HTTP 200 / 9 HTTP 429**,
  B override burst 2 → **2 HTTP 200 / 10 HTTP 429**. Exactly **5 backend calls**;
  positive integer Retry-After, isolated client quotas. With capacity 2, **20
  excess requests all returned 503**, calls/peak stayed **2** until release,
  and recovery returned **200** (3 total backend calls). No admission queue,
  generated credentials stay in child environment, subprocess/config cleanup.
- Automated concurrent HTTP-boundary tests additionally reject **64** simultaneous
  excess requests without new upstream calls, verify invalid attempts spend quota,
  exclude auth failures/model listing/health, and test refill with an injected
  clock. No request can allocate arbitrary new bucket state.
- Cancellation tests inject ASGI `http.disconnect` and cancel request tasks;
  slow response cleanup remains occupied beyond a 40 ms deadline, a competing
  request receives 503, cleanup completes, permit reaches zero, and recovery
  succeeds. These are ASGI-message/task tests, not a claim of raw TCP-close testing.
- Historical 1 s → **1.5665 s** recorded only external duration. The extra 567 ms
  cannot be causally partitioned from those artifacts. New request-local
  `Server-Timing` separates middleware entry→response-start and complete adapter elapsed
  (including preparation/validation/cleanup); neither is pure CPU/model time.
- A concrete deadline defect was separately reproduced against commit `331cf1f`:
  deliberately blocking mock transport, **5 ms budget**, old **200 in 31.4 ms**,
  fixed **504 in 30.7 ms**. Async timers cannot preempt synchronous work; the fix
  rejects late success after cleanup. A second fix disarms outstanding deadline
  cancellation before response close, preventing cleanup interruption on a
  disconnect; raw identity-only stream iteration avoids HTTPX iterator auto-close
  under the expiring timer. No hard client-latency promise.
- `devtools.deadline_probe` uses real HTTP chunks every 50 ms against a 1 s total
  budget. Client, gateway and adapter milliseconds are reported separately in
  ignored `artifacts/deadline-probe.json`: final client **1003.323 ms**, gateway
  **1001.801 ms**, adapter **1001.043 ms**, HTTP **504**, **20 chunks sent**.
  Slow-cleanup/trickle/blocking unit tests
  verify scope and resource release. This is not a model/throughput benchmark.
- Genuine **Ollama 0.40.11 / gemma3:1b** regression after the changes: authenticated
  alias `gemma` returned **200 `Paris`**, usage **23/2/25**, request ID
  `chatcmpl-679`, elapsed **2.790 s**; gateway/adapter headers **2788.944 / 2787.865 ms**.
  Used the unchanged model digest from milestone 2, one CPU model at context 2048,
  new Ollama capacity 1/burst 10/refill 0.2 settings; servers stopped afterward.

Reproduction:

```sh
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m devtools.smoke
.venv/bin/python -m devtools.traffic_probe --output artifacts/traffic-probe.json
.venv/bin/python -m devtools.deadline_probe --output artifacts/deadline-probe.json
```

Final aggregate run: **85 tests passed in 1.46 s**; Ruff lint passed; **20 files
already formatted**; `git diff --check` clean. Foundation HTTP smoke returned
401/422/404/200/200 as expected; both committed loopback probes passed. No dependency
changes. Verification subprocesses were stopped and temporary configs removed.
Limits are in-memory, single-process/single-loop, reset on restart; no fairness,
distributed correctness, ingress connection/upload bound, or control of remote
compute continuing after HTTP cancellation. Cleanup may extend the budget.
No cache, extra provider, dashboard, infrastructure, or performance optimization.

## 4 — Safe exact response caching (complete)

- [x] Conservative operator approval plus client opt-in; HMAC client/model/
  configuration/credential/policy fingerprints without plaintext prompt keys.
- [x] Process-local fixed TTL, LRU eviction, entry and serialized-byte budgets;
  immutable response bytes; counters and no persistence/background service.
- [x] Lookup after auth/quota before admission; complete-response commit after
  successful final ASGI send, observed-abort checks and fingerprint recheck.
- [x] Deterministic TTL/eviction/isolation/model/configuration/race/failure tests;
  verify hits still spend quota and work while inference capacity is occupied.
- [x] Reproducible HTTP fixture and genuine Gemma upstream-call reduction.

Exit: upstream call reduction measured with no cross-boundary reuse.

Verified 2026-10-29, preserving single-loop traffic controls and the API subset:

- Standard-library HMAC/OrderedDict, no dependency changes. Default disabled;
  approved literal single-user public/static prompts only, explicit `exact`
  request header, temperature 0/top_p 1/seed, pinned model and deployment revisions.
  Other contexts, sampling, private/unapproved prompts bypass by default; no
  semantic classifier or automatic privacy guarantee for misapproved input.
- Full canonical validated request/defaults, stable client ID, all gateway
  configuration and actual backend credential hash partition keys. Tests cover
  parameter, system/history, model/backend revision, namespace and policy changes.
  A final recheck prevents writes if configuration changes during the request.
  Revisions require operator-managed immutable deployments; no live update detector.
- TTL is insertion-based, non-sliding; LRU/expiry reclaim payloads within entry
  and digest+JSON byte budgets. Oversized responses skip without evicting existing
  entries. Expiry is lazy; idle expired bytes may persist in RAM, never returned.
  No exact RSS/secure-memory-erasure promise; values are plaintext responses.
- Only successful nonblank `stop` responses are candidates. No error/truncation/
  cancellation/disconnect/send-failure inserts; successful ASGI send is not proof
  of client receipt. Backend session/freshness/cache-control headers disallow
  insertion. Cookie persistence is rejected so sessions cannot leak across clients
  or change cache lookup context. Client cache bypass directives are honored.
- Deliberately **no coalescing**: duplicate cold misses remain subject to global
  admission and may duplicate inference; first fresh writer wins without TTL reset.
  Tests hold two misses in flight, reject an excess request, and verify warm hits
  create no upstream work or admission slots. Hit requests still consume quota.
- Internal hits/misses/bypasses/stores/expiration/eviction/upstream-attempt counters;
  no metrics dashboard/API. Independent fixture/access-log request counts verify
  actual call reduction. Cached id/created/usage remain the original generation.
- `devtools.cache_probe` over loopback HTTP: client A **MISS/HIT/HIT**, client B
  **MISS/HIT**, one sampled A **BYPASS** → **6 successful requests, 3 cache hits,
  2 misses, 1 bypass, 3 independently counted backend calls**, versus 6 uncached
  reference calls. Warm unauthorized request stays **401**, no extra backend call.
  Generated credentials stay in environment, temporary servers/configs cleaned up.
- Genuine Ollama **0.40.11 / gemma3:1b Q4_K_M**: live digest checked against
  configured pin `97558784bdc9175075a72655bf360b15da17df6b3b9e5182af9f8c0c8e58ba7c`.
  Three eligible requests returned **200 `Paris`**, **MISS/HIT/HIT**, byte-identical
  id/created/usage (original **23/2/25**), and exactly **one** Ollama chat POST in
  its dedicated access log: **two inference calls avoided**. Initial request ID
  `chatcmpl-718`; elapsed samples **5.289 s / 1.624 ms / 1.703 ms** are functional
  observations, not throughput or model-generation performance measurements.
  Local evidence: ignored `artifacts/milestone04/real-gemma-cache.json` and log.

Reproduce on a dedicated Ollama server using the README setup:

```sh
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m devtools.cache_probe --output artifacts/cache-probe.json
.venv/bin/python -m devtools.cache_probe --ollama-log artifacts/milestone02/ollama.log \
  --output artifacts/real-cache-probe.json
```

Final regressions: **141 passed in 6.73 s**; Ruff lint passed; **24 files already
formatted**; `git diff --check` clean. All prior HTTP smoke, quota/capacity and
deadline probes passed; the new fixture cache probe passed. No dependency changes.
The committed `--ollama-log` mode also passed against a fresh dedicated server:
version **0.40.11**, live digest and loaded context **2048** matched the example,
**MISS/HIT/HIT**, exactly **1** upstream POST, original usage **23/2/25**, response
ID `chatcmpl-688`, identical replay bytes. JSON report under
`artifacts/milestone04/real-cache-probe.json`; all verification servers stopped,
temporary configs removed, and cgroup OOM events remained zero.
No semantic caching, embeddings, Redis, dashboard or distributed consistency
added. Counters/cache reset on restart; one event loop/worker only.

## 5 — Conservative semantic cache (next; not started)

- [ ] Small CPU embedding model, bounded embedding execution and vector lookup.
- [ ] Explicit opt-in/domain constraints and equivalence safeguards.
- [ ] Curated calibration/held-out positives and adversarial negatives; measured
  threshold/precision/recall/false hits; keep disabled if safety gate fails.

Exit: explainable eligible hits and zero held-out false hits with limitations.

## 6 — Operational metrics and reproducible performance evidence

- [ ] Protected metrics, sanitized structured logs and usage provenance.
- [ ] Lightweight HTTP benchmark harness; fixed seeds, warmup, hardware/version
  capture and raw JSON results for scenarios A–F.
- [ ] Separate mock overhead/cache measurements from real-model inference results.
- [ ] Report RPS, median/p95, errors, hit/false-hit rates, quota correctness,
  upstream-call reduction; optimize only observed bottlenecks.
- [ ] Final portfolio demonstration and concise reproducible documentation.

Exit: completion gates in `SPEC.md` all have evidence; full project is complete
only here.
