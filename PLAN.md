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

## 5 — Evaluation-driven semantic cache (complete)

- [x] Freeze public/static equivalence and hard-negative cases before scoring;
  evaluate two pinned small CPU embedding models sequentially.
- [x] Explicit semantic opt-in, reviewed phrase families/answer requirements,
  tenant/model/credential/configuration/generation compatibility, exact lookup first.
- [x] Shared bounded TTL/LRU response/vector storage, no alias TTL extension;
  one off-loop embedding job, no queue, cancellation-safe capacity and cleanup.
- [x] Calibration-only threshold selection, held-out positive/negative results,
  separately report unsafe similarity-only retrieval and guarded reuse.
- [x] Regression/failure tests, loopback HTTP call counts, real Gemma examples,
  separate semantic lookup timing, reproducible commands and limitations.

Verified 2026-10-29 on Linux x86_64 / Python 3.12.14 / 8 GiB / two CPU cores' quota:

- Frozen dataset commit **`ac41d4a`**, SHA256
  `80e0ba7b0acaf85afaca2e9bd8fa468c426fe67d50acdfa07541bb98d1a04885`:
  **3 families, 54 queries**, calibration 27 / held-out 27, each 9 positives and 18
  negatives. Covers negation, entities, numbers, versions, instructions, context,
  output contracts and freshness. No label/wording/membership changes after scores.
- Optional extra only: **NumPy 2.5.4, ONNX Runtime 1.31.1, Tokenizers 0.23.3**; no
  Torch/Transformers, services or vector database. Fixed HF revisions/tokenizer
  blobs and graph hashes independently verify downloaded local artifacts; no
  runtime downloads. Source models Apache-2.0, pinned Xenova quantized conversions.
- Sequential candidate evaluation, 384-dimensional normalized vectors, masked
  mean pooling, CPU/one ORT thread, no truncation. **L6** revision
  `751bff37182d3f1213fa05d7196b954e230abad9`: graph 22,972,370 bytes, max 256 tokens;
  load 397.257 ms, process peak 117.94 MiB, 54-query encode median/p95 **2.732/5.153 ms**.
  **Selected L3** revision `4b544e74dfc3256b2b56849ea5d7064fee1ac846`:
  graph 17,452,106 bytes, max 128 tokens including special tokens; load 293.654 ms,
  process peak 111.29 MiB, encode **1.425/2.253 ms**. Memory includes Python/libs/runtime,
  not isolated model allocations; timings include tokenization/pooling. NumPy
  uses the platform BLAS build; one ORT thread is not a total-process thread cap.
  L3 matches guarded calibration recall with lower observed memory/latency.
- Fixed threshold grid 0.65/0.70/0.75/0.80/0.85/0.90/0.95/0.98/1.0. Choose highest
  threshold tied for maximum guarded calibration recall at zero false hits:
  **0.75** for both models. Selected L3 calibration **9/9 positives, 0/18 negatives**;
  unchanged held-out **9/9 positives, 0 misses, 0/18 negatives**, precision 9/9.
  Across both splits guarded false hits **0/36**. **Similarity alone is unsafe**:
  L3 at 0.75 accepts 11/18 held-out negatives (precision 9/20 = 45%). L6's raw
  calibration-selected 0.85 also fails (7/9 positive hits, 7 false held-out hits).
  Full per-case scores/calibration curves/memory/versions are tracked in
  `evaluation/l3_results.json` and `evaluation/l6_results.json`.
- Safety is finite reviewed-policy enforcement, not open-domain equivalence or
  answer-quality proof. Unknown wording bypasses. Five negatives (three held-out)
  distinguish identical reviewed text through external scope/answer metadata.
  Component evaluator checks that metadata. A separate synthetic-vector ASGI
  boundary test maps those five into explicit supported system context and
  verifies all 54 cases: **18 byte-identical correct-answer reuses, 36 bypasses**;
  3 warmed sources + 36 uncached negatives = 39 upstream attempts. Bare approved
  single-user text remains eligible; invisible context cannot be inferred.
- Existing exact protections remain: auth/quota first, single-user approved
  public/static input, temperature 0/top_p 1/seed, immutable deployment pins,
  per-client HMAC fingerprints, no cookies/cache directives, complete stop-only
  answers, final ASGI send and configuration recheck. Vectors attach to entries
  and share TTL/LRU/total bytes; another 32-byte partition + 1,536-byte vector/tag,
  maximum 128 tagged entries. No standalone index, alias promotion, TTL chaining
  or coalescing. Exact-only entries are not retroactively embedded on exact hits.
- One dedicated worker, zero queued jobs; busy/error/overlength paths fall through
  to ordinary admitted inference. Cooperative 250 ms lookup budget, monotonic late
  check, native capacity retained after caller cancellation/timeout; shutdown waits
  actual job completion. Token/graph output validation and artifact tamper tests.
  Default disabled; no distributed consistency, hard native preemption or live
  backend/model pin attestation. Configuration changes require new approval/eval.
- Actual ONNX + deterministic backend loopback probe: **8 HTTP 200**, **1 exact hit,
  2 semantic hits, 3 misses, 2 bypasses**, independently counted **5 upstream calls**.
  Client B and sampling reach upstream, wrong entity bypasses; replay bytes match.
  Baseline is one call/request; 3 calls avoided (37.5%), not generation throughput.
- Genuine **Ollama 0.40.11 / gemma3:1b**, verified manifest
  `97558784bdc9175075a72655bf360b15da17df6b3b9e5182af9f8c0c8e58ba7c`, context 2048,
  with actual L3 embeddings: **Paris/Tokyo/Berlin**, same 8-request classifications
  and exactly **5 chat POSTs** in a dedicated quiet Ollama log. Byte-identical
  Paris/Tokyo semantic replays and Paris exact replay; original Paris usage 23/2/25,
  ID **`chatcmpl-562`**. Final semantic-hit lookup samples **4.271/13.504 ms**, no
  adapter invocation; their source adapter samples **4585.997/1380.282 ms** include
  inference HTTP preparation/read/cleanup and possible loading. These are separate
  functional observations, not comparative model-generation benchmarks.
  Tracked `evaluation/gemma_results.json` records actual sample responses/timing;
  raw dedicated log is ignored under `artifacts/milestone05/ollama.log`.
- **195 tests passed in 3.36 s**, Ruff lint passed, **33 files formatted**. Core-only
  `uv sync --locked` verification also passed: **168 passed, 6 skipped** (the NumPy
  integration module is skipped as a unit); extra restored afterward. All previous
  smoke, traffic/admission, deadline and exact-cache HTTP probes passed. Final
  deadline probe 504: client 1016.877 ms / adapter 1013.225 ms / 18 trickle chunks;
  cooperative cleanup/scheduling scope remains documented. No OOM/max events.

Reproduce from the repository (models sequential; dedicated Ollama setup in README):

```sh
uv sync --locked --extra semantic --cache-dir .cache/uv
.venv/bin/python -m devtools.semantic_eval --validate-only
.venv/bin/python -m devtools.embedding_models --model l6
.venv/bin/python -m devtools.embedding_models --model l3
.venv/bin/python -m devtools.semantic_eval --model-dir artifacts/embeddings/l6 \
  --output artifacts/l6-evaluation.json
.venv/bin/python -m devtools.semantic_eval --model-dir artifacts/embeddings/l3 \
  --output artifacts/l3-evaluation.json
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m devtools.semantic_probe --output artifacts/semantic-probe.json
.venv/bin/python -m devtools.semantic_probe --ollama-log artifacts/milestone05/ollama.log \
  --output artifacts/real-semantic-probe.json
```

Remaining: human-curated finite vocabulary, no generalization/privacy classifier;
finite tests cannot guarantee universal correctness. A direct reviewed-family
alias map would be cheaper for this catalog; embeddings supply an evaluated extra
retrieval filter, not proof of unseen equivalence. Approval/configuration/model
changes need evaluation. Process-local state, lazy expiry, byte budget excludes
Python/RSS overhead; native jobs cannot be killed. Comprehensive metrics and
scenario A–F throughput benchmarks remain milestone 6. No Redis, embeddings service,
new LLM provider, training, dashboard, distributed state or unrelated optimization.

## 6 — Observability and reproducible benchmarking (complete)

- [x] Operator-only fixed-cardinality metrics and optional sanitized JSON logs;
  completion-aware outcomes, quota/capacity/deadline counts and usage provenance.
- [x] Bounded closed-loop HTTP benchmark driver; fixed scheduling, warm-up,
  sample-aware percentiles, source/lock manifests and independent upstream counts.
- [x] Separate direct/mock, gateway, cache, mixed and genuine Gemma measurements;
  record process CPU/RSS and preserve raw machine-readable evidence.
- [x] Semantic provenance audit and frozen unseen-wording follow-up; preserve
  existing eligibility and report the limits of curated vocabulary honestly.
- [x] Final regressions, HTTP enforcement/recovery/failure probes, reproducible
  commands and performance table in README.

Verified 2026-10-30 on Linux x86_64, Python 3.12.14, **8 GiB / two CPU cores' quota**
(three visible CPUs), one Uvicorn worker. No dependencies or production performance
optimizations added. Ollama **0.40.11 / gemma3:1b Q4_K_M**, existing manifest
`97558784bdc9175075a72655bf360b15da17df6b3b9e5182af9f8c0c8e58ba7c`, context 2048,
CPU only, one loaded model. macOS/Metal remains untested.

Metrics/security:

- `GET /v1/metrics` requires a distinct operator credential from
  `GATEWAY_METRICS_KEY`; disabled 404 if absent. Ordinary client keys cannot read
  aggregate state, operator keys cannot infer. No client/model/path labels,
  prompts, credentials, response bodies or retained raw metric samples.
- Completion scope includes authentication/schema/quota/backend errors; health,
  models and metrics scrapes do not spend quota or count themselves. Fixed counters
  distinguish response-start status from successful final sends, aborts, cache
  dispositions, upstream attempts and reported usage. Histograms use fixed buckets;
  rates expose denominators. Only new upstream usage is added, not cached history.
- An independent review found consistent large provider usage could exceed the
  telemetry 64-bit bound and turn 200 into 500. Fixed without narrowing the API:
  preserve response/cache bytes and increment ignored_usage_reports. Regression
  covers this boundary. Optional JSON logging is off during measured trials.
- Header/histogram stages distinguish gateway-to-start, exact, semantic, adapter,
  residual and request lifecycle. Residual contains scheduling, not isolated CPU;
  lifecycle includes final send/cache commit before metric update.

Recorded trials, source frozen and verified unchanged end-to-end:

- Benchmark source SHA256
  `890d3a1d43383d709eb7729b502deb649fb3cf8e2bc8a50bfa217b1eba6016fd`;
  raw reports contain the complete runtime/helper/lock manifests and UTC windows.
- **20 mock points**: 1,000 requests each, concurrency 1/4/8/16, deterministic 10 ms
  backend, direct/no-cache/warm-exact/warm-semantic/40-40-20 mixed. Fresh gateway
  per point, five sequential warm-ups plus cache seed, excluded from measurements;
  seed 42, request cap 32 output tokens, temperature 0/top_p 1/seed 42. High benchmark
  request quotas avoid accidental 429; capacity 8 and zero waiting queue remain.
- **Real trials**: 30 direct + 30 uncached gateway requests (p50 only); 1,000 exact
  + 1,000 semantic warm replays (p50/p95/p99); 100 mixed requests (p50/p95). All at
  concurrency 1/capacity 1. Exactly **30/30/0/0/20 independently counted chat POSTs**
  during the respective measured windows. Warm-up/seed calls are excluded.
- Final c1 mock results: direct **56.67 RPS, 14.47 ms p50**; gateway uncached
  **54.90 RPS, 17.30 ms**; exact **245.97 RPS, 3.60 ms**; semantic **150.71 RPS, 5.51 ms**;
  mixed **124.98 RPS, 5.71 ms**, 200 upstream calls. Gateway residual median 0.446 ms.
- Final real results: direct **1.81 RPS, 539.42 ms p50**, gateway uncached
  **1.76 RPS, 569.35 ms**; exact **283.22 RPS, 2.97/6.75/9.60 ms p50/p95/p99**;
  semantic **112.23 RPS, 7.93/16.35/21.23 ms**. Cached RPS is gateway replay, not new
  model generation. Mixed **8.50 RPS, 6.50/575.86 ms p50/p95**, 40 exact + 40 semantic
  hits, 20 no-store, 20 actual upstream calls: 80 calls avoided for this workload.
- Real model CPU averaged **1.84 cores** for uncached gateway inference; median
  adapter 566.341 ms vs gateway residual 0.770 ms. Real semantic lookup median 3.301 ms.
  Gateway sampled RSS **44.5 MiB uncached / 44.3 exact / 122.9 semantic**; model tree
  **~902 MiB** including during hits, when model CPU stayed below 0.03 core.
  Caching avoids inference work, not model residency. CPU ticks/RSS are sampled,
  shared pages may be counted twice, brief spikes may be missed, discovery adds
  measurement work. Client/sampler/servers share the same CPU budget.
- Mock uncached c8 **239.18 successful RPS**, no errors; c16 **316.81 attempted /
  211.32 successful RPS**, 333 HTTP 503, 667 upstream calls. Permits remain bounded.
  Concurrent semantic opt-in c4 was **995 exact hits, 2 semantic hits, 3 misses**;
  never label that result pure semantic serving. Warm controlled replays compare
  response SHA256; all HTTP 200 city-answer checks passed, no transport failures.
- Only one final observed trial per point: shared-host scheduling, sequential
  run order and load-generator work limit causal inference. Direct/gateway
  differences are not pure CPU costs; no optimization/speedup claim follows.
  Higher concurrency has noisy tails and saturation; profile/repeat before tuning.

A first sampler wrongly observed only bubblewrap because this kernel lacks
`/proc/PID/task/TID/children`. Excluded those resource results. Added bounded
GNU `ps --ppid` discovery for owned trees, completeness/null reporting, cap 32
processes/128 identities and one serial off-loop sampling job. Tests cover the
missing-proc path and cancellation. Reran **all four final suites** afterward;
real model resource windows observed all three processes with complete discovery.
An earlier setup trial was also interrupted before final source freeze and is
excluded. Final source manifests now reject changes during a suite.

Reliability and semantic audit:

- Real loopback quota probe: 24 attempts, client A 3 accepted/9 HTTP 429, B 2/10;
  exactly 5 upstream calls. Protected metrics agree: 24 completed, 5 success,
  19 quota errors. Positive Retry-After and client isolation verified.
- Held capacity 2: 20 excess requests all 503, no guessed Retry-After; upstream
  calls remain 2 until release. Recovery 200, 3 calls total, 23 completed responses,
  metrics report 20 overloads and zero remaining inference permits.
- Real HTTP trickle probe: 1 s cooperative upstream budget returns 504 in 1009.257 ms
  external / 1002.947 ms adapter; metrics record one timeout/attempt and no held slot.
  Cancellation, disconnect, slow cleanup, failed final sends and cache isolation
  remain covered by regressions. HTTP smoke, exact and actual-ONNX semantic probes
  passed; semantic probe independently counts 5 calls / 8 requests.
- Final audit frozen in commit **`95cba86`**, SHA256
  `b819653ab102a268a1720db80e5f7719a56f9743b7eb305efe46bc68460d4661`.
  Calibration-only threshold arithmetic reproduces M05, but all nine held-out
  positive literals were already approved. A membership-only lookup reproduces
  guarded admission; it is not independent unseen-wording equivalence evidence.
  New 18 phrases were absent from the registry and original cases. Fixed L3/0.75
  retrieves **4/6 equivalents, misses 2, accepts 6/12 negatives** before guards.
  Production bypasses all 18, including 6 true equivalents: served unseen recall 0/6.
  No threshold/approval expansion, no tuning on the final set. Five original
  metadata-only negatives require explicit system context at the wire boundary.
  The follow-up was authored after M05 results, not an independent external study.

Artifacts and commands:

- `benchmarks/summary.json`: all 25 points, durations, percentiles, counts, resource
  aggregates, source/lock hashes and exact commands. README has the concise table.
- `benchmarks/*.json.gz`: full bounded per-request raw trials, metrics snapshots,
  configs, UTC windows, replay hashes and sampler completeness. Integrity hashes
  are in the summary. `benchmarks/reliability.json` preserves the HTTP probes.
- `evaluation/semantic_audit_cases.json` and `semantic_audit_results.json`: frozen
  wording, protocol audit and per-case scores. Both original datasets stay intact.
- Reproduction: README benchmark commands; real commands use the dedicated
  loopback Ollama server with OLLAMA_NUM_PARALLEL=1, MAX_LOADED_MODELS=1,
  CONTEXT_LENGTH=2048, KEEP_ALIVE=1m, NO_CLOUD=1 and the repository-backed sandbox.

Final checks: **262 passed in 5.67 s**, Ruff lint passed, **40 files formatted**,
`git diff --check` clean. Core-only install: **235 passed, 6 skipped in 4.12 s**;
optional extra restored and full regressions rerun. The NumPy integration module
is skipped as a unit without the extra. No OOM/max memory events. All owned
verification servers are stopped afterward.

Highest-value remaining work: validate an independent equivalence gate on truly
unseen requests while maintaining negative safety, or keep curated alias reuse;
measure the target MacBook; use a separate load generator and repeated/randomized
baseline trials plus profiling before tuning concurrency or serialization. The
functional portfolio prototype is measurable and reproducible, with these limits.
No distributed consistency, streaming, native hard-kill, public-Internet security
or general semantic/model-performance guarantee is claimed. No Redis, training,
provider expansion, dashboard or production performance optimization was added.

## 7 — Independent semantic generalization validation (complete; not promoted)

- [x] Diagnose M06 unseen bypasses: literal eligibility rejected all six before
  embedding/retrieval; four exceeded 0.75, two would additionally miss similarity.
- [x] Freeze independently authored final wire cases before gate development:
  `f9686d5`, SHA256 `ad05be1634d5a890e62a74eceedcff4e1c6b2cc5ec218fcff1230077f5684845`.
  Separate development 14 positives/22 negatives; final 24/48. Zero wording overlap.
- [x] Implement/test bounded capital grammar with explicit France/Japan approval,
  strict city-only answer validation and preserved compatibility/security controls.
  Freeze candidate/harness before final inspection: `d9955fa`; fixed L3/0.75.
- [x] Score complete production ASGI pipeline, actual ONNX and source-only reset
  before each case. No final query becomes an approval or retrieval source.
- [x] Capture genuine Gemma sources and suitable HTTP replays; replay actual source
  answers through both datasets. Retain independent call counts and exact samples.
- [x] Apply fail-closed release decision: candidate runtime/config/tests/harness
  reverted on `main`; preserve immutable experiment, data, results and commands.
- [x] Re-run release regressions, traffic/deadline probes and actual Gemma semantic
  integration. Verify frozen-worktree reproduction and document limitations.

Results on 2026-10-30, same Linux CPU-only 8 GiB / two-core quota:

| Pipeline | Valid reuses | Negative hits | Precision |
|---|---:|---:|---:|
| Development baseline | 0/14 | 0/22 | undefined |
| Development candidate | 12/14 | 0/22 | 12/12 |
| Final baseline | 0/24 | 0/48 | undefined |
| Final candidate | 0/24 | 0/48 | undefined |

Fixture and genuine-source replay counts agree. Final **72/72 BYPASS** is eligibility
abstention; no final embedding search occurred. Thus bounded-language coverage
failed, rather than a measured embedding/compatibility failure. No threshold,
grammar or approvals were adjusted using final labels. The candidate demonstrates
development composition only. Reviewed-phrase caching remains the supported policy.

Real candidate HTTP verification: 6 requests, 4 independently counted Ollama POSTs;
canonical `Paris`/`Tokyo`, 2 suitable unseen development replays, Germany `Berlin`
BYPASS, tenant-B cold MISS. Ollama 0.40.11, pinned Gemma 3 1B Q4_K_M, context 2048.
All 12 development captured-source replays met the exact factual city-only oracle.
Final suitability is unobserved at zero reuses, not 100%. No fresh inference occurs
in captured-source ASGI runs; cold responses are no-store fixtures. Existing
release Gemma integration: 8 requests / 5 real calls, 1 exact / 2 semantic hits.
Release HTTP probes also verified 19 quota 429s / 24 attempts (5 upstream calls),
20 excess capacity 503s with recovery and zero held permits, and a controlled 504
at 1006.956 ms external / 1003.277 ms adapter for the 1 s cooperative budget.

Candidate: **314 passing tests**, including 40 gate/security/isolation/TTL/cancellation
checks and 12 evaluator checks. Release retains its 262 regression tests and
unchanged validated runtime (262 passed in 3.45 s; lint/format clean). No OOM/max
memory events; owned services and temporary worktree cleaned up. README records
exact frozen-worktree commands; actual
held-out genuine-source reproduction was verified after runtime reversion.
`evaluation/generalization_summary.json`, `generalization_review.json`, all
`generalization_*results.json`, protocol, datasets and source responses preserve
evidence. Earlier all-404 development trial (harness alias mismatch) was fixed
before freeze and excluded. Candidate was not promoted after coverage failed.

Limits: internally separate authorship, not external validation; final 24 positives
represent 12 country-paired patterns; 8 repeated strings vary real wire controls.
Final negatives stop at eligibility, so regression tests supply downstream control
evidence. Consumed final data cannot be reused as a new independent holdout.
Finite tests cannot prove universal safety. MacBook/Metal remains unverified.

Next priorities: independent-load/repeated benchmarking, target macOS validation,
and final release-readiness checks. General unseen equivalence is outside the
supported release scope; future investigation needs a new independent final set.
No new services, providers, training, infrastructure or performance tuning.
