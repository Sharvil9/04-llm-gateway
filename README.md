# LLM gateway

An independent Python infrastructure project for a 24 GB MacBook: a centralized
LLM HTTP API with conservative caching and process-local traffic controls.
**Current status: milestone 7 evaluated independent unseen-wording reuse and
did not establish useful generalization.** Exact/reviewed-phrase caching, bounded
traffic controls, protected metrics and verified Gemma inference remain intact.
The experimental eligibility gate was not promoted to the serving runtime.

The foundation uses FastAPI, strict Pydantic schemas, a fixed model-routing table,
and one pooled HTTPX async client. A separate deterministic HTTP service validates
the forwarding path without downloading a model. No database or paid service.

## Install and verify

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run from this repository:

```sh
cd /workspace/04-llm-gateway   # or your local checkout
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --locked
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m devtools.smoke
.venv/bin/python -m devtools.traffic_probe --output artifacts/traffic-probe.json
.venv/bin/python -m devtools.deadline_probe --output artifacts/deadline-probe.json
.venv/bin/python -m devtools.cache_probe --output artifacts/cache-probe.json
```

The smoke command starts and cleans up two loopback HTTP servers, generates an
ephemeral credential, and checks unauthorized, invalid, unknown-model, model-list,
and successful completion responses. It runs **no real inference**. Its temporary
non-secret configuration stays inside ignored `artifacts/` and is removed on exit.

## Run manually

Terminal 1, from the repository:

```sh
.venv/bin/python -m uvicorn devtools.mock_backend:app \
  --host 127.0.0.1 --port 9001 --no-access-log
```

Terminal 2, from the repository (keep this shell for the next example):

```sh
export GATEWAY_DEMO_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export GATEWAY_CLIENT_KEYS_JSON="{\"demo-client\":\"$GATEWAY_DEMO_KEY\"}"
export GATEWAY_CONFIG=config/mock.toml
.venv/bin/python -m uvicorn llm_gateway.app:create_app --factory \
  --host 127.0.0.1 --port 8000 --workers 1 --no-access-log &
GATEWAY_DEMO_PID=$!
```

Wait for the startup log, then send from the same shell:

```sh
curl -sS http://127.0.0.1:8000/healthz
curl -sS http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $GATEWAY_DEMO_KEY"
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_DEMO_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo","messages":[{"role":"user","content":"Hello"}],"temperature":0,"max_tokens":64}'
```

Stop the gateway with `kill "$GATEWAY_DEMO_PID"` and the mock with Ctrl-C.

Expected assistant text: `Deterministic mock response.` This fixture returns no
token usage rather than pretending to tokenize. API responses expose the public
alias `demo`; backend credentials, addresses, and extra fields are not forwarded.

## Configuration and API

`GATEWAY_CONFIG` selects TOML (default `config/mock.toml`). Configure `[backends.NAME]`
with a fixed HTTP(S) `base_url` ending in `/v1`, and `[models.ALIAS]` with `backend`
and physical `model`. Optional `api_key_env` names a backend-secret environment
variable. Backend destinations are operator-controlled; requests cannot change
them. Redirects, implicit environment proxies/netrc, and automatic retries are
disabled. Generation failures are never retried against another model.

`GATEWAY_CLIENT_KEYS_JSON` is required: client IDs map to distinct 32–256 character
URL-safe keys. Use a generated key per client; never commit real credentials.
`.env.example` is a reference, not an automatically loaded configuration file.
Authentication is required for `/v1/*`; `/healthz` is public liveness only.

Supported chat fields: `model`, string-only `messages` with system/user/assistant
roles, `temperature` (default 0), `top_p` (1), `max_tokens` (128, cap 2,048),
optional nonnegative `seed`, and `stream: false`. Final message must be user.
Unknown fields and coercions are rejected. Up to 32 messages, 16,000 characters
per message, 32,000 content characters total, and 64 KiB uncompressed JSON body.
No streaming, tools, multimodal content, or full OpenAI API compatibility.

Errors have `error.code` and `error.message`, without submitted inputs. 401 means
authentication, 400/422 malformed or invalid input, 413 oversized body, 415 media
type, 404 unknown model, 502 invalid/rejected backend response, 503 unavailable or
overloaded backend, and 504 upstream deadline. Client request quotas return 429;
upstream 429 still maps to 503. Model identity must match the
configured physical name; canonical-name variants require explicit configuration.

TOML controls total upstream deadline, connection-pool capacity, and maximum
backend response size. HTTPX phase timeouts alone are not total deadlines; an
outer async deadline includes pool wait and response reading. Response token
counts, when present, are provider-reported and checked for consistent totals.

## Request quotas and inference admission

One worker owns all in-memory state. Token buckets are keyed by configured
authenticated client IDs, not IP addresses or arbitrary request fields. Each
authenticated completion attempt costs one token, including invalid requests,
backend failures and admission rejections. No refund; quota-denied attempts do
not spend another token. Failed authentication, model listing and liveness are
exempt. These are request counts, not generated-token limits.

```toml
max_in_flight = 8             # root TOML setting; no admission waiting queue

[rate_limit]
burst = 60                   # initial tokens and maximum stored tokens
requests_per_second = 1.0     # continuous refill using a monotonic clock

[client_rate_limits.demo-client]  # optional override for a configured client ID
burst = 5
requests_per_second = 0.5
```

Unknown override IDs fail startup. Exhaustion returns **429**
`rate_limit_exceeded` with `Retry-After` rounded up to whole seconds until one
token refills. That guidance describes the current deficit; other callers using
the same client identity may consume the next token first. The critical
check/update contains no await, so concurrent tasks on this event loop cannot
overspend the bucket.

Full inference capacity returns **503** `gateway_overloaded` immediately;
no task waits for an admission slot. There is no `Retry-After` because completion
time is unknown. Slots are released after HTTP cleanup on success, failure,
timeout or cancellation. An ASGI disconnect cancels the upstream operation;
an empty internal 499 is used only after the peer has departed. The Ollama example
uses **capacity 1, burst 10, refill 0.2/second** to suit the CPU runner.

Admission bounds gateway adapter operations, including HTTPX pool wait. Setting
`max_connections` below `max_in_flight` permits bounded waiting inside that
admitted set. It does not bound uploads, all incoming sockets or server backlog.
There is no per-client concurrency reservation/fairness: a client within quota
can occupy all global slots. Restart resets buckets; multiple workers/replicas
multiply quotas and capacity. This implementation has **no distributed consistency**
and must run with `--workers 1`; traffic state is not thread-safe.

Reproducible loopback HTTP evidence (`traffic_probe`, deterministic fixture):
12 simultaneous requests per client produced **3 accepted / 9 quota-rejected**
for client A and **2 / 10** for client B; exactly **5 upstream calls**. With two
held inference slots, **20 excess requests returned 503**, upstream calls and
peak active count stayed at **2**, and a request succeeded after release.
These are enforcement checks, not model or cached-response throughput numbers.

## Safe exact response caching

Disabled by default. Enable the existing `[exact_cache]` table only for reviewed
**public, static** prompts. Both backend and model `revision` pins are required
(examples are provided); restart and change pins before changing model weights,
server/template/context/defaults. Pins are operator assertions, not live change
detection. Never approve sensitive, personal, stateful or freshness-sensitive text.

```toml
[exact_cache]
enabled = true
ttl_seconds = 60.0
max_entries = 128
max_bytes = 8388608
namespace = "public-static-v1"
approved_prompts = ["What is the capital of France? Answer with only the city name."]
```

The request must also have one literal `X-Gateway-Cache: exact` header, one user
message equal to an approved prompt, `temperature: 0`, `top_p: 1`, and an explicit
seed. Missing pins, system/conversation messages, unapproved text, sampling,
cookies and client no-store/no-cache bypass; streaming remains 422. Approval is
human policy, not a content classifier. A seed/zero temperature does not promise
provider determinism: opt-in accepts replay of the first complete answer for TTL.

Repeat the following against an enabled manual mock gateway (use `gemma` instead
of `demo` for the configured Ollama gateway):

```sh
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_DEMO_KEY" -H 'X-Gateway-Cache: exact' \
  -H 'Content-Type: application/json' -D - \
  -d '{"model":"demo","messages":[{"role":"user","content":"What is the capital of France? Answer with only the city name."}],"temperature":0,"top_p":1,"seed":42,"max_tokens":32}'
```

Responses report `X-Gateway-Cache: MISS`, then `HIT`, or `BYPASS` when ineligible.
Only complete, nonblank, stop-finished responses can fill the cache; errors,
truncation, cancellation and failed ASGI sends do not. Provider cache/freshness
or session headers conservatively prohibit insertion. Cookie persistence is
disabled. Outgoing `Cache-Control: no-store` governs downstream HTTP caches;
explicit application reuse is separate. HIT preserves original response bytes,
including id, creation time and provider usage; it represents **no new inference**.

HMAC fingerprints include client identity, full validated input/defaults, resolved
model/backend revisions, generation settings, full gateway config, credential
scope and policy version. Raw prompts are not stored in keys or logs. Response
values remain plaintext in RAM. Fixed insertion TTL and LRU eviction bound the
combined digest/serialized-response bytes and entry count, not exact process RSS. Expiration is
lazy on cache operations; idle expired bytes may remain in memory until then or
restart. See [SPEC.md](SPEC.md) for the precise policy and limits.

Hits follow auth/rate enforcement and bypass inference admission; they still spend
request quota. Concurrent cold misses are deliberately not coalesced: they may
generate duplicate calls within the existing admission cap, or receive 503. First
fresh writer wins without extending TTL. Candidates commit after a successful
final ASGI send and revision recheck, not proof that the client received the bytes.
Observed disconnects and aborted sends never insert. No shared/distributed cache.

The committed cache probe independently counted **3 upstream calls for 6
successful HTTP requests** (3 hits, 2 misses, 1 sampling bypass), with tenant
isolation and byte-identical replay. For genuine inference, start a dedicated
Ollama server using the earlier Linux recipe, then run:

```sh
.venv/bin/python -m devtools.cache_probe \
  --ollama-log artifacts/milestone02/ollama.log \
  --output artifacts/real-cache-probe.json
```

This mode checks the example's live server version, model digest and context,
starts a temporary cache-enabled gateway, and counts dedicated Ollama access-log
POSTs. Other traffic invalidates the call-count experiment. The initial real run
returned **`Paris`**, `MISS/HIT/HIT`, identical bytes and **one actual inference
POST for three gateway requests**—two calls avoided. These are call-reduction
checks, not a model-generation speed or broad performance benchmark. Internal
`app.state.response_cache.snapshot()` exposes counters to tests; `upstream_calls`
means adapter attempts, not confirmed provider completion. No public stats API.

## Evaluation-driven semantic reuse

Optional and **disabled by default**. This implementation accepts only literal
phrases in operator-reviewed equivalence families. Unknown paraphrases bypass;
it does not infer that arbitrary similar questions have interchangeable answers.
The approved domains are city-only France/Japan capital questions and a narrow
Python 3 list/tuple comparison with identical two-bullet requirements. Never
approve personal, sensitive, conversational, changing or tool-dependent content.

The frozen [dataset](evaluation/semantic_cases.json) contains **18 paraphrases and
36 hard negatives**, split equally into calibration and held-out sets. It was
committed before scoring (`ac41d4a`, SHA256
`80e0ba7b0acaf85afaca2e9bd8fa468c426fe67d50acdfa07541bb98d1a04885`).
Negatives cover negation, countries/entities, quantities, Python versions, output
format/case, additional instructions, system context, private and time-dependent
information. Metadata-only boundary cases represent incompatible request context;
they are not extra supported request fields. **Five of the 36 negatives (three
held-out) use this metadata**: the component evaluator rejects its mismatches.
A separate 54-case ASGI/HTTP-boundary test with synthetic vectors maps those five
cases into explicit system messages and verifies 18 positive reuses / 36 bypasses.
Sending the bare approved user wording alone remains eligible. Labels/membership are a curated
engineering fixture, not an independent human study.

Measured sequentially on the Linux/8 GiB/two-CPU-quota environment below, with
one ORT intra-op thread, batch size one, quantized ONNX and 384-dimensional vectors:

| Candidate | Graph size | Peak process RSS | Encode median / p95 (54 queries) | Guarded calibration recall |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 21.91 MiB | 117.94 MiB | 2.732 / 5.153 ms | 9/9 |
| paraphrase-MiniLM-L3-v2 (selected) | 16.64 MiB | 111.29 MiB | 1.425 / 2.253 ms | 9/9 |

RSS includes Python/libraries/runtime, not just weights; these are small sequential
component measurements, not gateway throughput. L3 provides equal guarded
calibration retrieval at lower observed memory/latency. Both are Apache-2.0
Sentence Transformers models, using pinned Xenova ONNX conversions. Graphs,
tokenizers and manifests are verified against source pins; no implicit downloads.
Full scores, candidate revisions, package versions, memory and calibration curves:
[L3 results](evaluation/l3_results.json), [L6 results](evaluation/l6_results.json).

The threshold grid was fixed at 0.65/0.70/0.75/0.80/0.85/0.90/0.95/0.98/1.0.
Calibration alone selects the highest threshold tied for maximum recall with zero
guarded false hits: **0.75**, unchanged for held-out evaluation. L3 then retrieves
**9/9 held-out positives, 0 misses, 0/18 accepted negatives, precision 9/9**.
Across both splits, guarded false hits are **0/36 negatives**. Similarity alone
at 0.75 accepts **11/18 held-out negatives** (9 correct + 11 false reuses,
45% precision); it is unsafe. The finite membership/family/answer-contract guard
provides the safety boundary. Zero on this finite fixture is no universal
correctness guarantee and does not establish the model's answer quality. Keep
reuse disabled when approval or evaluation is unsatisfactory; never tune on
held-out failures to improve this reported result.

Reproduce without paid services (run model evaluations sequentially):

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
.venv/bin/python -m devtools.semantic_probe --output artifacts/semantic-probe.json
# Start the dedicated Ollama server using the Linux/macOS recipe below first:
.venv/bin/python -m devtools.semantic_probe \
  --ollama-log artifacts/milestone02/ollama.log --output artifacts/real-semantic-probe.json
```

The extra installs NumPy, ONNX Runtime and Tokenizers; the ordinary gateway needs
none of them. Core-only tests skip the optional NumPy integration cases. No
model weights are committed. macOS/Metal is not verified here.
The probe uses the frozen families and checks the evaluation/model fingerprint;
its temporary configuration enables reuse, then cleans up its gateway. Dedicated
Ollama logs independently count actual chat POSTs, so unrelated traffic invalidates
that experiment. It validates actual reused city answers and byte compatibility.
The [recorded Gemma run](evaluation/gemma_results.json) returned **Paris/Tokyo/Berlin**:
8 successful requests, **1 exact hit, 2 semantic hits, 3 misses, 2 bypasses**,
**5 actual Ollama chat POSTs** (3 calls avoided). Tenant changes and sampling
reached inference; semantic replays retained the source response bytes. These
functional samples do not establish model-generation throughput. Semantic-hit
lookup samples were **4.271 / 13.504 ms**, separately from their source adapter
operations **4585.997 / 1380.282 ms**; loading, scheduling and scope differ.

For manual use, enable the existing exact-cache store and retain model/backend
pins. Append a reviewed semantic policy such as:

```toml
[semantic_cache]
enabled = true
model_dir = "artifacts/embeddings/l3"
threshold = 0.75
timeout_seconds = 0.25
max_entries = 128

[[semantic_cache.families]]
id = "france-capital"
domain = "public-static-v1"
answer_contract = "city-name-only-v1"
prompts = [
  "What is the capital of France? Answer with only the city name.",
  "Identify the capital city of France; reply with the city name alone."
]
```

Send one literal `X-Gateway-Cache: semantic` header and the same generation fields
as the exact example (`temperature=0`, `top_p=1`, explicit seed). This opts into
exact lookup **first**, then guarded semantic lookup. Exact-only opt-in never
performs semantic search. `HIT` means exact; `SEMANTIC_HIT` means reviewed semantic
reuse; `MISS` means eligible lookup found no reusable response; `BYPASS` means
ineligible. Both hit types spend quota and bypass inference admission. A miss
still uses bounded inference admission; no model fallback or retry is added.

All exact safeguards apply: one public/static user message, revision/credential/
tenant/generation partitioning, no cookies/cache-bypass directives, complete stop
response, no errors/truncation/aborted sends, final configuration recheck. The
semantic partition replaces only reviewed wording with family/answer identity
and includes the embedding pipeline fingerprint. Wrong entities, numbers,
negation or response requirements are separate families or unapproved text.
`answer_contract` is an operator assertion, not an automatic answer validator.

Vectors are attached to response entries: shared insertion TTL, LRU and total
byte budget (another 32-byte partition + 1,536-byte float32 vector per tagged
entry), plus at most 128 tagged entries. Expiry/eviction removes both; no stale
index or paraphrase aliases extend TTL. The first fresh writer still wins. Cache
values/vectors remain plaintext RAM; metadata and Python overhead are not RSS
accounted. Approval configuration contains only public text.

A dedicated one-job worker runs tokenization/encoding/cosine off the event loop;
ORT uses one CPU thread with spinning disabled. No embedding queue. Busy,
failed or >128-token L3 inputs fall through to ordinary admitted inference
without semantic metadata. Exact-only entries are not retroactively embedded on
an exact hit, which can reduce semantic opportunities. The cooperative 250 ms
lookup budget also falls through and rejects late results; it is not a hard
client wall-clock deadline;
cancel/timeout stops waiting, while the native job retains its capacity until
finished. Shutdown waits for native cleanup. Exact storage can still accept an
approved full-text response; no truncation is used to authorize semantic reuse.

`Server-Timing: semantic` measures embedding-worker submission, encoding and
retrieval, separate from `adapter` (inference HTTP operation). Exact hits have
neither. `app.state.response_cache.snapshot()` counts exact hits/lookup misses
and admitted upstream attempts; `app.state.semantic_cache.snapshot()` separately
counts semantic hits/misses, busy skips and failures. These are stage counters:
an exact miss followed by a semantic hit is one request with two lookup stages.
Use the response disposition for mutually exclusive request outcomes. The new
operator-only metrics endpoint reports these stages and completed dispositions.
State resets on restart; one worker only, no distributed claims.

## Operational metrics

`GET /v1/metrics` provides JSON counters, gauges and fixed cumulative latency
histograms. Set a **separate** `GATEWAY_METRICS_KEY` before gateway startup;
without it the endpoint returns 404. Client credentials cannot read global
metrics, and the operator key cannot make inference requests. For example:

```sh
export GATEWAY_METRICS_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# Start/restart the gateway with this environment, then:
curl -sS http://127.0.0.1:8000/v1/metrics \
  -H "Authorization: Bearer $GATEWAY_METRICS_KEY"
```

Measurements cover `POST /v1/chat/completions`, including failed authentication,
validation, quota/overload and backend errors. Health, model listing and metrics
scrapes do not count or spend completion quotas. Started/finished requests,
completed responses, successful 2xx, errors, aborts and observed 429/503/504 are
separate. A response-start 200 followed by a cancelled/failed body send is not
counted as a successful completion. ASGI acceptance is not proof of client receipt.
Gauges distinguish HTTP requests in progress from admitted inference operations.
Fixed status families/outcomes replace per-client/model/path labels; prompts,
credentials, responses and raw latency samples are never retained as metrics.

Exact/semantic hit rates expose numerator and denominator (eligible completed
lookups); cache storage counters are separate stage diagnostics. Upstream attempts
include admitted connection failures and are distinct from independently observed
backend calls. Token counters add only newly reported upstream usage, never
historical cache-hit usage or estimates. Consistent counts outside the telemetry
64-bit range remain in the API response but increment `ignored_usage_reports`
instead of changing request behavior. Scrapes expose global state only to the
operator and do not include bearer keys, backend addresses or tenant identifiers.

Latency scopes, all in milliseconds:

| Measurement | Scope |
| --- | --- |
| `gateway` / histogram `gateway_to_start` | Middleware entry through response-start |
| `adapter` | Upstream request preparation, pool/network/model wait, normalization and cleanup |
| `exact` / `exact_lookup` | Eligibility, canonical fingerprint and exact lookup |
| `semantic` | Worker submission, embedding, cosine retrieval and compatibility checks |
| `residual` / `gateway_residual` | Gateway-to-start minus those sequential stages |
| `request_lifecycle` histogram | Middleware entry through final send/cache commit, before final metric update |
| Benchmark client latency | Request submission through full HTTP response receipt |

Residual includes authentication, parsing, serialization and event-loop scheduling;
it is **not isolated gateway CPU time**. Adapter is not pure token-generation
latency. Residual headers are present after the cache decision; early rejections
retain only gateway timing. Histograms keep fixed bucket counts/sum/max, not raw
samples or fabricated quantiles. Labels/state are process-local and reset on
restart; use one worker. Benchmarks compute empirical percentiles from bounded
raw client samples instead.

Optional top-level TOML `log_requests = true` writes one JSON operational record
per completion attempt to stderr: event/status/outcome/cache/elapsed time only.
Default off; no prompts, keys, model or client identifiers. Logging is synchronous
and excluded from the recorded performance configuration.

## Reproducible benchmarks and semantic audit

The recorded profile is Linux x86_64, Python 3.12.14, **8 GiB / two CPU cores'
quota**, one Uvicorn worker, loopback HTTP/1.1, pooled HTTPX, no request/access logs.
The driver and servers share that CPU budget. The mock adds a deterministic 10 ms
wait and returns a known city. Real tests use pinned Ollama 0.40.11 / Gemma 3 1B
Q4_K_M, context 2048, CPU only. Message/generation settings match direct/gateway
baselines; real model identifiers differ only by public alias vs physical name.
Repeated real prompts may reuse Ollama KV state even when gateway caching is off.
This is a short city-answer workload, not general model-generation performance.

Each point uses a fresh gateway and five sequential warm-up requests (plus one
cache seed where needed), excluded from measurement. Seed 42 fixes mixed request
kinds at 40% exact / 40% semantic / 20% no-store. Benchmark quotas are deliberately
10000 burst/refill per second; inference capacity is 8 mock / 1 real and HTTP pool 64.
Caches retain the existing 60 s TTL, bounded entries/bytes, tenant partitioning,
revision pins, request opt-in and response safety checks. Exact and semantic
workloads are verified warm; source-response hashes check controlled replays.
Concurrent semantic opt-in can fall through on a busy worker and then become
exact hits; actual dispositions are recorded and must accompany its throughput.

The bounded closed-loop driver records attempted, HTTP-completed, verified-correct
successful, rejected and failed requests separately. RPS means successful RPS in
the tables. p50/p95/p99 use nearest rank; p95 is withheld below 100 samples, p99
below 1000. Empirical tails are not confidence guarantees. Source/lock manifests
must remain unchanged throughout each suite. Independent fixture/quiet Ollama
access-log counts must reconcile with gateway upstream attempts. Linux owned-PID
sampling records CPU cores used and RSS on a nominal 100 ms cadence; missing
or incomplete process trees are null, not zero. Bounded GNU `ps --ppid` discovery
handles kernels without proc child files; sampling runs off the driver loop.
RSS excludes unsampled spikes and may count shared pages more than once.

Reproduce sequentially after installing the semantic extra/model as above:

```sh
.venv/bin/python -m devtools.benchmark --requests 1000 --concurrency 1 4 8 16 \
  --warmup 5 --seed 42 --delay-ms 10 --scenarios direct uncached exact semantic mixed \
  --output artifacts/milestone06/mock-benchmark.json
# Use only the Ollama-server portion of the setup below, with KEEP_ALIVE=1m
# and log destination artifacts/milestone06/ollama.log (create its directory).
# Set OLLAMA_PID=$OLLAMA_DEMO_PID. Do not run other inference traffic. Optionally add
# --ollama-pid "$OLLAMA_PID" to sample only that owned server and its children.
.venv/bin/python -m devtools.benchmark --requests 30 --concurrency 1 --warmup 5 \
  --seed 42 --scenarios direct uncached --ollama-log artifacts/milestone06/ollama.log \
  --output artifacts/milestone06/gemma-uncached.json
.venv/bin/python -m devtools.benchmark --requests 1000 --concurrency 1 --warmup 5 \
  --seed 42 --scenarios exact semantic --ollama-log artifacts/milestone06/ollama.log \
  --output artifacts/milestone06/gemma-cache.json
.venv/bin/python -m devtools.benchmark --requests 100 --concurrency 1 --warmup 5 \
  --seed 42 --scenarios mixed --ollama-log artifacts/milestone06/ollama.log \
  --output artifacts/milestone06/gemma-mixed.json
.venv/bin/python -m devtools.traffic_probe --output artifacts/milestone06/traffic-probe.json
.venv/bin/python -m devtools.deadline_probe --output artifacts/milestone06/deadline-probe.json
.venv/bin/python -m devtools.semantic_audit --validate-only
.venv/bin/python -m devtools.semantic_audit --output artifacts/milestone06/semantic-audit.json
```

Observed final trial, **concurrency 1**, all shown requests HTTP 200 with correct
city answers. Latencies are client p50/p95/p99 in milliseconds; — is withheld.
Cached response RPS describes gateway replay, not new model inference.

| Backend / path | N | Successful RPS | p50 / p95 / p99 ms | Actual upstream calls |
| --- | ---: | ---: | ---: | ---: |
| Mock 10 ms — direct baseline | 1000 | 56.67 | 14.47 / 24.58 / 128.26 | 1000 |
| Mock 10 ms — gateway, no cache | 1000 | 54.90 | 17.30 / 23.37 / 27.74 | 1000 |
| Mock 10 ms — warm exact replay | 1000 | 245.97 | 3.60 / 7.44 / 9.96 | 0 |
| Mock 10 ms — warm semantic replay | 1000 | 150.71 | 5.51 / 12.11 / 16.89 | 0 |
| Mock 10 ms — 40/40/20 mixed | 1000 | 124.98 | 5.71 / 20.02 / 24.69 | 200 |
| Gemma — direct baseline | 30 | 1.81 | 539.42 / — / — | 30 |
| Gemma — gateway, no cache | 30 | 1.76 | 569.35 / — / — | 30 |
| Gemma — warm exact replay | 1000 | 283.22 | 2.97 / 6.75 / 9.60 | 0 |
| Gemma — warm semantic replay | 1000 | 112.23 | 7.93 / 16.35 / 21.23 | 0 |
| Gemma — 40/40/20 mixed | 100 | 8.50 | 6.50 / 575.86 / — | 20 |

Mock uncached saturation, 1,000 attempts per point, capacity 8:

| Client concurrency | Attempted RPS | Successful RPS | HTTP 503 | Upstream calls |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 54.90 | 54.90 | 0 | 1000 |
| 4 | 134.55 | 134.55 | 0 | 1000 |
| 8 | 239.18 | 239.18 | 0 | 1000 |
| 16 | 316.81 | 211.32 | 333 | 667 |

The concurrent semantic opt-in scenario at 4 workers had **995 exact hits,
2 semantic hits and 3 misses**, not 1,000 semantic hits. Its workload label must
not be interpreted as semantic-serving throughput. All measured HTTP 200 responses passed
the fixed city-answer check; only intentional saturation produced HTTP errors.
Controlled warm replays were also checked against the source response SHA256.

The matched mock baseline observed 14.47 ms direct versus 17.30 ms gateway median
client latency; median gateway residual was 0.446 ms. At 8 uncached workers, the
gateway used 0.90 CPU core and the load generator 0.65, sharing the two-core quota.
At 16, their CPU use plus the mock approached 1.95 cores and 333 requests were
rejected. The observed CPU time approaches the quota, with noisy tail/scaling behavior.
It cannot isolate a precise causal CPU overhead or prove a speedup.
No gateway performance optimization was made from these measurements.

Real uncached inference used **1.84 model CPU cores**, versus a median gateway
residual of **0.770 ms** inside a roughly 569 ms client request. Semantic lookup
itself took **3.301 ms median** in the real-response cache trial. Gateway sampled
RSS was **44.5 MiB uncached / 44.3 MiB exact / 122.9 MiB semantic**. Ollama plus its
model runner remained about **902 MiB resident**, including during cache hits;
caching avoided inference work, not model residency. Model CPU during warm-hit
windows was below 0.03 core. These are sampled process RSS, not allocator or
whole-machine memory guarantees.

Real mixed traffic made **20 upstream calls for 100 requests** (40 exact + 40
semantic hits, 20 no-store), avoiding 80 calls for this workload. New reported
usage was 460 prompt / 40 completion / 500 total tokens, excluding cache replays.
No monetary or general token-generation-rate claim follows.

The held-backend reliability probe independently verified **19 HTTP 429 across
24 quota attempts**, isolated client bursts 3 and 2, then **20 overload 503**
rejections while two inference slots were held. Recovery succeeded after release.
A separate real HTTP trickle returned 504 in **1009.257 ms** for a 1 s cooperative
upstream deadline; its metrics recorded one timeout/attempt and zero held permits.
Cancellation, disconnect, send failures, private/tenant boundaries and TTL/races
remain covered by regression tests.

Machine-readable [summary](benchmarks/summary.json), [reliability evidence](benchmarks/reliability.json),
and full bounded raw trials: [mock](benchmarks/mock-benchmark.json.gz),
[Gemma uncached](benchmarks/gemma-uncached.json.gz), [Gemma cache](benchmarks/gemma-cache.json.gz),
[Gemma mixed](benchmarks/gemma-mixed.json.gz). JSON Gzip files retain every bounded
request record, metrics snapshots, source/configuration and sampled resource aggregates;
the summary records content hashes and exact commands. Read one without extra tools:

```sh
.venv/bin/python -c 'import gzip,json; d=json.load(gzip.open("benchmarks/gemma-uncached.json.gz")); print(d["results"][1]["latency_all"])'
```

The new semantic audit verifies calibration-only threshold recomputation, but
**all nine original held-out positive literals were already approved**. A
membership-only lookup reproduces the guarded admission result without embeddings.
Only three canonical vectors were stored, so raw cosine retrieval was real;
perfect guarded recall nevertheless does not demonstrate unseen-phrase support.

A second 18-case set was committed before scoring (`95cba86`, SHA256
`b819653ab102a268a1720db80e5f7719a56f9743b7eb305efe46bc68460d4661`).
All wording is absent from the original cases/approval registry. With the unchanged
L3 model and threshold 0.75, raw retrieval finds **4/6 equivalents**, misses 2, and
incorrectly accepts **6/12 negatives**. Production correctly bypasses all 18 new
phrases, including all six equivalents: **served unseen-positive recall 0/6**.
Thresholds, original labels and eligibility were not relaxed or retuned. This
follow-up was authored after M05 results and is not an independent external study;
training-data overlap and prior human inspection cannot be disproved. The finite
reviewed vocabulary remains the operational safety boundary. See the frozen
[audit cases](evaluation/semantic_audit_cases.json) and [results](evaluation/semantic_audit_results.json).

## Independent generalization validation (milestone 7)

**Verdict: useful unseen-paraphrase generalization remains unproven.** The frozen
candidate reused development wording but bypassed every independently authored
positive. `main` retains the verified reviewed-phrase policy; unknown wording
continues to reach inference. No new eligibility option ships in configuration.

| Complete production pipeline | Valid semantic reuses | False negative-case reuses | Semantic-hit precision | Bypasses |
|---|---:|---:|---:|---:|
| Development, reviewed baseline | 0/14 | 0/22 | undefined | 36/36 |
| Development, candidate | 12/14 (85.7%) | 0/22 | 12/12 | 24/36 |
| Independent final, baseline | 0/24 | 0/48 | undefined | 72/72 |
| Independent final, candidate | 0/24 | 0/48 | undefined | 72/72 |

The genuine-Gemma-source replay evaluations produced the same counts. All 12
development reuses satisfied the actual factual/output requirement: exactly
`Paris` or `Tokyo` after trimming surrounding whitespace. Final replay suitability
is **unobserved (0 reuses)**. The raw frozen report's `all_semantic_answers_suitable`
boolean is vacuously true when empty; it is not a measured 100% success rate.
Zero final false hits here reflects abstention, not demonstrated useful safety.

Why the earlier six positives bypassed: literal family membership failed before
embedding, retrieval or compatibility checks ran. Four exceeded the existing
0.75 cosine threshold; two additionally scored 0.58748/0.74360. Raw similarity
also accepted 6/12 earlier negatives, so removing membership was unsafe.

The tested candidate required operator approval of France/Japan and a complete,
bounded ASCII grammar: a national-capital question, explicit sentence boundary,
and city-name-only instruction, with no unconsumed modifiers or context. Only
case and ordinary repeated spaces normalized. Parsed entity, fixed answer
contract and grammar version replaced wording in the existing isolated HMAC
partition. Exact lookup remained first; the pinned L3 encoder and **unchanged
0.75 threshold** supplied an additional retrieval filter. Correct city-only
answers were required before insertion and replay. Authentication, quotas,
tenant/model/revision/settings compatibility, opt-in, context/cookie/directive
restrictions, TTL, bounded storage and worker/admission cancellation stayed intact.
Python remained reviewed-only. This was rule composition, not a language judge.

Final positives used different linguistic constructions, all outside that
grammar. They bypassed before embedding: the result measures eligibility
coverage, not a failure of final candidate retrieval or embeddings. Final
negative cases also failed grammar, so they do not isolate downstream controls;
accepted-syntax security, concurrency, isolation and expiry tests cover those
separately. The candidate was reverted after evaluation rather than widened
using final outcomes. Any future gate needs a fresh independent final set.

The final set was frozen in **`f9686d5`** before gate development/scoring, SHA256
`ad05be1634d5a890e62a74eceedcff4e1c6b2cc5ec218fcff1230077f5684845`.
Candidate code, tests, harness and source manifest were frozen in **`d9955fa`**
before reading final wording. Development: 36 cases; final: 72 cases with 24
positives/48 negatives. There is zero wording overlap between final and development,
prior cases or approved literals. Final positives have 12 linguistic patterns
paired across countries; eight repeated final strings test different wire inputs.
An independently delegated internal author created the final set without seeing
the gate/development set. This is not an external study, random sample, or proof
of training-data novelty. No threshold, grammar or approvals were tuned afterward.

Each case traversed the real authenticated ASGI gateway with actual local ONNX
embeddings; only two canonical source entries were restored before each case.
Evaluation queries could not expand retrieval sources. Cold requests used a
counted no-store fixture; these calls are not real model inference. Separately,
the frozen candidate's genuine loopback HTTP probe made six requests and **four
independently counted Ollama calls**: two source inferences (`Paris`, `Tokyo`),
two suitable unseen development replays, Germany→`Berlin` BYPASS, tenant-B MISS.
Captured genuine source answers were replayed through both full datasets.
Ollama 0.40.11 / pinned Gemma 3 1B Q4_K_M, context 2048, CPU-only Linux 8 GiB /
two-core quota; macOS remains untested. This is correctness evidence, not a new
performance experiment. The restored release's real semantic probe also passed.

Artifacts: [summary](evaluation/generalization_summary.json),
[independent review](evaluation/generalization_review.json),
[protocol](evaluation/generalization_protocol.md),
[final cases](evaluation/generalization_holdout.json),
[candidate freeze](evaluation/generalization_candidate_freeze.json),
`evaluation/generalization_*results.json`, and
[genuine samples](evaluation/generalization_gemma_sources.json).
Frozen candidate: **314 passing tests**; release: **262 passing tests**.

Reproduce the unmodified experiment in an isolated worktree inside this project.
The editable installation points at `main`, so `PYTHONPATH` explicitly selects
the frozen source. Reuse the installed semantic extra and pinned L3 files above:

```sh
# From the project root; use a fresh worktree path.
git worktree add --detach artifacts/m07-candidate d9955fa00f345d55b13c1dbf5d8afe2f451e693f
ln -s ../../.venv artifacts/m07-candidate/.venv
cd artifacts/m07-candidate
PYTHONPATH="$PWD/src" .venv/bin/pytest
PYTHONPATH="$PWD/src" .venv/bin/python -m devtools.generalization_eval \
  --dataset dev --model-dir ../embeddings/l3 --output ../milestone07/reproduced-dev.json
PYTHONPATH="$PWD/src" .venv/bin/python -m devtools.generalization_eval \
  --dataset holdout --model-dir ../embeddings/l3 \
  --gemma-sources ../../evaluation/generalization_gemma_sources.json \
  --output ../milestone07/reproduced-heldout-gemma.json
# Optional fresh inference: first start the dedicated, quiet Ollama using the
# README server recipe, logging to main's artifacts/milestone07/ollama.log.
PYTHONPATH="$PWD/src" .venv/bin/python -m devtools.generalization_live \
  --model-dir ../embeddings/l3 --ollama-log ../milestone07/ollama.log \
  --output ../milestone07/reproduced-sources.json
cd ../..
git worktree remove --force artifacts/m07-candidate
```

The exact final/Gemma replay command above was verified from the frozen worktree.
Using captured sources performs no new model inference. Current release checks:

```sh
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m devtools.semantic_probe \
  --ollama-log artifacts/milestone07/ollama.log \
  --output artifacts/milestone07/release-semantic-probe.json
```

## Timeout scope and investigation

`request_timeout_seconds` bounds the upstream operation: connection/pool wait,
I/O, reading and normalization. Cleanup completes before a slot is released,
even beyond the budget; a monotonic post-check changes late success to 504.
The timer is disarmed before cleanup to avoid a second deadline cancellation
interrupting cleanup already triggered by a disconnect. Identity-only raw
response iteration avoids HTTPX auto-close under the expiring read timer.

`Server-Timing` reports **gateway** time from request-boundary middleware entry to response-start and
**adapter** time around the complete adapter call, including cleanup. The latter
also includes request preparation; neither is isolated model inference or CPU
overhead. Response delivery, client scheduling and network time occur outside
the gateway measurement. Async deadlines are cooperative, not hard client
wall-clock bounds; a hung transport cleanup can retain capacity beyond the budget.

The old **1 s → 1.567 s** sample recorded only external time; the available data
cannot attribute the extra 567 ms to a specific cause. New `deadline_probe`
measurements separate client, gateway and adapter scopes over actual HTTP with
50 ms trickled chunks: the final run measured **1003.323 ms client**, **1001.801 ms
gateway**, and **1001.043 ms adapter**, returning 504. Tests separately demonstrate delayed cleanup and event-loop
blocking. A real defect was reproduced using a deliberately blocking test
transport: with a 5 ms budget, the previous code returned **200 after 31.4 ms**;
the monotonic check now returns **504** after cleanup. This fixes classification,
not the inability to preempt synchronous work. Overlapping cancellation/deadline
tests also verify cleanup finishes while its inference slot remains occupied.

## Real model: verified milestone 2

Verified on 2026-10-27 using **Ollama 0.40.11**, `gemma3:1b`, 999.89M parameters,
GGUF `Q4_K_M`. The unchanged `config/ollama.toml` routes alias `gemma` to that
physical model. This section records milestone 2; milestone 3 added traffic
settings while preserving this model mapping and adapter contract.

Model manifest SHA-256:
`97558784bdc9175075a72655bf360b15da17df6b3b9e5182af9f8c0c8e58ba7c`.
The 806,058,400-byte model-weight layer was independently hashed as
`2f8c3d137f2496dcb4019e17a50f2974b89c1be1a742db3c80cc86f91cafeeef`.
Gemma Terms of Use apply. No alternate inference server or mock supplied these
results; Ollama itself uses its bundled llama.cpp runner for this model.

### Linux setup used

The runner had 8 GiB cgroup memory, two CPU cores' quota, no GPU, and initially
29.8 GiB free disk. Installation used the official archive rather than modifying
system services. Requires `curl`, `tar`, `zstd`, and **bubblewrap 0.12.0** with
working namespaces. From the repository:

```sh
mkdir -p artifacts/ollama-runtime artifacts/ollama-state artifacts/ollama-models \
  artifacts/ollama-tmp artifacts/milestone02
curl --fail --location --silent --show-error --max-time 600 \
  --output artifacts/ollama-linux-amd64-v0.40.11.tar.zst \
  https://github.com/ollama/ollama/releases/download/v0.40.11/ollama-linux-amd64.tar.zst
printf '%s  %s\n' \
  7bedab90dd122661be78798e9f0edf37fccf0cccf2699d2c58ea1a3afdc04b6d \
  artifacts/ollama-linux-amd64-v0.40.11.tar.zst | sha256sum --check
tar --zstd --extract --file artifacts/ollama-linux-amd64-v0.40.11.tar.zst \
  --directory artifacts/ollama-runtime
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$PWD/artifacts/ollama-models"
export OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=2048
export OLLAMA_KEEP_ALIVE=10m OLLAMA_NO_CLOUD=1
bwrap --die-with-parent --ro-bind / / --dev /dev --tmpfs "$HOME" \
  --bind "$PWD/artifacts/ollama-state" "$HOME/.ollama" \
  --bind "$PWD/artifacts/ollama-models" "$PWD/artifacts/ollama-models" \
  --bind "$PWD/artifacts/ollama-tmp" "$PWD/artifacts/ollama-tmp" \
  --chdir "$PWD" --setenv TMPDIR "$PWD/artifacts/ollama-tmp" \
  -- "$PWD/artifacts/ollama-runtime/bin/ollama" serve \
  > artifacts/milestone02/ollama.log 2>&1 &
OLLAMA_DEMO_PID=$!
```

Ollama creates a key in `~/.ollama` even with cloud disabled. This namespace keeps
the host filesystem read-only, provides an ephemeral home filesystem, and binds
only state/models/temp back to ignored repository directories. The `HOME`
environment variable and inherited network proxy/CA settings are unchanged.
`--dev /dev` provides usable sandbox devices, including `/dev/null`.

After `/api/version` responds, pull **one model**, then start the gateway with
the real-model configuration. Skip the mock-server command:

```sh
curl --fail --silent --show-error --noproxy 127.0.0.1 \
  http://127.0.0.1:11434/api/version
curl --fail --silent --show-error --noproxy 127.0.0.1 --max-time 600 \
  http://127.0.0.1:11434/api/pull -H 'Content-Type: application/json' \
  -d '{"model":"gemma3:1b","stream":false}'
export GATEWAY_CONFIG=config/ollama.toml
export GATEWAY_DEMO_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export GATEWAY_CLIENT_KEYS_JSON="{\"demo-client\":\"$GATEWAY_DEMO_KEY\"}"
.venv/bin/python -m uvicorn llm_gateway.app:create_app --factory \
  --host 127.0.0.1 --port 8000 --workers 1 --no-access-log &
GATEWAY_DEMO_PID=$!
```

The pull returned `{"status":"success"}`. Never run the mock backend as a
substitute for this step. The following reproduces the direct request **first**,
then the authenticated gateway request with the same messages and parameters.
Keep `GATEWAY_DEMO_KEY` from the gateway startup shell:

```sh
.venv/bin/python - <<'PY'
import json, os, httpx
payload = {
    "model": "gemma3:1b",
    "messages": [{"role": "user", "content":
                  "What is the capital of France? Answer with only the city name."}],
    "temperature": 0.2, "top_p": 0.8, "max_tokens": 32,
    "seed": 42, "stream": False,
}
with httpx.Client(trust_env=False, timeout=120) as client:
    direct = client.post("http://127.0.0.1:11434/v1/chat/completions", json=payload)
    direct.raise_for_status()
    print("direct", json.dumps(direct.json()))
    gateway = client.post("http://127.0.0.1:8000/v1/chat/completions",
        json=payload | {"model": "gemma"},
        headers={"Authorization": "Bearer " + os.environ["GATEWAY_DEMO_KEY"]})
    gateway.raise_for_status()
    print("gateway", json.dumps(gateway.json()))
PY
curl --silent --show-error --noproxy 127.0.0.1 http://127.0.0.1:11434/api/tags
curl --silent --show-error --noproxy 127.0.0.1 http://127.0.0.1:11434/api/ps
```

Both actual responses contained assistant text **`Paris`**, `finish_reason: stop`,
and provider-reported usage `{prompt_tokens: 23, completion_tokens: 2, total_tokens: 25}`.
The direct response used `model: gemma3:1b`; the gateway normalized it to `gemma`
and removed `system_fingerprint` and nested provider-only token details. Direct
request ID was `chatcmpl-325`; gateway ID was `chatcmpl-613`. Matching outputs
support the integration evidence, not a guarantee of deterministic inference.
Ollama's runner logs confirmed `temp = 0.200` and `top_p = 0.800`; existing tests
verify forwarding of all supported parameters, including `seed`.

### Real-backend checks and limitations

All requests were sequential. These elapsed samples are functional evidence,
not a latency/throughput benchmark:

| Check | Observed result |
| --- | --- |
| Direct / gateway Paris request | HTTP 200 / 200; 0.962 s / 0.555 s |
| `max_tokens: 1` through gateway | HTTP 200; completion tokens 1; finish `length` |
| System message "Answer with only one word."; capital of Japan | HTTP 200; `Tokyo` |
| Missing / invalid bearer key | HTTP 401 / 401 |
| Authenticated model listing | Only public alias `gemma` |
| Physical name `gemma3:1b` as client alias / `stream: true` | HTTP 404 / 422 |
| 50 ms deadline, live Ollama | HTTP 504 in 0.082 s; inference start not established |
| 1 s deadline, warmed Ollama, enumerate 1–100, cap 128 | HTTP 504 in 1.567 s; runner started task, then logged cancellation |
| Stop actual Ollama, request through gateway | HTTP 503 `backend_unavailable`; liveness still 200 |
| Restart same Ollama/model, request through same gateway | HTTP 200; `Paris` (8.894 s including model reload) |

For the two timeout checks, temporary copies of the existing TOML changed only
`request_timeout_seconds` to `0.05` and `1.0`, respectively; separate gateway
processes listened on port 8001. Reproduce the stronger check while Ollama is warm:

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
Path("artifacts/milestone02/timeout-1s.toml").write_text(
    Path("config/ollama.toml").read_text().replace(
        "request_timeout_seconds = 60.0", "request_timeout_seconds = 1.0"))
PY
GATEWAY_CONFIG=artifacts/milestone02/timeout-1s.toml \
  .venv/bin/python -m uvicorn llm_gateway.app:create_app --factory \
  --host 127.0.0.1 --port 8001 --workers 1 --no-access-log &
GATEWAY_TIMEOUT_PID=$!
# Wait for /healthz, then send this authenticated request:
curl --silent --show-error --noproxy 127.0.0.1 --max-time 120 \
  --write-out '\nHTTP %{http_code}; elapsed %{time_total}s\n' \
  http://127.0.0.1:8001/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_DEMO_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gemma","messages":[{"role":"user","content":"List the integers from 1 through 100 separated by spaces."}],"temperature":0.2,"top_p":0.8,"max_tokens":128,"seed":42,"stream":false}'
kill "$GATEWAY_TIMEOUT_PID"
```

On faster hardware this request may finish before the deadline; the 504 result
above applies to this tested CPU workload.

The deadline is cooperative and covers the adapter, not all gateway work; the
observed client wall time exceeded it under CPU-limited conditions. Recovery
requests succeeded after both timeout probes; latency
varied, including a 52.14 s direct cold/reload request. Closing HTTP does not
guarantee prompt cancellation on every backend, even though this active probe
logged `cancel task` and a subsequent direct request succeeded. To reproduce the
outage check, stop the owned Ollama process (`kill "$OLLAMA_DEMO_PID"`), repeat
the normal authenticated request, then restart the same server and retry manually.
There are no inference retries or model fallback in the gateway.

Runtime feasibility was measured: `/api/ps` reported 881,076,469 bytes loaded,
`size_vram: 0`, and `context_length: 2048`; process RSS was 38,220 KiB for Ollama
and 891,444 KiB for its runner. Whole-cgroup peak was **6,067,605,504 bytes
(5.65 GiB)**, including approximately 4.54 GiB of file cache; sampled peak
anonymous memory was 0.964 GiB. Cgroup `oom`, `oom_kill`,
and `max` event counts remained zero. This establishes feasibility only for the
tested short single-model, single-inference workload, not large contexts or
concurrent inference on this runner. Model allocation, RSS, and cgroup totals
measure different things and must not be added together.

One quality probe failed: `Compute 17 + 25. Reply with only the integer.` returned
`32` (HTTP 200, 24 prompt / 4 completion tokens), not 42. Infrastructure success
does not establish answer correctness. Local raw JSON, logs, and one-off drivers
remain in ignored `artifacts/milestone02/`; committed evidence is summarized here
and in `PLAN.md`. All verification servers were stopped afterward.

No installation, registry-access, or OOM blocker remained. macOS/Metal has not
been tested. On macOS 14+, install the [official Ollama app](https://ollama.com/download),
set the same context/concurrency controls when launching `ollama serve`, pull
`gemma3:1b`, and use the same gateway/request commands; native Ollama writes to
`~/.ollama`, while the filesystem-isolation wrapper above is Linux-specific.
References: [release](https://github.com/ollama/ollama/releases/tag/v0.40.11),
[Linux installation](https://docs.ollama.com/linux),
[memory/concurrency controls](https://docs.ollama.com/faq), and
[OpenAI subset](https://docs.ollama.com/api/openai-compatibility).

## Evidence and limitations

Verification environment: Linux x86_64, Python 3.12.14, 8 GiB cgroup memory limit,
two CPU cores' quota (three visible logical CPUs). No Ollama or GPU tooling was
available initially. These are not MacBook results.

Milestones 1–7 evidence is recorded in `PLAN.md`, including **262 release tests**,
lint/format checks, deterministic HTTP enforcement/deadline probes, and real-model
verification. Concurrent forwarding tests are not a load benchmark.
Recorded RPS applies only to the named workload and conditions. Cache/mock RPS
is separate from actual Gemma inference; no general token-rate, cost or hardware
performance guarantee is claimed.

Use loopback and one worker. There is no public metrics endpoint or ingress
slow-client deadline, or shared state across workers. Pool capacity and inference
admission are distinct from request quotas. Task/ASGI-disconnect cancellation
closes active HTTPX responses; the automated disconnect tests inject ASGI events.
HTTP cleanup
cannot guarantee the model server stops inference. Do not expose this foundation
directly to the Internet. Prompts, credentials, and upstream error bodies are not
logged by application code; access logging is disabled in examples.

The benchmark harness covers uncached/exact/semantic/mixed, concurrency, quotas
and failure probes with raw results and separate mock, overhead and real-model
measurements. Generalization remains outside the supported cache policy. Remaining
priorities are independent-load benchmarking, target-MacBook measurements and final
release-readiness verification, with profiling before optimization. See [SPEC.md](SPEC.md) for cache
safety/security boundaries and [PLAN.md](PLAN.md) for the remaining gates.
