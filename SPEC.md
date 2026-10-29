# LLM gateway specification

## Status and architecture

Milestones 1–5 provide an async gateway, verified local Gemma inference, client
request quotas, bounded inference admission, exact caching and optional reviewed-
domain semantic reuse. Operational metrics and comprehensive benchmarks remain.
Python 3.12+, FastAPI/Pydantic v2, Uvicorn, and one shared
HTTPX `AsyncClient` per application lifespan. One worker; no Redis or database.

Flow: bearer authentication → client request quota → body-size boundary → strict
request validation → fixed public-model lookup → eligible exact-cache lookup →
optional reviewed-family semantic lookup → hit response, or immediate inference
admission → HTTP backend adapter → validated
normalized response → release inference slot → successful ASGI send → cache commit.
The deterministic mock is a separate local HTTP service, so demonstrations cross
a real network boundary. The adapter targets the documented OpenAI-compatible
non-streaming subset used by Ollama. There is no automatic model fallback.

## Supported contract

- `GET /healthz`: public liveness only, not model readiness.
- `GET /v1/models`: authenticated public aliases, no backend addresses.
- `POST /v1/chat/completions`: authenticated JSON with `model`, `messages`,
  optional `temperature`, `top_p`, `max_tokens`, `seed`, and `stream: false`.
  String content only; roles `system`, `user`, `assistant`; final message must
  be `user`. No tools, images, streaming, response formats, or full OpenAI API
  compatibility. Unknown fields and coercions are rejected.
- `X-Gateway-Cache: exact` explicitly requests exact reuse, subject to the
  operator's policy. `semantic` opts into exact-first and guarded semantic reuse.
  Responses expose `X-Gateway-Cache: HIT|SEMANTIC_HIT|MISS|BYPASS` after
  reaching the cache decision; earlier auth/quota/schema errors omit it. MISS
  identifies a lookup, not a guarantee of inference or insertion. All HTTP
  responses carry `Cache-Control: no-store` for downstream HTTP caches.
- Up to 32 messages, 16,000 characters per message, 32,000 total content
  characters, and 64 KiB body. Generation cap: 2,048 tokens; default 128.
- Normalize to one assistant choice, public model alias, completion ID/time,
  finish reason, and optional upstream-reported token usage. Never invent token
  counts. Private upstream fields and errors are not forwarded.
- Errors use `{"error":{"code":...,"message":...}}`; 401 unauthorized,
  400 malformed body encoding/nesting, 413 body too large, 415 media type,
  422 invalid schema/JSON syntax, 404 unknown model, 429 client quota exhausted,
  502 invalid backend response, 503 unavailable/overloaded backend, 504 timeout.
  Upstream 429 becomes 503 because it is backend capacity, not a client quota.

## Configuration and security

TOML holds fixed backend base URLs and model mappings. Only HTTP(S) URLs without
credentials, query, or fragment; no client-supplied destinations. Redirects and
ambient HTTPX proxy/credential discovery are disabled for local inference.
Operator configuration is trusted; keep destinations on loopback by default.
Secrets come from `GATEWAY_CLIENT_KEYS_JSON`, mapping stable client identifiers
to distinct strong bearer keys. Startup fails without valid credentials. Optional
backend API-key environment-variable names may be configured; keys stay outside
TOML. Documentation describes single-user local development, not Internet exposure.

Authentication precedes JSON/schema validation for protected endpoints. Validation
responses omit submitted input. Do not log prompts, credentials, or backend
response bodies. API docs are disabled by default; use the documented contract.
TLS termination, untrusted configuration, and multi-host identity are out of scope.

## Networking and reliability

Reuse a bounded HTTP connection pool. Configurable upstream deadline,
including pool wait, network I/O, response reading and validation; short
connect/pool timeouts; bounded response size. Disable
automatic retries: a timed-out generation might already have run. Propagate task
cancellation and close responses/connections at shutdown. Admission has zero
waiting slots: reject with 503 before starting a backend task when full. HTTPX
pool waiting remains possible within the bounded admitted set if the operator
sets `max_connections` lower than `max_in_flight`.

On task cancellation or ASGI `http.disconnect` after upload, cancel the backend
operation and await cleanup before releasing its slot. The disconnected path
returns an empty internal 499; the departed peer cannot receive it. Response
cleanup is allowed to finish beyond the upstream budget: disarm any outstanding
deadline cancellation before closing, and check monotonic time afterward so
late success becomes 504. A cancelled read's already-expired timer cannot cancel
cleanup again. Identity-only response streams are read directly to keep HTTPX
iterator auto-close inside the explicitly managed cleanup boundary.

The deadline is cooperative, not a hard client wall-clock bound. It excludes
authentication, request upload/parsing, route lookup, response serialization and
delivery. Cleanup and event-loop scheduling can extend elapsed time. Closing an
HTTP call cannot guarantee a remote server stops generation; admission bounds
gateway operations, not abandoned compute on an uncooperative backend. A hung
transport cleanup can hold its slot beyond the budget; no forced resource kill.
Ingress slow-client deadlines are deferred; loopback-only deployment is supported.

`Server-Timing` provides request-local durations in milliseconds: `gateway` from
ASGI middleware entry until response-start, and `adapter` around `complete()`
including payload preparation, pool/I/O, validation and cleanup. Adapter timing
is absent for local rejections. Gateway timing excludes subsequent ASGI sends,
response-body transmission and client-side work. Neither duration is pure gateway
CPU time or isolated model inference; differences include scheduling/boundaries.

## Backends and embeddings

Ollama 0.40.11 / `gemma3:1b` remains the verified model backend, alias `gemma`,
endpoint `http://127.0.0.1:11434/v1`; CPU, one model, context 2048. macOS is untested.
No Qwen models or implicit fallback.

Compared quantized ONNX conversions of all-MiniLM-L6-v2 and paraphrase-MiniLM-L3-v2
(Apache-2.0, 384 dimensions). Selected L3 using calibration recall and measured
CPU latency/process memory. Source and tokenizer pins, mean pooling, normalization,
package versions and pipeline policy identify vectors. L3 rejects more than 128
tokens including special tokens; L6 rejects more than 256. No truncation, implicit
download, GPU provider or vector database. Optional dependencies are NumPy,
ONNX Runtime and Tokenizers. See README and tracked evaluation JSON for commands,
hardware, revisions, raw scores and results.

## Exact cache (implemented in milestone 4)

Process-local `OrderedDict` stores immutable normalized JSON response bytes.
Default disabled, empty approval list unless explicitly configured. Eligibility
requires **all** of: enabled operator policy, one literal `X-Gateway-Cache: exact`
header, exactly one user message equal to an operator-approved public/static
prompt, `temperature=0`, `top_p=1`, explicit seed, and configured model **and**
backend revision pins. System instructions, conversation history, unapproved
text, sampling, missing pins/seed, duplicate opt-in headers and cookies bypass.
Streaming remains unsupported/422. Private, sensitive, stateful and freshness-
sensitive prompts must never be approved. This is an operator allowlist, not an
automatic content/privacy classifier. No whitespace or semantic equivalence.

Temperature zero and a seed do not guarantee provider determinism. Opt-in
explicitly accepts replay of the first complete successful answer for the TTL;
sampling requests bypass even with a seed. Never cache errors, `length`-truncated
or blank answers. Backend `Cache-Control` (any directive), `Expires`, `Vary`,
`Set-Cookie` or outgoing Cookie disallow insertion rather than attempting to model
provider freshness/session semantics. Client no-store/no-cache and Pragma:no-cache
bypass lookup and storage. HTTPX rejects cookie persistence, preventing hidden
shared backend session state across clients.
Session-cookie-dependent backends are outside this stateless adapter contract.

Keys are per-process-secret HMAC-SHA256 digests over canonical sorted compact JSON:
full validated request with all defaults/ordered messages, stable authenticated
client ID, all gateway configuration (including resolved backend URL, physical
model, revisions, settings, cache namespace/policy), code policy version and hash
of the actual forwarded backend credential. Canonical plaintext exists transiently
for hashing; it is neither retained as a key nor logged. Cache values remain
plaintext responses in RAM. Credentials/prompts are never diagnostics fields.

Model `revision` identifies the approved immutable artifact; backend `revision`
identifies server/template/context/default changes. These are **operator-managed
assertions**, not live attestation or automatic model-change discovery. Before
replacing a model/server/template/context, disable caching, change revisions and
restart to clear state. Configuration is a startup contract with no hot reload;
a final fingerprint recheck also rejects writes if configuration changed during
the request. The real Ollama probe checks live version, digest and context for
that controlled run only. Silent replacement behind unchanged pins is unsafe.

`[exact_cache]`: enabled false, TTL 60 seconds (0.01–3600), max_entries 128
(1–4096), max_bytes 8,388,608 (1024–67,108,864), namespace `public-static-v1`,
approved_prompts (at most 32 strings, 1–4000 characters each). TTL starts at
successful insertion and does not slide on hits. LRU evicts least-recently-read
entries until both limits fit; oversized values are skipped without evicting
existing entries. Byte accounting is **digest bytes + serialized response bytes + attached semantic metadata**,
not RSS or exact Python allocator overhead; entry count separately bounds metadata.
Expiration is lazy on get/put/snapshot. Expired data is never returned but can
remain in RAM while idle until another cache operation or restart.

Cache lookup follows auth, request quota, validation and routing, **before**
inference admission. Hits still spend quota but hold no inference slot and perform
no adapter call. On misses, existing admission/cancellation/timeout protection
is unchanged. No coalescing or waiter registry: simultaneous cold misses may
duplicate inference up to global capacity, with excess rejected 503. The first
fresh cache writer wins; later completions do not replace it or extend its TTL.
All cache mutations have no await on the single owning event loop.

Candidate responses are committed only after the final successful ASGI body send,
HTTP 200, no observed disconnect/cancellation/send failure, and an unchanged
fingerprint. ASGI acceptance is not proof of client receipt/ACK; unobserved network
disconnects remain possible. HIT replays the original id/created/usage bytes;
those usage counts describe the original generation, not new provider work.

Internal counters: hits, misses, bypasses, stores, evictions, expirations and
upstream_calls, plus occupancy. `upstream_calls` counts admitted **adapter
attempts**, including connection/pool failures, not confirmed inference. Verification
also counts fixture requests or dedicated Ollama access-log POSTs independently.
No public stats endpoint or comprehensive telemetry yet. Restart clears responses,
HMAC secret and counters. No persistence, cross-instance sharing or cache fairness.

## Reviewed-domain semantic cache (milestone 5)

Disabled by default and requires the exact store to be enabled. One literal
`X-Gateway-Cache: semantic` requests exact lookup first and semantic lookup second,
only after authentication, request quota, validation and fixed routing. Both hit
types bypass inference admission; misses retain existing zero-queue admission,
cleanup, deadlines and no-retry behavior. Exact-only opt-in never searches vectors.

Additional eligibility: exact literal membership in an operator-reviewed family
of public/static questions with identical answer requirements. Each family has a
unique ID, domain, answer contract and 2–32 distinct approved phrases; at most 16
families; no duplicate/ambiguous phrase membership. Current evaluated domains:
France/Japan capital city-only answers and Python 3 list/tuple comparison with
exactly two short bullets about mutability/literal syntax. Unknown wording,
private/personal/sensitive/contextual/dynamic/freshness/tool workloads bypass.
All original single-user-message, temperature=0/top_p=1/seed, pin, cookie, request/
provider cache directive, successful complete stop, abort and send protections apply.

Cosine similarity retrieves candidates; it never authorizes answer equivalence.
The operator-reviewed family is that assertion. `answer_contract` is metadata,
not an answer validator. The semantic HMAC partition replaces only reviewed
wording with family/domain/contract identity while retaining all other request
fields, full configuration, tenant, model/deployment/credential revisions and
embedding pipeline fingerprint. Only that partition is searched, and freshness/
compatibility are rechecked after embedding and before commit. No general
unseen-paraphrase classifier or automatic privacy assurance is claimed.

Vectors attach directly to response entries, with shared non-sliding insertion
TTL, LRU eviction and byte budget; no independent stale index. Each tag costs a
32-byte partition plus 1,536-byte float32 vector. `semantic_cache.max_entries`
(1–128, default 128) separately bounds tagged entries: remove the oldest tagged
response when full. Response TTL/eviction removes its vector atomically. Exact
entries remain subject to existing total budgets. Oversized combined entries skip
storage without evicting. Hits never create aliases, overwrite content or extend
TTL; simultaneous cold misses are not coalesced.

A dedicated one-job ThreadPoolExecutor runs tokenization, encoding and NumPy
cosine search off the event loop. ORT explicitly uses CPUExecutionProvider,
sequential execution, one intra/inter-op thread and no spinning. Zero embedding
waiters; busy requests fall through to admitted inference. A configurable separate
lookup budget defaults to 0.25 s (0.01–5), threshold defaults to 0.75 (0.65–1). Invalid/
overlength/native errors also fall through without a semantic tag; eligible exact
responses can still be stored. Exact-only entries are not retroactively embedded
on exact hits. Caller cancellation/lookup timeout stops waiting,
not a running native call: worker capacity stays occupied until actual completion.
A monotonic post-check rejects late results if the loop was starved. The budget
is cooperative, not a hard client wall-clock bound. Shutdown waits for native
cleanup; there is no native hard-kill or CPU preemption.
Disconnect watching covers both embedding and backend I/O.

Evaluation: the SHA-pinned 54-case dataset was frozen/committed before scoring.
Only calibration selects a threshold from the fixed grid; held-out cases are not
retuned. At 0.75, selected L3 yields 9/9 held-out positive retrievals, 0 misses,
0/18 negative acceptances; across both splits 0/36 negatives. Similarity alone
at 0.75 accepts 11/18 held-out negatives, so open-domain reuse is unsafe and disabled.
Zero guarded false hits measures finite reviewed-policy enforcement, not general
semantic correctness. Approval/labels are a curated engineering fixture, not an
external study. Five negatives (three held-out) use external scope/response metadata;
these are component-policy cases. Separate synthetic-vector ASGI boundary tests map
them into explicit system context and verify 18 valid reuses / 36 bypasses. The
bare approved user wording alone remains eligible. Keep serving disabled if
approval/evaluation fails; changing the
model, tokenizer, domain, generation or policy requires new approval/evaluation.
See `evaluation/semantic_cases.json`, `evaluation/l3_results.json`,
`evaluation/l6_results.json`, `evaluation/gemma_results.json` and README.

`HIT` is exact; `SEMANTIC_HIT` is reviewed semantic; `MISS` and `BYPASS` retain
prior meanings. Replay returns original id/created/usage bytes. Stage counters
separate exact hits/misses, semantic hits/misses/busy/failures and admitted adapter
attempts. An exact miss followed by a semantic hit is one request with two stages;
response dispositions are mutually exclusive. `Server-Timing: semantic` measures
worker submission/encoding/retrieval, separate from adapter elapsed. No public
metrics endpoint, persistence, thread safety, cross-worker/distributed consistency
or model-generation throughput claim.

## Traffic controls (implemented in milestone 3)

Each configured authenticated client ID gets a token bucket, initially full:
`tokens = min(burst, tokens + elapsed_monotonic_seconds * requests_per_second)`.
Consume one token per authenticated `POST /v1/chat/completions` attempt before
body parsing. Failures, malformed input, unknown models and global overload spend
quota; denied attempts do not further debit it. Health/model listing and failed
authentication do not spend completion quota. This is a request quota, not a
token-usage or billing quota. State is bounded to at most 128 configured clients.

`[rate_limit]` sets a default burst/refill; `[client_rate_limits.CLIENT_ID]`
overrides it. Reject unknown override IDs at startup. Exhaustion returns 429
`rate_limit_exceeded` with integer delta-seconds
`Retry-After = ceil((1 - tokens) / requests_per_second)`, at least 1. This estimates
when the next token is available at that decision; competing requests can consume
it. No reservation or promise of success after that delay.

`max_in_flight` limits all admitted adapter operations across clients/models.
Capacity entry, exit, quota refill and debit have no awaits and are atomic on the
single owning event loop. No asyncio waiting queue or slot reservation. Full
capacity returns 503 `gateway_overloaded` without `Retry-After`: completion time
is unknown. Unknown models return 404 without acquiring slots. Slots remain held
through cleanup on success, backend failure, timeout and cancellation.

Defaults: burst 60, refill 1 request/second, global capacity 8. Ollama example:
burst 10, refill 0.2/second, global capacity 1. Overrides require positive bounded
values; minimum refill 0.000001/second, maximum burst/refill 10,000, capacity 1–64.
There is no per-client concurrency reservation or fairness policy: a client may
occupy all global slots while within quota. Limits apply after authentication and,
for inference slots, after full body/schema validation; they do not bound all
incoming sockets, uploads, server backlog or pre-admission parsing tasks.

All state is in-memory and process-local, reset on restart, neither thread-safe
nor shared between event loops/instances. Run **one Uvicorn worker**, no replicas
or rolling multi-instance enforcement. Do not claim distributed consistency.

## Planned operational metrics

Add protected metrics for requests/outcomes, cache disposition, quota rejection,
in-flight counts, upstream errors, latency histograms, and reported token usage.
Measure gateway overhead separately from inference and queue/embedding time.
Structured logs contain operational identifiers/timing, never content.

## Measurable completion gates

1. Foundation: authenticated validated HTTP round trip, fixed routing, safe errors,
   timeout/connection/malformed-response tests, cancellation and lifecycle cleanup.
2. Actual inference: documented local Gemma request with recorded model/backend
   versions and hardware, distinct from mock evidence.
3. Exact cache: TTL/capacity/isolation/settings/cancellation tests and counted
   reduction of upstream calls. Coalescing is deliberately deferred; concurrent
   misses must still respect admission and first-writer/TTL invariants.
4. Semantic: curated equivalent and adversarial non-equivalent requests; report
   threshold, precision, recall, false-hit count/rate and sample size. Require zero
   false hits on the held-out curated set before opt-in serving. This is finite
   evidence, not a guarantee; document abstention and remaining risks.
5. Traffic/reliability: concurrent quota tests cannot exceed the configured burst;
   in-flight/queue limits hold under failure, saturation, and cancellation.
6. Evidence: reproducible scenarios A–F (uncached, exact, semantic, concurrent,
   quotas, failures) report RPS, median/p95, errors, cache hit rates, false hits,
   call reduction, and raw experimental conditions. Repeat a separate real-model
   workload; never label cached/mock throughput as model-generation throughput.

## Essential documentation checked

Checked 2026-10-27; selected design choices are not performance guarantees.

- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/) and
  [async handlers](https://fastapi.tiangolo.com/async/).
- [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/).
- [HTTPX async pooling](https://www.python-httpx.org/async/),
  [phase timeouts](https://www.python-httpx.org/advanced/timeouts/), and
  [transport retries/lifespan testing](https://www.python-httpx.org/advanced/transports/).
- [Ollama OpenAI subset](https://docs.ollama.com/api/openai-compatibility) and
  [Gemma 3 1B](https://ollama.com/library/gemma3:1b) (806 MB artifact; runtime
  memory is larger; Gemma Terms of Use apply).
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- [Python asyncio timeout semantics](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.timeout)
  and [HTTP Retry-After](https://www.rfc-editor.org/rfc/rfc9110.html#name-retry-after).

Semantic dependency/model checks (2026-10-29):

- [L3 model card](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2)
  and [L6 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- [ONNX mean pooling requirements](https://sbert.net/docs/sentence_transformer/usage/efficiency.html#onnx).
- [ORT thread control](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
  and [Tokenizers stable API](https://huggingface.co/docs/tokenizers/v0.23.1/en/api/tokenizer).
- [Python running-future cancellation](https://docs.python.org/3.12/library/concurrent.futures.html).
