# LLM gateway specification

## Status and architecture

Milestones 1–3 provide an async gateway, verified local Gemma inference, client
request quotas, and bounded inference admission. Caching, operational metrics,
and performance benchmarks are still required for the complete project.
Python 3.12+, FastAPI/Pydantic v2, Uvicorn, and one shared
HTTPX `AsyncClient` per application lifespan. One worker; no Redis or database.

Flow: bearer authentication → client request quota → body-size boundary → strict
request validation → fixed public-model lookup → immediate inference admission →
HTTP backend adapter → validated normalized response → release inference slot.
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

## Backends and future embeddings

Ollama 0.40.11 with `gemma3:1b` was verified on the 8 GiB CPU-only runner in
milestone 2. The adapter targets `http://127.0.0.1:11434/v1`; the public alias is
`gemma`. macOS remains untested. No Qwen models or implicit fallback. Do not load
multiple models simultaneously.

For semantic embeddings, evaluate `sentence-transformers/all-MiniLM-L6-v2`
(384 dimensions, CPU) using normalized vectors and bounded NumPy cosine search.
The model truncates beyond 256 word pieces: semantic eligibility must reject
overlength prompts instead of silently losing constraints. No vector database
is justified for the initial small cache. Run embedding work
off the event loop with bounded work; measure its latency before enabling it.

## Planned cache policy (not implemented)

- Exact: opt-in deterministic requests (`temperature=0`), normalized full
  messages and all supported settings; partition by client, backend/model
  identity, system context, and cache-policy version. TTL and capacity bound;
  no persistence initially. Coalesce identical eligible in-flight requests and
  test leader/follower cancellation without leaking shared state.
- Semantic: disabled by default; explicitly opt-in public factual single-turn
  questions under an allowlisted context. Always isolated per client. Exclude
  personal/private data, conversational state, tools, time-dependent information,
  sensitive domains, and sampling. Preserve entities, numbers, negation, and
  requested constraints; unsupported ambiguity results in a miss.
- Similarity is candidate retrieval, not proof of equivalent answers. Evaluate
  a curated positive/negative set and calibrate a threshold; no arbitrary
  universal threshold. Use a conservative domain/intent guard and abstain unless
  equivalence is supported. Hold semantic serving disabled if false hits remain.
- Record exact/semantic/miss/bypass separately. TTL, expiry, client/context/settings
  isolation, false-hit rate, and coalescing under concurrency must be tested.

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
3. Exact cache: TTL/capacity/isolation/settings/coalescing tests and counted
   reduction of upstream calls on repeated requests.
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
