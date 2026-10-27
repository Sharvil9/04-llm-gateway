# LLM gateway specification

## Status and architecture

Milestone 1 is an async HTTP gateway foundation. Later milestones must add and
measure caching, quotas, and concurrency controls before this is a complete
high-throughput gateway. Python 3.12+, FastAPI/Pydantic v2, Uvicorn, and one shared
HTTPX `AsyncClient` per application lifespan. One worker; no Redis or database.

Flow: bearer authentication → body-size boundary → strict request validation →
fixed public-model lookup → HTTP backend adapter → validated normalized response.
The deterministic mock is a separate local HTTP service, so demonstrations cross
a real network boundary. The adapter targets the documented OpenAI-compatible
non-streaming subset used by Ollama. There is no automatic model fallback.

## Supported contract (milestone 1)

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
  422 invalid schema/JSON syntax, 404 unknown model,
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

Reuse a bounded HTTP connection pool. Configurable total upstream deadline,
including pool wait; short connect/pool timeouts; bounded response size. Disable
automatic retries: a timed-out generation might already have run. Propagate task
cancellation and close responses/connections at shutdown. HTTP pooling is not
admission control; explicit bounded queues and request quotas arrive in milestone 5.
Task cancellation closes the HTTP response; detecting a caller socket disconnect
after request upload is deferred to milestone 5. Closing an HTTP call does not
guarantee the inference server stops generation. Ingress slow-client deadlines
are also deferred; loopback-only operation is the supported initial deployment.

## Selected next integrations

Ollama with `gemma3:1b` is the first real backend target, appropriate for a 24 GB
MacBook and smaller CPU-only demonstrations. No Qwen models. Configure the same
adapter against `http://127.0.0.1:11434/v1`; verify a genuine inference round trip
before calling integration complete. Do not load multiple models simultaneously.

For semantic embeddings, evaluate `sentence-transformers/all-MiniLM-L6-v2`
(384 dimensions, CPU) using normalized vectors and bounded NumPy cosine search.
The model truncates beyond 256 word pieces: semantic eligibility must reject
overlength prompts instead of silently losing constraints. No vector database
is justified for the initial small cache. Run embedding work
off the event loop with bounded work; measure its latency before enabling it.

## Planned cache policy (not implemented in milestone 1)

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

## Planned traffic controls and observability

Atomic per-client token bucket for request counts using monotonic time;
independent per-client in-flight and global upstream limits with bounded queue
wait. Cache hits still consume request quota. Return 429 and useful `Retry-After`
for client limits, 503 for global capacity, and release permits on every exit.
All state is process-local; run exactly one worker. Token estimates, if used, are
labeled separately from reported usage. Do not claim distributed enforcement.

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
