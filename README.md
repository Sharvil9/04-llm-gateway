# LLM gateway

An independent Python infrastructure project for a 24 GB MacBook: a centralized
LLM HTTP API with planned conservative semantic caching and process-local traffic
controls. **Current status: milestone 1, authenticated HTTP foundation.** Caching,
rate limiting, metrics, performance benchmarks, and genuine inference are not
complete. This is not yet a high-throughput gateway demonstration.

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
overloaded backend, and 504 upstream deadline. Upstream 429 maps to 503; client
quota enforcement with 429 is a later milestone. Model identity must match the
configured physical name; canonical-name variants require explicit configuration.

TOML controls total upstream deadline, connection-pool capacity, and maximum
backend response size. HTTPX phase timeouts alone are not total deadlines; an
outer async deadline includes pool wait and response reading. Response token
counts, when present, are provider-reported and checked for consistent totals.

## Real model: next milestone

The ready-to-use routing example is `config/ollama.toml`. On the target machine,
install [Ollama](https://ollama.com/download), then run sequentially:

```sh
ollama serve              # only if the Ollama app/service is not already running
ollama pull gemma3:1b     # run in another shell once the server is ready
export GATEWAY_CONFIG=config/ollama.toml
# Start the gateway as above, retaining GATEWAY_CLIENT_KEYS_JSON.
# Send the completion example with model="gemma".
```

This configuration targets the same documented non-streaming HTTP subset.
**Real inference has not been executed or benchmarked in this milestone.**
Gemma 3 1B is the first target; its download is about 806 MB, not its full runtime
memory requirement. Gemma Terms of Use apply. No Qwen models are selected.

## Evidence and limitations

Verification environment: Linux x86_64, Python 3.12.14, 8 GiB cgroup memory limit,
two CPU cores' quota (three visible logical CPUs). No Ollama or GPU tooling was
available initially. These are not MacBook results.

Milestone 1 evidence is recorded in `PLAN.md`: **58 passing automated tests**,
passing lint/format checks, and a two-server HTTP smoke demonstration. The tests exercise
concurrent async forwarding; they are not a throughput benchmark. **No observed
RPS, p95, cache savings, or model-generation performance is claimed yet.**

Use loopback and one worker. There is no admission control, quota, cache, metrics
endpoint, or shared state across workers yet. Pool capacity is not a rate limiter.
Task cancellation closes active HTTPX responses; socket-disconnect detection
after upload and slow-client ingress deadlines remain later work. HTTP cleanup
cannot guarantee the model server stops inference. Do not expose this foundation
directly to the Internet. Prompts, credentials, and upstream error bodies are not
logged by application code; access logging is disabled in examples.

Benchmark commands are intentionally pending until caching and traffic controls
exist. Milestone 6 will add a reproducible harness for uncached/exact/semantic,
concurrency, quota and failure scenarios, with raw results and separate mock,
gateway-overhead and real-model measurements. See [SPEC.md](SPEC.md) for cache
safety/security boundaries and [PLAN.md](PLAN.md) for the remaining gates.
