# LLM gateway

An independent Python infrastructure project for a 24 GB MacBook: a centralized
LLM HTTP API with planned conservative semantic caching and process-local traffic
controls. **Current status: milestone 3, client quotas and bounded inference
admission implemented; genuine Gemma inference verified.** Caching, metrics, and performance
benchmarks remain pending. This is not yet a high-throughput gateway demonstration.

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

Milestones 1–3 evidence is recorded in `PLAN.md`, including **85 passing tests**,
lint/format checks, deterministic HTTP enforcement/deadline probes, and real-model
verification. Concurrent forwarding tests are not a load benchmark.
**No RPS, p95, cache savings, or model-generation throughput is claimed yet.**

Use loopback and one worker. There is no cache, metrics endpoint, ingress
slow-client deadline, or shared state across workers. Pool capacity and inference
admission are distinct from request quotas. Task/ASGI-disconnect cancellation
closes active HTTPX responses; the automated disconnect tests inject ASGI events.
HTTP cleanup
cannot guarantee the model server stops inference. Do not expose this foundation
directly to the Internet. Prompts, credentials, and upstream error bodies are not
logged by application code; access logging is disabled in examples.

Throughput benchmark commands remain pending until caching and metrics exist.
Milestone 6 will add a reproducible harness for uncached/exact/semantic,
concurrency, quota and failure scenarios, with raw results and separate mock,
gateway-overhead and real-model measurements. See [SPEC.md](SPEC.md) for cache
safety/security boundaries and [PLAN.md](PLAN.md) for the remaining gates.
