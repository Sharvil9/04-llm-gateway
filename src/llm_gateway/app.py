import asyncio
import hashlib
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from llm_gateway.backend import GatewayError, HTTPBackend, ResponseCachePolicy
from llm_gateway.cache import ExactResponseCache
from llm_gateway.config import Settings, load_settings
from llm_gateway.metrics import Metrics
from llm_gateway.schemas import ChatRequest, ChatResponse
from llm_gateway.semantic import SemanticCache
from llm_gateway.traffic import CapacityFull, InferenceAdmission, RequestLimiter

MAX_BODY_BYTES = 65_536


def error_response(
    status: int, code: str, message: str, *, retry_after: int | None = None
) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else {}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse({"error": {"code": code, "message": message}}, status, headers=headers)


class RequestBoundary:
    """Authenticate before parsing; enforce actual body size, including chunked bodies."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        response_status = 0
        cache_chunks = []
        cache_bytes = 0
        cache_overflow = False
        body_complete = False
        send_failed = False
        outcome = "completed"
        gateway_ms = None
        tracked = scope["method"] == "POST" and scope["path"] == "/v1/chat/completions"
        metrics = scope["app"].state.metrics
        if tracked:
            metrics.start()

        async def observed_receive():
            message = await receive()
            if message["type"] == "http.disconnect":
                scope.setdefault("state", {})["client_disconnected"] = True
            return message

        async def timed_send(message):
            nonlocal response_status, cache_bytes, cache_overflow, body_complete, send_failed
            nonlocal gateway_ms
            state = scope.get("state", {})
            if message["type"] == "http.response.start":
                response_status = message["status"]
                gateway_ms = (time.perf_counter() - started) * 1000
                timings = [f"gateway;dur={gateway_ms:.3f}"]
                adapter_ms = scope.get("state", {}).get("adapter_ms")
                if adapter_ms is not None:
                    timings.append(f"adapter;dur={adapter_ms:.3f}")
                if (semantic_ms := state.get("semantic_ms")) is not None:
                    timings.append(f"semantic;dur={semantic_ms:.3f}")
                if (exact_ms := state.get("exact_ms")) is not None:
                    timings.append(f"exact;dur={exact_ms:.3f}")
                if tracked and "exact_ms" in state:
                    residual = max(
                        0.0,
                        gateway_ms
                        - sum(
                            state.get(key, 0.0) for key in ("adapter_ms", "semantic_ms", "exact_ms")
                        ),
                    )
                    timings.append(f"residual;dur={residual:.3f}")
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (b"server-timing", ", ".join(timings).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ]
                if cache_status := state.get("cache_status"):
                    message["headers"].append((b"x-gateway-cache", cache_status.encode("ascii")))
            pending = state.get("cache_pending")
            if message["type"] == "http.response.body" and pending and not cache_overflow:
                body = message.get("body", b"")
                cache_bytes += len(body)
                limit = scope["app"].state.settings.config.exact_cache.max_bytes
                tag = state.get("cache_semantic")
                if cache_bytes + len(pending) + (tag.size if tag else 0) > limit:
                    cache_overflow = True
                    cache_chunks.clear()
                else:
                    cache_chunks.append(body)
            try:
                await send(message)
            except BaseException:
                send_failed = True
                raise
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                body_complete = True
            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
                and response_status == 200
                and pending
                and not cache_overflow
                and not state.get("client_disconnected", False)
                and state["cache_recheck"]()
            ):
                # Commit only after the complete ASGI send succeeds. This is not
                # a guarantee of receipt/ACK by the remote client.
                scope["app"].state.response_cache.put(
                    pending, b"".join(cache_chunks), semantic=state.get("cache_semantic")
                )
                state.pop("cache_pending", None)

        try:
            await self.handle(scope, observed_receive, timed_send)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException:
            outcome = "send_error" if send_failed else "unhandled_error"
            raise
        finally:
            if tracked:
                state = scope.get("state", {})
                if outcome == "completed" and (
                    state.get("client_disconnected", False) or response_status == 499
                ):
                    outcome = "disconnected"
                elif outcome == "completed" and not body_complete:
                    outcome = "aborted"
                durations = {"request_lifecycle": (time.perf_counter() - started) * 1000}
                for source, target in (
                    ("adapter_ms", "adapter"),
                    ("semantic_ms", "semantic"),
                    ("exact_ms", "exact_lookup"),
                ):
                    if source in state:
                        durations[target] = state[source]
                if gateway_ms is not None:
                    durations["gateway_to_start"] = gateway_ms
                    durations["gateway_residual"] = max(
                        0.0,
                        gateway_ms
                        - sum(
                            durations.get(key, 0.0)
                            for key in ("adapter", "semantic", "exact_lookup")
                        ),
                    )
                metrics.finish(
                    status=response_status or None,
                    complete=body_complete,
                    outcome=outcome,
                    durations_ms=durations,
                    cache_status=state.get("cache_status"),
                )
                if logger := scope["app"].state.request_logger:
                    logger.info(
                        json.dumps(
                            {
                                "event": "chat_request",
                                "status": response_status or None,
                                "outcome": outcome,
                                "cache": state.get("cache_status", "EARLY"),
                                "duration_ms": round(durations["request_lifecycle"], 3),
                            },
                            allow_nan=False,
                        )
                    )

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            await self.app(scope, receive, send)
            return

        settings = scope["app"].state.settings
        auth = [value for name, value in scope["headers"] if name.lower() == b"authorization"]
        if scope["path"] == "/v1/metrics":
            if settings.metrics_key_hash is None:
                await error_response(404, "not_found", "Endpoint is not enabled")(
                    scope, receive, send
                )
                return
            valid = False
            if len(auth) == 1:
                scheme, _, token = auth[0].partition(b" ")
                if scheme.lower() == b"bearer" and 32 <= len(token) <= 256:
                    valid = hmac.compare_digest(
                        hashlib.sha256(token).digest(), settings.metrics_key_hash
                    )
            if not valid:
                await error_response(
                    401, "unauthorized", "A valid operator metrics key is required"
                )(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return
        client_id = None
        if len(auth) == 1:
            scheme, _, token = auth[0].partition(b" ")
            if scheme.lower() == b"bearer" and 32 <= len(token) <= 256:
                digest = hashlib.sha256(token).digest()
                for expected, candidate in settings.client_key_hashes.items():
                    if hmac.compare_digest(digest, expected):
                        client_id = candidate
        if client_id is None:
            await error_response(401, "unauthorized", "A valid bearer API key is required")(
                scope, receive, send
            )
            return
        scope.setdefault("state", {})["client_id"] = client_id

        if scope["method"] == "POST" and scope["path"] == "/v1/chat/completions":
            retry_after = scope["app"].state.request_limiter.consume(client_id)
            if retry_after is not None:
                await error_response(
                    429,
                    "rate_limit_exceeded",
                    "Client request quota is exhausted",
                    retry_after=retry_after,
                )(scope, receive, send)
                return
            headers = {name.lower(): value for name, value in scope["headers"]}
            media_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
            if (
                media_type != b"application/json"
                or headers.get(b"content-encoding", b"identity").lower() != b"identity"
            ):
                await error_response(415, "unsupported_media_type", "Send uncompressed JSON")(
                    scope, receive, send
                )
                return

            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > MAX_BODY_BYTES:
                    await error_response(413, "request_too_large", "Request exceeds 64 KiB")(
                        scope, receive, send
                    )
                    return
                body.extend(chunk)
                if not message.get("more_body", False):
                    break

            replayed = False

            async def bounded_receive():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, bounded_receive, send)
            return
        await self.app(scope, receive, send)


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    embedding_encoder=None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = settings if settings is not None else load_settings()
        app.state.settings = resolved
        app.state.metrics = Metrics()
        app.state.request_logger = None
        handler = None
        if resolved.config.log_requests:
            logger = logging.Logger("llm_gateway.requests", level=logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            app.state.request_logger = logger
        app.state.request_limiter = RequestLimiter(
            set(resolved.client_key_hashes.values()),
            resolved.config.rate_limit,
            resolved.config.client_rate_limits,
        )
        app.state.admission = InferenceAdmission(resolved.config.max_in_flight)
        app.state.response_cache = ExactResponseCache(resolved)
        app.state.semantic_cache = None
        embedder = None
        if resolved.config.semantic_cache.enabled:
            # Optional imports/model loading never occur on the default path.
            from llm_gateway.embeddings import BoundedEmbedder, LocalEmbedder

            try:
                encoder = embedding_encoder or await asyncio.to_thread(
                    LocalEmbedder, Path(resolved.config.semantic_cache.model_dir)
                )
                embedder = BoundedEmbedder(encoder)
            except Exception:
                raise RuntimeError("Cannot load verified local semantic embedding model") from None
            app.state.semantic_cache = SemanticCache(
                app.state.response_cache, embedder, resolved.config.semantic_cache
            )
        # No environment proxies/netrc, redirects, or automatic inference retries.
        async with httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            # A stateless backend adapter must not share HTTP sessions between
            # clients or let a later cookie change an earlier cache partition.
            cookies=CookieJar(DefaultCookiePolicy(allowed_domains=[])),
            timeout=httpx.Timeout(resolved.config.request_timeout_seconds, connect=2.0, pool=1.0),
            limits=httpx.Limits(
                max_connections=resolved.config.max_connections,
                max_keepalive_connections=resolved.config.max_connections,
            ),
        ) as client:
            app.state.http_client = client
            app.state.backend = HTTPBackend(resolved, client)
            try:
                yield
            finally:
                if embedder is not None:
                    await embedder.close()
                if handler is not None:
                    handler.close()

    app = FastAPI(
        title="Local LLM Gateway",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(RequestBoundary)

    @app.exception_handler(GatewayError)
    async def gateway_error(_request: Request, exc: GatewayError):
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(CapacityFull)
    async def capacity_error(_request: Request, _exc: CapacityFull):
        # Capacity release time is unknown; do not invent a Retry-After value.
        return error_response(503, "gateway_overloaded", "Gateway inference capacity is full")

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError):
        # Default validation details can include prompt content and other input.
        return error_response(422, "invalid_request", "Request does not match the supported schema")

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException):
        if exc.status_code == 400:
            return error_response(400, "invalid_request", "Malformed JSON request")
        return error_response(exc.status_code, "http_error", "Endpoint or method is not supported")

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {"id": alias, "object": "model"} for alias in app.state.settings.config.models
            ],
        }

    @app.get("/v1/metrics")
    async def operational_metrics():
        snapshot = app.state.metrics.snapshot()
        snapshot["scope"] = "POST /v1/chat/completions; process-local; since startup"
        snapshot["counts"]["upstream_attempts"] = app.state.response_cache.stats.upstream_calls
        snapshot["gauges"]["inference_in_flight"] = app.state.admission.active
        snapshot["cache_storage"] = app.state.response_cache.snapshot()
        snapshot["semantic_lookup"] = (
            app.state.semantic_cache.snapshot() if app.state.semantic_cache else None
        )
        return snapshot

    @app.post("/v1/chat/completions", response_model=ChatResponse, response_model_exclude_none=True)
    async def complete(payload: ChatRequest, request: Request):
        if payload.model not in app.state.settings.config.models:
            raise GatewayError(404, "unknown_model", "The requested model is not configured")

        exact_started = time.perf_counter()
        cache = app.state.response_cache
        controls = {
            directive.split("=", 1)[0].strip().lower()
            for value in request.headers.getlist("cache-control")
            for directive in value.split(",")
        }
        pragma = {
            part.strip().lower()
            for value in request.headers.getlist("pragma")
            for part in value.split(",")
        }
        semantic_opt_in = request.headers.getlist("x-gateway-cache") == ["semantic"]
        opt_in = (
            (request.headers.getlist("x-gateway-cache") == ["exact"] or semantic_opt_in)
            and not controls.intersection({"no-store", "no-cache"})
            and "no-cache" not in pragma
            and "cookie" not in request.headers
            and not app.state.http_client.cookies
        )
        client_id = request.state.client_id
        key = cache.key_for(payload, client_id, opt_in, semantic=semantic_opt_in)
        request.state.cache_status = "BYPASS" if key is None else "MISS"
        cached = cache.get(key) if key is not None else None
        request.state.exact_ms = (time.perf_counter() - exact_started) * 1000
        if cached is not None:
            request.state.cache_status = "HIT"
            return Response(content=cached, media_type="application/json")

        async def disconnected():
            while True:
                if (await request.receive())["type"] == "http.disconnect":
                    request.state.client_disconnected = True
                    return

        async def serve():
            tag = None
            semantic = app.state.semantic_cache
            if key is not None and semantic_opt_in and semantic and cache.family_for(payload):
                started = time.perf_counter()
                try:
                    cached, tag = await semantic.lookup(payload, client_id)
                finally:
                    request.state.semantic_ms = (time.perf_counter() - started) * 1000
                if (
                    cached is not None
                    and cache.key_for(payload, client_id, opt_in, semantic=True) == key
                ):
                    request.state.cache_status = "SEMANTIC_HIT"
                    return Response(content=cached, media_type="application/json")

            cache_policy = ResponseCachePolicy()
            # No await in slot entry/exit: atomic on this event loop.
            with app.state.admission.slot():
                started = time.perf_counter()
                try:
                    cache.stats.upstream_calls += 1
                    result = await app.state.backend.complete(payload, cache_policy=cache_policy)
                    if result.usage is not None:
                        try:
                            app.state.metrics.record_upstream_usage(
                                result.usage.prompt_tokens,
                                result.usage.completion_tokens,
                                result.usage.total_tokens,
                            )
                        except ValueError:
                            # Telemetry bounds must not change an otherwise
                            # supported provider response or cache compatibility.
                            app.state.metrics.ignore_upstream_usage()
                finally:
                    request.state.adapter_ms = (time.perf_counter() - started) * 1000
                if (
                    key is not None
                    and cache_policy.allowed
                    and result.choices[0].finish_reason == "stop"
                    and result.choices[0].message.content.strip()
                    and not request.scope["state"].get("client_disconnected", False)
                    and not app.state.http_client.cookies
                ):
                    request.state.cache_pending = key
                    request.state.cache_semantic = tag

                    def recheck():
                        return (
                            not app.state.http_client.cookies
                            and cache.key_for(payload, client_id, opt_in, semantic=semantic_opt_in)
                            == key
                            and (
                                tag is None
                                or cache.semantic_partition(
                                    payload, client_id, semantic.embedder.fingerprint
                                )
                                == tag.partition
                            )
                        )

                    request.state.cache_recheck = recheck
                return result

        # Watch disconnects during embedding as well as upstream I/O. Worker
        # cancellation retains its own capacity until the native call finishes.
        operation = asyncio.create_task(serve())
        watcher = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait((operation, watcher), return_when=asyncio.FIRST_COMPLETED)
            if operation in done:
                return await operation
            return Response(status_code=499)
        finally:
            for task in (operation, watcher):
                if not task.done():
                    task.cancel()
            await asyncio.gather(operation, watcher, return_exceptions=True)

    return app
