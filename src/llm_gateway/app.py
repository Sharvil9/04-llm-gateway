import asyncio
import hashlib
import hmac
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from llm_gateway.backend import GatewayError, HTTPBackend
from llm_gateway.config import Settings, load_settings
from llm_gateway.schemas import ChatRequest, ChatResponse
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

        async def timed_send(message):
            if message["type"] == "http.response.start":
                timings = [f"gateway;dur={(time.perf_counter() - started) * 1000:.3f}"]
                adapter_ms = scope.get("state", {}).get("adapter_ms")
                if adapter_ms is not None:
                    timings.append(f"adapter;dur={adapter_ms:.3f}")
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (b"server-timing", ", ".join(timings).encode("ascii"))
                ]
            await send(message)

        await self.handle(scope, receive, timed_send)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            await self.app(scope, receive, send)
            return

        settings = scope["app"].state.settings
        auth = [value for name, value in scope["headers"] if name.lower() == b"authorization"]
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
    settings: Settings | None = None, *, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = settings if settings is not None else load_settings()
        app.state.settings = resolved
        app.state.request_limiter = RequestLimiter(
            set(resolved.client_key_hashes.values()),
            resolved.config.rate_limit,
            resolved.config.client_rate_limits,
        )
        app.state.admission = InferenceAdmission(resolved.config.max_in_flight)
        # No environment proxies/netrc, redirects, or automatic inference retries.
        async with httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(resolved.config.request_timeout_seconds, connect=2.0, pool=1.0),
            limits=httpx.Limits(
                max_connections=resolved.config.max_connections,
                max_keepalive_connections=resolved.config.max_connections,
            ),
        ) as client:
            app.state.http_client = client
            app.state.backend = HTTPBackend(resolved, client)
            yield

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

    @app.post("/v1/chat/completions", response_model=ChatResponse, response_model_exclude_none=True)
    async def complete(payload: ChatRequest, request: Request):
        if payload.model not in app.state.settings.config.models:
            raise GatewayError(404, "unknown_model", "The requested model is not configured")

        async def infer():
            started = time.perf_counter()
            try:
                return await app.state.backend.complete(payload)
            finally:
                request.state.adapter_ms = (time.perf_counter() - started) * 1000

        async def disconnected():
            while True:
                if (await request.receive())["type"] == "http.disconnect":
                    return

        # No await in slot entry/exit: check-and-increment is atomic on this loop.
        with app.state.admission.slot():
            operation = asyncio.create_task(infer())
            watcher = asyncio.create_task(disconnected())
            try:
                done, _ = await asyncio.wait(
                    (operation, watcher), return_when=asyncio.FIRST_COMPLETED
                )
                if operation in done:
                    return await operation
                # Internal bookkeeping only: the peer has already disconnected.
                return Response(status_code=499)
            finally:
                for task in (operation, watcher):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(operation, watcher, return_exceptions=True)

    return app
