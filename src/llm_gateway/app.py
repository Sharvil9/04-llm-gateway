import hashlib
import hmac
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from llm_gateway.backend import GatewayError, HTTPBackend
from llm_gateway.config import Settings, load_settings
from llm_gateway.schemas import ChatRequest, ChatResponse

MAX_BODY_BYTES = 65_536


def error_response(status: int, code: str, message: str) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return JSONResponse({"error": {"code": code, "message": message}}, status, headers=headers)


class RequestBoundary:
    """Authenticate before parsing; enforce actual body size, including chunked bodies."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
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
    async def complete(payload: ChatRequest):
        return await app.state.backend.complete(payload)

    return app
