import asyncio
import json
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from llm_gateway.config import Settings
from llm_gateway.schemas import ChatRequest, ChatResponse


@dataclass
class ResponseCachePolicy:
    allowed: bool = False


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(code)


class HTTPBackend:
    """Fixed-route OpenAI-compatible text subset; never retries inference."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client

    async def complete(
        self, request: ChatRequest, *, cache_policy: ResponseCachePolicy | None = None
    ) -> ChatResponse:
        config = self.settings.config
        route = config.models.get(request.model)
        if route is None:
            raise GatewayError(404, "unknown_model", "The requested model is not configured")
        backend = config.backends[route.backend]
        payload = request.model_dump(exclude_none=True)
        payload["model"] = route.model
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if token := self.settings.backend_tokens.get(route.backend):
            headers["Authorization"] = f"Bearer {token}"

        try:
            # HTTPX phase timeouts alone do not enforce a whole-request deadline.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + config.request_timeout_seconds
            async with asyncio.timeout_at(deadline) as budget:
                async with self.client.stream(
                    "POST", f"{backend.base_url}/chat/completions", json=payload, headers=headers
                ) as upstream:
                    try:
                        if upstream.status_code == 429 or upstream.status_code >= 500:
                            raise GatewayError(
                                503, "backend_unavailable", "The backend is unavailable"
                            )
                        if upstream.status_code != 200:
                            raise GatewayError(
                                502, "backend_error", "The backend rejected the request"
                            )
                        if (
                            upstream.headers.get("content-type", "")
                            .split(";", 1)[0]
                            .strip()
                            .lower()
                            != "application/json"
                            or upstream.headers.get("content-encoding", "identity").lower()
                            != "identity"
                        ):
                            raise GatewayError(
                                502, "invalid_backend_response", "Invalid backend response"
                            )
                        data = bytearray()
                        # Identity encoding only: own the response close in the
                        # context below, rather than aiter_bytes() auto-closing
                        # under an expiring deadline at the end of iteration.
                        async for chunk in upstream.stream:
                            if len(data) + len(chunk) > config.max_response_bytes:
                                raise GatewayError(
                                    502,
                                    "invalid_backend_response",
                                    "Backend response exceeds size limit",
                                )
                            data.extend(chunk)
                        result = ChatResponse.model_validate(json.loads(data))
                        if result.model != route.model:
                            raise GatewayError(
                                502,
                                "invalid_backend_response",
                                "Backend returned an unexpected model",
                            )
                        result = result.model_copy(update={"model": request.model})
                        if cache_policy is not None:
                            cache_policy.allowed = not (
                                "cache-control" in upstream.headers
                                or "expires" in upstream.headers
                                or "set-cookie" in upstream.headers
                                or "vary" in upstream.headers
                                or "cookie" in upstream.request.headers
                            )
                    finally:
                        # A second deadline cancellation must not interrupt cleanup
                        # already triggered by disconnect/caller cancellation. Close
                        # first, then reject late success with the monotonic check.
                        if not budget.expired():
                            budget.reschedule(None)
            # Synchronous validation or a delayed event loop can finish after the
            # budget before the cancellation callback runs. Never return late success.
            if loop.time() >= deadline:
                raise TimeoutError
            return result
        except (TimeoutError, httpx.TimeoutException):
            raise GatewayError(504, "backend_timeout", "The backend request timed out") from None
        except httpx.RequestError:
            raise GatewayError(
                503, "backend_unavailable", "Cannot reach the configured backend"
            ) from None
        except (ValidationError, ValueError, UnicodeError, RecursionError):
            raise GatewayError(
                502, "invalid_backend_response", "Invalid backend response"
            ) from None
