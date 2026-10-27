import asyncio
import json

import httpx
from pydantic import ValidationError

from llm_gateway.config import Settings
from llm_gateway.schemas import ChatRequest, ChatResponse


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

    async def complete(self, request: ChatRequest) -> ChatResponse:
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
            async with asyncio.timeout(config.request_timeout_seconds):
                async with self.client.stream(
                    "POST", f"{backend.base_url}/chat/completions", json=payload, headers=headers
                ) as upstream:
                    if upstream.status_code == 429 or upstream.status_code >= 500:
                        raise GatewayError(503, "backend_unavailable", "The backend is unavailable")
                    if upstream.status_code != 200:
                        raise GatewayError(502, "backend_error", "The backend rejected the request")
                    if (
                        upstream.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                        != "application/json"
                        or upstream.headers.get("content-encoding", "identity").lower()
                        != "identity"
                    ):
                        raise GatewayError(
                            502, "invalid_backend_response", "Invalid backend response"
                        )
                    data = bytearray()
                    async for chunk in upstream.aiter_bytes():
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
                            502, "invalid_backend_response", "Backend returned an unexpected model"
                        )
                    return result.model_copy(update={"model": request.model})
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
