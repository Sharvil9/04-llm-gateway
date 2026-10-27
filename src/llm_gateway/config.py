import hashlib
import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import httpx
from pydantic import Field, field_validator, model_validator

from llm_gateway.schemas import StrictModel


class BackendConfig(StrictModel):
    base_url: str
    api_key_env: str | None = None

    @field_validator("base_url")
    @classmethod
    def fixed_http_destination(cls, value: str) -> str:
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError(
                "backend must be a fixed HTTP(S) URL without credentials/query/fragment"
            )
        # Accessing .port also validates malformed ports.
        _ = url.port
        try:
            httpx.URL(value)
        except httpx.InvalidURL:
            raise ValueError("invalid HTTP backend URL") from None
        return value.rstrip("/")

    @field_validator("api_key_env")
    @classmethod
    def environment_name(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
            raise ValueError("invalid backend credential environment variable name")
        return value


class ModelRoute(StrictModel):
    backend: Annotated[str, Field(min_length=1, max_length=128)]
    model: Annotated[str, Field(min_length=1, max_length=128)]


class GatewayConfig(StrictModel):
    backends: dict[str, BackendConfig]
    models: dict[str, ModelRoute]
    request_timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 30.0
    max_connections: Annotated[int, Field(ge=1, le=64)] = 16
    max_response_bytes: Annotated[int, Field(ge=1024, le=1_048_576)] = 262_144

    @model_validator(mode="after")
    def valid_routes(self) -> "GatewayConfig":
        if not self.backends or not self.models:
            raise ValueError("at least one backend and model must be configured")
        for alias, route in self.models.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", alias):
                raise ValueError("invalid public model alias")
            if route.backend not in self.backends:
                raise ValueError("model refers to an unconfigured backend")
        return self


@dataclass(frozen=True)
class Settings:
    config: GatewayConfig
    client_key_hashes: dict[bytes, str] = field(repr=False)
    backend_tokens: dict[str, str] = field(default_factory=dict, repr=False)


def load_settings() -> Settings:
    """Load trusted operator configuration without leaking validation input/secrets."""
    try:
        path = Path(os.environ.get("GATEWAY_CONFIG", "config/mock.toml"))
        config = GatewayConfig.model_validate(tomllib.loads(path.read_text()))
    except (OSError, ValueError):
        raise RuntimeError("Invalid gateway TOML configuration; check GATEWAY_CONFIG") from None

    try:
        credentials = json.loads(os.environ.get("GATEWAY_CLIENT_KEYS_JSON", ""))
        if not isinstance(credentials, dict) or not 1 <= len(credentials) <= 128:
            raise ValueError
        hashes: dict[bytes, str] = {}
        for client_id, key in credentials.items():
            if not isinstance(client_id, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{1,64}", client_id
            ):
                raise ValueError
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", key):
                raise ValueError
            digest = hashlib.sha256(key.encode()).digest()
            if digest in hashes:
                raise ValueError
            hashes[digest] = client_id
    except (ValueError, TypeError):
        raise RuntimeError(
            "GATEWAY_CLIENT_KEYS_JSON must map client IDs to distinct 32–256 character "
            "URL-safe bearer keys (maximum 128 clients)"
        ) from None

    backend_tokens = {}
    for name, backend in config.backends.items():
        if backend.api_key_env:
            token = os.environ.get(backend.api_key_env, "")
            if not token or any(ord(char) < 33 or ord(char) > 126 for char in token):
                raise RuntimeError("Missing or invalid configured backend API credential")
            backend_tokens[name] = token
    return Settings(config=config, client_key_hashes=hashes, backend_tokens=backend_tokens)
