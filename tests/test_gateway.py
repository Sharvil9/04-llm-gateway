import asyncio
import hashlib
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from pydantic import ValidationError

from llm_gateway.app import create_app
from llm_gateway.config import GatewayConfig, Settings, load_settings

CLIENT_KEY = "test-client-key-" + "a" * 32
BACKEND_KEY = "test-backend-key-" + "b" * 32
AUTH = {"Authorization": f"Bearer {CLIENT_KEY}"}
QUESTION = {"model": "demo", "messages": [{"role": "user", "content": "Hello"}]}


def make_settings(**overrides):
    values = {
        "backends": {
            "local": {"base_url": "http://backend.invalid/v1", "api_key_env": "BACKEND_KEY"}
        },
        "models": {"demo": {"backend": "local", "model": "physical-model"}},
        "request_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return Settings(
        config=GatewayConfig.model_validate(values),
        client_key_hashes={hashlib.sha256(CLIENT_KEY.encode()).digest(): "client-a"},
        backend_tokens={"local": BACKEND_KEY},
    )


def completion(**overrides):
    result = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "physical-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello back"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    result.update(overrides)
    return result


@asynccontextmanager
async def gateway(handler=None, *, settings=None):
    if handler is None:

        def handler(request):
            return httpx.Response(200, json=completion())

    app = create_app(settings or make_settings(), transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as client:
            yield client, app


async def test_liveness_is_public_and_model_listing_is_authenticated():
    async with gateway() as (client, _):
        assert (await client.get("/healthz")).json() == {"status": "ok"}
        assert (await client.get("/v1/models")).status_code == 401
        response = await client.get("/v1/models", headers=AUTH)
        assert response.status_code == 200
        assert response.json() == {"object": "list", "data": [{"id": "demo", "object": "model"}]}
        assert "backend.invalid" not in response.text
        assert (await client.get("/docs")).status_code == 404


@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Basic abc", "Bearer"])
async def test_authentication_precedes_body_validation(authorization):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion())

    headers = {"Authorization": authorization} if authorization is not None else {}
    async with gateway(handler) as (client, _):
        response = await client.post(
            "/v1/chat/completions", content=b"{secret malformed", headers=headers
        )
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert "secret" not in response.text
        assert calls == []


async def test_duplicate_authorization_headers_are_rejected():
    async with gateway() as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json=QUESTION,
            headers=[("Authorization", f"Bearer {CLIENT_KEY}")] * 2,
        )
        assert response.status_code == 401


async def test_forwarding_preserves_settings_and_normalizes_private_backend_fields():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion(private_backend_diagnostic="secret detail"))

    payload = QUESTION | {"temperature": 0.4, "top_p": 0.8, "max_tokens": 37, "seed": 42}
    async with gateway(handler) as (client, _):
        response = await client.post("/v1/chat/completions", json=payload, headers=AUTH)
        assert response.status_code == 200
        assert response.json() == completion(model="demo")
    assert len(calls) == 1
    forwarded = calls[0]
    assert str(forwarded.url) == "http://backend.invalid/v1/chat/completions"
    assert json.loads(forwarded.content) == payload | {"model": "physical-model", "stream": False}
    assert forwarded.headers["authorization"] == f"Bearer {BACKEND_KEY}"
    assert CLIENT_KEY not in str(forwarded.headers)


async def test_missing_provider_usage_is_omitted_without_fabricating_counts():
    async with gateway(lambda _: httpx.Response(200, json=completion(usage=None))) as (client, _):
        response = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert response.status_code == 200
        assert response.json().get("usage") is None


@pytest.mark.parametrize(
    "changes",
    [
        {"stream": True},
        {"stream": 0},
        {"max_tokens": True},
        {"max_tokens": "10"},
        {"max_tokens": 2049},
        {"temperature": "0.5"},
        {"temperature": -0.1},
        {"top_p": 0.0},
        {"tools": []},
        {"messages": []},
        {"messages": [{"role": "tool", "content": "sensitive marker"}]},
        {"messages": [{"role": "assistant", "content": "sensitive marker"}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "secret"}]}]},
        {"messages": [{"role": "user", "content": "x" * 16_001}]},
        {"messages": [{"role": "user", "content": "x"}] * 33},
        {"messages": [{"role": "user", "content": "x" * 11_000}] * 3},
    ],
)
async def test_invalid_requests_never_reach_backend_or_echo_inputs(changes):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion())

    async with gateway(handler) as (client, _):
        response = await client.post("/v1/chat/completions", json=QUESTION | changes, headers=AUTH)
        assert response.status_code == 422
        assert set(response.json()) == {"error"}
        assert "sensitive marker" not in response.text
        assert calls == []


async def test_malformed_json_and_wrong_content_type_have_distinct_errors():
    async with gateway() as (client, _):
        malformed = await client.post(
            "/v1/chat/completions",
            content=b'{"private":',
            headers=AUTH | {"Content-Type": "application/json"},
        )
        wrong_type = await client.post(
            "/v1/chat/completions", content=json.dumps(QUESTION), headers=AUTH
        )
        assert malformed.status_code == 422
        assert "private" not in malformed.text
        assert wrong_type.status_code == 415


async def test_deeply_nested_json_is_rejected_without_backend_io():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion())

    async with gateway(handler) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            content=b'{"private":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}",
            headers=AUTH | {"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "private" not in response.text
        assert calls == []


async def test_body_limit_applies_to_declared_and_chunked_requests():
    async def chunks():
        yield b" " * 40_000
        yield b" " * 40_000

    headers = AUTH | {"Content-Type": "application/json"}
    async with gateway() as (client, _):
        declared = await client.post("/v1/chat/completions", content=b" " * 65_537, headers=headers)
        chunked = await client.post("/v1/chat/completions", content=chunks(), headers=headers)
        assert declared.status_code == 413
        assert chunked.status_code == 413


async def test_unconfigured_model_does_not_trigger_network_io():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion())

    async with gateway(handler) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json=QUESTION | {"model": "http://untrusted.invalid"},
            headers=AUTH,
        )
        assert response.status_code == 404
        assert calls == []


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_upstream_capacity_errors_are_sanitized_and_never_retried(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"secret": "internal backend details"})

    async with gateway(handler) as (client, _):
        response = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "backend_unavailable"
        assert "internal backend details" not in response.text
        assert len(calls) == 1


async def test_upstream_redirect_is_rejected_without_following_it():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(307, headers={"Location": "http://untrusted.invalid"})

    async with gateway(handler) as (client, _):
        response = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert response.status_code == 502
        assert len(calls) == 1


@pytest.mark.parametrize(
    "result",
    [
        completion(model="different-model"),
        completion(choices=[]),
        completion(
            choices=[
                {
                    "index": False,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ]
        ),
        completion(usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 999}),
        {"secret": "unexpected raw provider response"},
    ],
)
async def test_invalid_backend_schemas_are_not_forwarded(result):
    async with gateway(lambda _: httpx.Response(200, json=result)) as (client, _):
        response = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_backend_response"
        assert "unexpected raw provider response" not in response.text


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="private non-JSON response"),
        httpx.Response(200, content=b"{", headers={"Content-Type": "application/json"}),
        httpx.Response(
            200,
            content=b"[" * 200 + b"0" + b"]" * 200,
            headers={"Content-Type": "application/json"},
        ),
        httpx.Response(200, content=b"x" * 1025, headers={"Content-Type": "application/json"}),
    ],
)
async def test_invalid_or_oversized_backend_bodies_are_controlled(response):
    async with gateway(lambda _: response, settings=make_settings(max_response_bytes=1024)) as (
        client,
        _,
    ):
        result = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert result.status_code == 502
        assert "private" not in result.text


@pytest.mark.parametrize(
    ("failure", "status"),
    [(httpx.ConnectError("private address"), 503), (httpx.ReadTimeout("private timeout"), 504)],
)
async def test_network_errors_are_sanitized(failure, status):
    def handler(_):
        raise failure

    async with gateway(handler) as (client, _):
        response = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert response.status_code == status
        assert "private" not in response.text


async def test_total_deadline_cancels_slow_backend_without_retry():
    attempts = 0
    cancelled = asyncio.Event()

    async def handler(_):
        nonlocal attempts
        attempts += 1
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async with gateway(handler, settings=make_settings(request_timeout_seconds=0.02)) as (
        client,
        _,
    ):
        response = await client.post("/v1/chat/completions", json=QUESTION, headers=AUTH)
        assert response.status_code == 504
        assert attempts == 1
        assert cancelled.is_set()


async def test_request_cancellation_closes_active_upstream_response():
    started = asyncio.Event()
    closed = asyncio.Event()

    class WaitingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self):
            closed.set()

    async with gateway(
        lambda _: httpx.Response(
            200, headers={"Content-Type": "application/json"}, stream=WaitingStream()
        )
    ) as (client, _):
        task = asyncio.create_task(client.post("/v1/chat/completions", json=QUESTION, headers=AUTH))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert closed.is_set()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def test_distinct_requests_progress_concurrently_without_cross_talk():
    arrived = set()
    all_arrived = asyncio.Event()

    async def handler(request):
        text = json.loads(request.content)["messages"][-1]["content"]
        arrived.add(text)
        if len(arrived) == 6:
            all_arrived.set()
        await asyncio.wait_for(all_arrived.wait(), timeout=1)
        result = completion()
        result["choices"][0]["message"]["content"] = text
        return httpx.Response(200, json=result)

    async with gateway(handler) as (client, _):
        requests = [
            client.post(
                "/v1/chat/completions",
                json={"model": "demo", "messages": [{"role": "user", "content": f"request-{i}"}]},
                headers=AUTH,
            )
            for i in range(6)
        ]
        responses = await asyncio.gather(*requests)
        assert all(response.status_code == 200 for response in responses)
        assert [r.json()["choices"][0]["message"]["content"] for r in responses] == [
            f"request-{i}" for i in range(6)
        ]


async def test_shared_client_is_closed_when_lifespan_ends():
    async with gateway() as (_, app):
        upstream_client = app.state.http_client
        assert not upstream_client.is_closed
    assert upstream_client.is_closed


@pytest.mark.parametrize("credential_json", [None, "{}", '{"client":"short"}', "not JSON"])
def test_missing_or_invalid_credentials_fail_closed_without_leaking_input(
    monkeypatch, credential_json
):
    monkeypatch.setenv("GATEWAY_CONFIG", "config/mock.toml")
    if credential_json is None:
        monkeypatch.delenv("GATEWAY_CLIENT_KEYS_JSON", raising=False)
    else:
        monkeypatch.setenv("GATEWAY_CLIENT_KEYS_JSON", credential_json)
    with pytest.raises(RuntimeError, match="GATEWAY_CLIENT_KEYS_JSON") as caught:
        load_settings()
    assert "short" not in str(caught.value)


def test_duplicate_client_keys_are_rejected(monkeypatch):
    monkeypatch.setenv("GATEWAY_CONFIG", "config/mock.toml")
    monkeypatch.setenv(
        "GATEWAY_CLIENT_KEYS_JSON", json.dumps({"alice": CLIENT_KEY, "bob": CLIENT_KEY})
    )
    with pytest.raises(RuntimeError, match="distinct"):
        load_settings()


@pytest.mark.parametrize(
    "destination",
    [
        "http://backend.invalid/\x7f",
        "http://back\x7fend.invalid/v1",
        "http://backend.invalid:invalid/v1",
        "http://secret:credential@backend.invalid/v1",
        "http://backend.invalid/v1?route=untrusted",
        "file:///etc/passwd",
    ],
)
def test_invalid_backend_destinations_fail_configuration_validation(destination):
    with pytest.raises(ValidationError):
        make_settings(backends={"local": {"base_url": destination}})
