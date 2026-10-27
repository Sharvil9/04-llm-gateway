"""Deterministic HTTP fixture. No language-model inference or token estimates."""

import asyncio
import hashlib

from fastapi import FastAPI

from llm_gateway.schemas import ChatRequest

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.post("/v1/chat/completions")
async def complete(request: ChatRequest):
    # Nonblocking fixed delay for manual demonstrations, not a model benchmark.
    await asyncio.sleep(0.01)
    digest = hashlib.sha256(request.model_dump_json().encode()).hexdigest()[:16]
    return {
        "id": f"chatcmpl-mock-{digest}",
        "object": "chat.completion",
        "created": 0,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Deterministic mock response."},
                "finish_reason": "stop",
            }
        ],
    }
