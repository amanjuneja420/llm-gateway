"""
LLM Gateway MVP
===============
Single-endpoint FastAPI gateway that routes prompts between two local Ollama
models using a heuristic (see router.py), semantically caches similar prompts
to skip redundant model calls (see cache.py), and logs every request to
SQLite (see db.py).

Section 4 status: heuristic routing + semantic cache + SQLite logging all
wired in. This is the complete gateway; benchmark.py (section 5) exercises
it and reports real numbers.
"""

import time

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from cache import SemanticCache
from db import init_db, log_request
from router import route

OLLAMA_URL = "http://localhost:11434/api/generate"

app = FastAPI(title="LLM Gateway MVP")

# Loaded once at startup - loading the sentence-transformer per request would
# dominate latency and defeat the point of caching.
cache = SemanticCache()
init_db()


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str
    model_used: str
    cache_hit: bool
    latency_ms: float


async def call_ollama(model: str, prompt: str) -> dict:
    """
    Call Ollama's generate endpoint, non-streaming, and return the full
    response payload - we need more than just the generated text: Ollama
    also reports load_duration, eval_count, and eval_duration per request,
    which we log to help tell apart "slow because CPU generation is slow"
    from "slow because the model keeps getting reloaded".

    keep_alive="30m" is passed explicitly on every request so the model
    stays resident in Ollama between requests for the duration of a
    benchmark run, regardless of Ollama's own default keep-alive setting
    (Ollama runs as an already-started background service here, so an
    OLLAMA_KEEP_ALIVE env var set in our own process's shell would not
    reach it - the per-request keep_alive field is the mechanism that
    actually takes effect).
    """
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m"}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()

    cached_entry, query_embedding = cache.find(req.prompt)
    if cached_entry is not None:
        latency_ms = (time.perf_counter() - start) * 1000
        log_request(req.prompt, "cache", True, latency_ms)
        return ChatResponse(
            response=cached_entry.response,
            model_used="cache",
            cache_hit=True,
            latency_ms=latency_ms,
        )

    model_used = route(req.prompt)
    ollama_data = await call_ollama(model_used, req.prompt)
    response_text = ollama_data["response"]
    cache.add(req.prompt, query_embedding, response_text, model_used)

    # Ollama reports these in nanoseconds (eval_count is a plain token
    # count, not a duration) - convert to ms so they're directly
    # comparable to our own latency_ms.
    load_duration_ms = ollama_data.get("load_duration", 0) / 1e6
    eval_count = ollama_data.get("eval_count", 0)
    eval_duration_ms = ollama_data.get("eval_duration", 0) / 1e6

    latency_ms = (time.perf_counter() - start) * 1000
    log_request(
        req.prompt,
        model_used,
        False,
        latency_ms,
        load_duration_ms=load_duration_ms,
        eval_count=eval_count,
        eval_duration_ms=eval_duration_ms,
    )

    return ChatResponse(
        response=response_text,
        model_used=model_used,
        cache_hit=False,
        latency_ms=latency_ms,
    )
