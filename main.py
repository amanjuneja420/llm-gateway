"""
LLM Gateway MVP
===============
Single-endpoint FastAPI gateway that routes prompts between two local Ollama
models using a heuristic (see router.py), semantically caches similar prompts
to skip redundant model calls (see cache.py), logs every request to SQLite
(see db.py), and fails over to Gemini if the routed local Ollama call fails
(see call_gemini and chat() below).

Section 4 status: heuristic routing + semantic cache + SQLite logging all
wired in. This is the complete gateway; benchmark.py (section 5) exercises
it and reports real numbers. Gemini failover is additive on top of that -
see the README's "Gemini failover" section for exactly what counts as a
failure.
"""

import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from cache import SemanticCache
from db import init_db, log_request
from router import route

load_dotenv()  # reads .env in the project root, if present; no-op otherwise

OLLAMA_URL = "http://localhost:11434/api/generate"
# Also the failover trigger threshold: any Ollama call that runs past this
# many seconds (or raises any other exception) counts as a failure and gets
# retried against Gemini - see chat() below.
OLLAMA_TIMEOUT_SECONDS = 120.0

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# gemini-1.5-flash (the originally planned model) has been deprecated and no
# longer exists on the live API - confirmed by querying
# https://generativelanguage.googleapis.com/v1beta/models for this key,
# which 404s on gemini-1.5-flash but lists gemini-3.8-flash as available.
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
GEMINI_TIMEOUT_SECONDS = 60.0

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
    failed_over: bool = False


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
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json()


async def call_gemini(prompt: str) -> str:
    """
    Call Gemini's REST API directly (no SDK - one more dependency isn't
    worth it when we already use httpx everywhere else). Used only as a
    failover backend when the routed local Ollama call fails - see chat().

    The API key is sent as the x-goog-api-key header, never as a URL query
    parameter. This is deliberate: httpx exceptions (and any traceback that
    ends up in a log) include the request URL in their message, so a
    key-in-URL would leak the secret into any error output. A header never
    appears in that message.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set (add it to .env - see .env.example) - "
            "cannot fail over to Gemini without it."
        )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT_SECONDS) as client:
        resp = await client.post(GEMINI_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


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
    failed_over = False
    load_duration_ms = None
    eval_count = None
    eval_duration_ms = None

    try:
        ollama_data = await call_ollama(model_used, req.prompt)
        response_text = ollama_data["response"]
        # Ollama reports these in nanoseconds (eval_count is a plain token
        # count, not a duration) - convert to ms so they're directly
        # comparable to our own latency_ms.
        load_duration_ms = ollama_data.get("load_duration", 0) / 1e6
        eval_count = ollama_data.get("eval_count", 0)
        eval_duration_ms = ollama_data.get("eval_duration", 0) / 1e6
    except Exception as exc:
        # Failure-triggered failover - deliberately NOT a full circuit
        # breaker: no failure counter, no open/half-open/closed state, no
        # cooldown before trying Ollama again on the next request. Every
        # request independently tries Ollama first and only falls back to
        # Gemini if THIS request's call fails. "Fails" means any exception
        # out of call_ollama, which covers both a raised error (connection
        # refused, HTTP 4xx/5xx from Ollama) and a timeout past
        # OLLAMA_TIMEOUT_SECONDS (httpx raises ReadTimeout/ConnectTimeout,
        # both plain exceptions here, once that threshold is hit).
        print(f"Ollama call failed ({type(exc).__name__}: {exc}) - failing over to Gemini")
        response_text = await call_gemini(req.prompt)
        model_used = GEMINI_MODEL
        failed_over = True

    cache.add(req.prompt, query_embedding, response_text, model_used)

    latency_ms = (time.perf_counter() - start) * 1000
    log_request(
        req.prompt,
        model_used,
        False,
        latency_ms,
        load_duration_ms=load_duration_ms,
        eval_count=eval_count,
        eval_duration_ms=eval_duration_ms,
        failed_over=failed_over,
    )

    return ChatResponse(
        response=response_text,
        model_used=model_used,
        cache_hit=False,
        latency_ms=latency_ms,
        failed_over=failed_over,
    )
