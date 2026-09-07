"""
LLM Gateway MVP
===============
Single-endpoint FastAPI gateway that routes prompts between two local Ollama
models using a heuristic (see router.py), semantically caches similar prompts
to skip redundant model calls (see cache.py), logs every request to SQLite
(see db.py), and fails over to Gemini if the routed local Ollama call fails
(see call_gemini and chat() below).

Phase 1 additions on top of that: a GET /health endpoint, a per-request
correlation ID threaded through the cache/router/model-call/log path, input
validation on the prompt, and cache persistence across restarts (see
lifespan() below and cache.py's save()/load()).
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cache import SemanticCache
from db import init_db, log_request
from router import route

load_dotenv()  # reads .env in the project root, if present; no-op otherwise

OLLAMA_URL = "http://localhost:11434/api/generate"
# Lightweight reachability check for /health - lists local models instead of
# running a real generation, so it doesn't cost CPU time or tie up a model.
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
# Also the failover trigger threshold: any Ollama call that runs past this
# many seconds (or raises any other exception) counts as a failure and gets
# retried against Gemini - see chat() below.
OLLAMA_TIMEOUT_SECONDS = 120.0

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Model selection history, in order:
#   1. gemini-1.5-flash (originally planned) - deprecated, 404s on the live
#      API. Confirmed by querying https://generativelanguage.googleapis.com
#      /v1beta/models for this key and finding it absent from the list.
#   2. gemini-3.1-flash (considered next, on the assumption a newer point
#      release would exist) - does NOT exist on the live API either.
#      Confirmed the same way: absent from /v1beta/models. Only
#      gemini-3.1-flash-lite/-lite-preview/-image/-image-preview/-tts-preview
#      exist under the 3.1 line, not a plain gemini-3.1-flash.
#   3. gemini-3.8-flash (newest flash model actually in the list) - exists,
#      but unreliable in real testing: 4 of 6 real generateContent calls
#      across two separate testing sessions returned 503 Service
#      Unavailable (likely an overloaded/rate-limited preview-tier model).
#   4. gemini-2.5-flash (chosen) - 4 of 4 real generateContent calls
#      succeeded across the same testing sessions. This is the failover
#      backend for when the local model has already failed, so a model
#      that reliably answers beats a newer one that occasionally doesn't -
#      picked on that basis, not because it's the newest available.
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
# Same models-list endpoint used to pick GEMINI_MODEL above, reused here as
# the cheap reachability check for /health (lists models instead of
# generating anything).
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_TIMEOUT_SECONDS = 60.0

# /health uses a short timeout of its own - it should fail fast, not wait
# anywhere near as long as a real generation call would.
HEALTH_CHECK_TIMEOUT_SECONDS = 5.0

# Prompts longer than this are rejected with 400 rather than silently
# accepted and passed to a model. 2000 chars is generous for the kind of
# short-to-medium prompts this gateway is exercised with (the benchmark's
# longest prompt is well under 300 chars) while still catching obviously
# oversized/malformed input.
MAX_PROMPT_LENGTH = 2000

# Loaded once at startup - loading the sentence-transformer per request would
# dominate latency and defeat the point of caching.
cache = SemanticCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    loaded = cache.load()
    if loaded:
        print(f"Loaded {loaded} cache entries from disk (cache_state.json/.npz)")
    yield
    saved = cache.save()
    if saved:
        print(f"Saved {saved} cache entries to disk (cache_state.json/.npz)")


app = FastAPI(title="LLM Gateway MVP", lifespan=lifespan)


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str
    model_used: str
    cache_hit: bool
    latency_ms: float
    failed_over: bool = False
    request_id: str


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


@app.get("/health")
async def health() -> dict:
    """
    Lightweight reachability check for both backends - lists models rather
    than generating anything, so it's fast and doesn't cost CPU/quota.
    Doesn't touch the cache, router, or DB; this is purely "can we reach
    these two services right now".
    """
    ollama_status = "down"
    try:
        async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT_SECONDS) as client:
            resp = await client.get(OLLAMA_TAGS_URL)
            if resp.status_code == 200:
                ollama_status = "up"
    except Exception:
        ollama_status = "down"

    if not GEMINI_API_KEY:
        gemini_status = "unconfigured"
    else:
        gemini_status = "down"
        try:
            async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT_SECONDS) as client:
                resp = await client.get(
                    GEMINI_MODELS_URL, headers={"x-goog-api-key": GEMINI_API_KEY}
                )
                if resp.status_code == 200:
                    gemini_status = "up"
        except Exception:
            gemini_status = "down"

    return {
        "gateway": "up",
        "ollama": ollama_status,
        "gemini": gemini_status,
        "cache_size": len(cache),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # Generated once per request and carried through every observable
    # artifact of handling it - the DB log row and the response itself, plus
    # any print() emitted along the way (see the failover branch below) - so
    # one request's full path through the system can be traced by grepping
    # for this one value.
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty or whitespace-only")
    if len(req.prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"prompt exceeds MAX_PROMPT_LENGTH ({MAX_PROMPT_LENGTH} chars)",
        )

    cached_entry, query_embedding = cache.find(req.prompt)
    if cached_entry is not None:
        latency_ms = (time.perf_counter() - start) * 1000
        log_request(req.prompt, "cache", True, latency_ms, request_id=request_id)
        return ChatResponse(
            response=cached_entry.response,
            model_used="cache",
            cache_hit=True,
            latency_ms=latency_ms,
            request_id=request_id,
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
        print(
            f"[{request_id}] Ollama call failed ({type(exc).__name__}: {exc}) "
            "- failing over to Gemini"
        )
        response_text = await call_gemini(req.prompt)
        model_used = GEMINI_MODEL
        failed_over = True

    cache.add(req.prompt, query_embedding, response_text, model_used)
    # Persist immediately rather than only on clean shutdown (see lifespan()
    # above) - a crash or force-kill doesn't fire ASGI shutdown handlers, and
    # in practice that's exactly when you'd most want the cache not to be
    # lost. At this cache's demo scale (dozens to low hundreds of entries),
    # rewriting the whole file after every new entry is cheap enough not to
    # matter for latency.
    cache.save()

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
        request_id=request_id,
    )

    return ChatResponse(
        response=response_text,
        model_used=model_used,
        cache_hit=False,
        latency_ms=latency_ms,
        failed_over=failed_over,
        request_id=request_id,
    )
