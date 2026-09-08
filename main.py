"""
LLM Gateway MVP
===============
Single-endpoint FastAPI gateway that routes prompts between two local Ollama
models using a heuristic (see router.py), semantically caches similar prompts
to skip redundant model calls (see cache.py), logs every request to SQLite
(see db.py), and fails over through a three-tier chain - Ollama, then
Gemini, then Groq - if the routed local call fails (see backends.py's
Backend classes and chat() below).

Phase 1 additions on top of that: a GET /health endpoint, a per-request
correlation ID threaded through the cache/router/model-call/log path, input
validation on the prompt, and cache persistence across restarts (see
lifespan() below and cache.py's save()/load()).

Phase 2: call_ollama()/call_gemini() were pulled out into backends.py as
OllamaBackend/GeminiBackend, both implementing the same Backend interface
(generate(prompt) -> BackendResponse) - a pure refactor, chat()'s behavior
was unchanged by it. Groq was then added as a third backend and the
failover chain extended to try all three in order, stopping at the first
success (see the loop in chat() below) and returning 502 only if all three
fail. served_by ("ollama"/"gemini"/"groq"/"cache") records which one
actually answered; failed_over is explicitly `served_by in ("gemini",
"groq")`, kept for backward compatibility with Phase 1's simpler two-tier
framing - it's True only when an actual backend failover occurred, and
False for both "ollama" and "cache" (a cache hit is the healthy fast path,
not a failover, and is set explicitly rather than relying on a default).
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backends import GeminiBackend, GroqBackend, OllamaBackend
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
# Same models-list endpoint used to pick GEMINI_MODEL above, reused here as
# the cheap reachability check for /health (lists models instead of
# generating anything).
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_TIMEOUT_SECONDS = 60.0

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Picked by querying Groq's live /openai/v1/models list for this key rather
# than assuming a name (same lesson as the Gemini model situation above).
# The full list returned: meta-llama/llama-prompt-guard-2-86m/-22m (input
# classifiers, not instruction models), groq/compound and compound-mini
# (Groq's own agentic/tool-calling wrapper - unpredictable for a plain
# completions call, since it may invoke tools on its own), canopylabs/
# orpheus-v1-english and orpheus-arabic-saudi (text-to-speech),
# whisper-large-v3 and whisper-large-v3-turbo (speech-to-text),
# qwen/qwen3.6-27b and qwen3.8-27b, allam-2-7b (Arabic-focused),
# openai/gpt-oss-120b, openai/gpt-oss-20b, and openai/gpt-oss-safeguard-20b.
# Of the plain instruction-following chat models, gpt-oss-20b is the
# smallest/fastest (this failover tier only gets used after Ollama AND
# Gemini have both already failed, so low latency matters) - a real test
# call returned in ~39ms server-side (via the response's
# usage.completion_time) and 3/3 test prompts succeeded.
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_TIMEOUT_SECONDS = 30.0
# Same models-list endpoint used to pick GROQ_MODEL above, reused as the
# /health reachability check, consistent with how Ollama/Gemini are checked.
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

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

# One Backend instance per router-selectable local model, keyed by the exact
# model name route() returns, so chat() can do ollama_backends[model_used]
# without an if/elif per model. Both share OLLAMA_URL/OLLAMA_TIMEOUT_SECONDS
# as constructor args (rather than backends.py hardcoding them) so tests can
# monkeypatch a single instance's base_url to simulate a local failure
# without touching real Ollama or the other model's backend.
ollama_backends = {
    "qwen2.5:1.5b": OllamaBackend(
        model="qwen2.5:1.5b", base_url=OLLAMA_URL, timeout=OLLAMA_TIMEOUT_SECONDS
    ),
    "qwen2.5:3b": OllamaBackend(
        model="qwen2.5:3b", base_url=OLLAMA_URL, timeout=OLLAMA_TIMEOUT_SECONDS
    ),
}
gemini_backend = GeminiBackend(
    model=GEMINI_MODEL, api_key=GEMINI_API_KEY, timeout=GEMINI_TIMEOUT_SECONDS
)
groq_backend = GroqBackend(model=GROQ_MODEL, api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)


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
    served_by: str = "ollama"  # "ollama" | "gemini" | "groq" | "cache"
    request_id: str


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

    if not GROQ_API_KEY:
        groq_status = "unconfigured"
    else:
        groq_status = "down"
        try:
            async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT_SECONDS) as client:
                resp = await client.get(
                    GROQ_MODELS_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}
                )
                if resp.status_code == 200:
                    groq_status = "up"
        except Exception:
            groq_status = "down"

    return {
        "gateway": "up",
        "ollama": ollama_status,
        "gemini": gemini_status,
        "groq": groq_status,
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
        # failed_over is set explicitly here, not left to the ChatResponse
        # default: a cache hit is the healthy fast path, not a failover, and
        # that shouldn't be an implicit fact someone could break by changing
        # a default elsewhere later. failed_over is only ever True when
        # served_by is "gemini" or "groq" specifically - see the assignment
        # below the backend_chain loop for the miss path's equivalent.
        log_request(
            req.prompt,
            "cache",
            True,
            latency_ms,
            request_id=request_id,
            served_by="cache",
            failed_over=False,
        )
        return ChatResponse(
            response=cached_entry.response,
            model_used="cache",
            cache_hit=True,
            latency_ms=latency_ms,
            served_by="cache",
            failed_over=False,
            request_id=request_id,
        )

    routed_model = route(req.prompt)
    # Three-tier failover chain, tried in order, stopping at the first
    # success. This is still NOT a circuit breaker - no failure counter, no
    # open/half-open state, no cooldown. Every request independently starts
    # at Ollama; only that one request's own failures decide how far down
    # the chain it falls. "Fails" means any exception out of generate() -
    # a raised error (connection refused, HTTP 4xx/5xx) or a timeout past
    # that backend's own configured timeout (OLLAMA_TIMEOUT_SECONDS is also
    # the original failover trigger threshold from Phase 0).
    backend_chain = [
        ("ollama", ollama_backends[routed_model]),
        ("gemini", gemini_backend),
        ("groq", groq_backend),
    ]

    served_by = None
    model_used = None
    response_text = None
    load_duration_ms = None
    eval_count = None
    eval_duration_ms = None
    failures = []

    for tier_name, backend in backend_chain:
        try:
            backend_response = await backend.generate(req.prompt)
        except Exception as exc:
            failures.append(f"{tier_name} ({type(exc).__name__})")
            print(f"[{request_id}] {tier_name} backend failed: {type(exc).__name__}: {exc}")
            continue

        served_by = tier_name
        model_used = backend_response.model_name
        response_text = backend_response.text
        load_duration_ms = backend_response.load_duration_ms
        eval_count = backend_response.eval_count
        eval_duration_ms = backend_response.eval_duration_ms
        break

    if served_by is None:
        # All three backends failed for this request - a clear 502 instead
        # of an unhandled exception turning into a generic 500. Nothing is
        # cached or logged to gateway.db in this case, same as a validation
        # failure above: there's no successful response to record.
        raise HTTPException(
            status_code=502,
            detail=f"All backends failed for this request: {'; '.join(failures)}",
        )

    # Explicit membership check, not `served_by != "ollama"`: failed_over
    # means "an actual backend failover occurred", true only for gemini/groq.
    # served_by can't be "cache" here (that path already returned above),
    # but spelling it out this way doesn't rely on that being true forever.
    failed_over = served_by in ("gemini", "groq")

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
        served_by=served_by,
    )

    return ChatResponse(
        response=response_text,
        model_used=model_used,
        cache_hit=False,
        latency_ms=latency_ms,
        served_by=served_by,
        failed_over=failed_over,
        request_id=request_id,
    )
