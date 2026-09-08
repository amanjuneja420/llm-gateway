"""
Backend interface.
====================
A common shape every model backend implements, so chat() in main.py can try
them in a fixed order without caring which one is actually serving a given
request beyond the .name it reports back afterwards.

This module intentionally holds no failover logic - it just defines what a
backend IS (an object with a name and a generate() coroutine) and provides
the concrete Ollama and Gemini implementations. The failover chain itself
(try Ollama, then Gemini, ...) lives in main.py's chat(), since that's
where request-level concerns like caching and logging already are.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


@dataclass
class BackendResponse:
    """What every backend returns, regardless of what it actually is
    underneath. load_duration_ms/eval_count/eval_duration_ms are Ollama-
    specific timing details (see main.py's docstring on why we log them);
    they're None for backends that don't report anything equivalent."""

    text: str
    model_name: str
    load_duration_ms: float | None = None
    eval_count: int | None = None
    eval_duration_ms: float | None = None


class Backend(ABC):
    """One model backend. `name` is the short id logged in the DB's
    served_by column ("ollama" / "gemini" / "groq") - not the specific
    model name, which is on BackendResponse.model_name instead, since one
    backend (Ollama) serves two different models depending on the router."""

    name: str

    @abstractmethod
    async def generate(self, prompt: str) -> BackendResponse:
        """Generate a response to `prompt`. Raises on any failure (timeout,
        connection error, non-2xx HTTP status, missing API key, ...) -
        callers decide what "failure" means for their purposes and how to
        react to it (see main.py's chat()), this method never swallows
        anything itself."""
        raise NotImplementedError


class OllamaBackend(Backend):
    """One Ollama model. Each router-selectable model (qwen2.5:1.5b,
    qwen2.5:3b) gets its own instance, constructed with that model name
    fixed in - the router picks which *instance* to call, not a parameter
    passed at call time, so `generate(prompt)` has the exact same shape
    across every backend."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/api/generate",
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    async def generate(self, prompt: str) -> BackendResponse:
        # keep_alive="30m" is passed explicitly on every request so the
        # model stays resident in Ollama between requests for the duration
        # of a benchmark run, regardless of Ollama's own default keep-alive
        # setting (Ollama runs as an already-started background service
        # here, so an OLLAMA_KEEP_ALIVE env var set in our own process's
        # shell would not reach it - the per-request keep_alive field is
        # the mechanism that actually takes effect).
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.base_url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Ollama reports these in nanoseconds (eval_count is a plain token
        # count, not a duration) - convert to ms so they're directly
        # comparable to the gateway's own latency_ms.
        return BackendResponse(
            text=data["response"],
            model_name=self.model,
            load_duration_ms=data.get("load_duration", 0) / 1e6,
            eval_count=data.get("eval_count", 0),
            eval_duration_ms=data.get("eval_duration", 0) / 1e6,
        )


class GeminiBackend(Backend):
    """Gemini via its REST API directly (no SDK - one more dependency
    isn't worth it when httpx is already used everywhere else)."""

    name = "gemini"

    def __init__(self, model: str, api_key: str | None, timeout: float = 60.0):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async def generate(self, prompt: str) -> BackendResponse:
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set (add it to .env - see .env.example) - "
                "cannot use the Gemini backend without it."
            )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        # The API key is sent as the x-goog-api-key header, never as a URL
        # query parameter. This is deliberate: httpx exceptions (and any
        # traceback that ends up in a log) include the request URL in
        # their message, so a key-in-URL would leak the secret into any
        # error output. A header never appears in that message.
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return BackendResponse(text=text, model_name=self.model)


class GroqBackend(Backend):
    """
    Groq via its OpenAI-compatible chat completions endpoint - third tier
    of the failover chain, tried only after both Ollama and Gemini have
    failed for a given request.

    GROQ_MODEL (see main.py) was picked by querying Groq's live
    /openai/v1/models list rather than assuming a name, same lesson as the
    gemini-1.5-flash/gemini-3.1-flash situation - see the comment next to
    GROQ_MODEL in main.py for the full reasoning and what else was in the
    list.
    """

    name = "groq"

    def __init__(
        self,
        model: str,
        api_key: str | None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    async def generate(self, prompt: str) -> BackendResponse:
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set (add it to .env - see .env.example) - "
                "cannot use the Groq backend without it."
            )
        # Bearer token in the Authorization header, not the URL - same
        # reasoning as GeminiBackend's x-goog-api-key header above.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        # Groq reports completion_time in seconds (not Ollama-style
        # nanoseconds) - convert to ms for the same units as the other
        # backends' eval_duration_ms.
        completion_time = usage.get("completion_time")
        return BackendResponse(
            text=text,
            model_name=self.model,
            eval_count=usage.get("completion_tokens"),
            eval_duration_ms=completion_time * 1000 if completion_time is not None else None,
        )
