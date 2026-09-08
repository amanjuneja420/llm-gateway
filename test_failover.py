"""
Standalone failover test.
===========================
Run manually, after activating the venv:

    venv\\Scripts\\python test_failover.py

Imports main.py directly (not via HTTP) so it can monkeypatch backend
instances' base_url/url for exactly one call each, simulating local/remote
failures without touching the real services, restarting anything, or
needing a live uvicorn server. Each backend is its own object with its
target URL fixed in at construction (see backends.py), so patching a
specific instance's url attribute affects only that instance - the other
backends, and the real services, are untouched.

Three checks, matching the three-tier failover chain (Ollama -> Gemini ->
Groq, stopping at the first success):

  1. A normal prompt succeeds through Ollama, completely unaffected by the
     failover code path (failed_over=False, served_by="ollama").
  2. With the routed Ollama backend's base_url pointed at a port nothing
     listens on, the request still succeeds - via Gemini (served_by=
     "gemini", failed_over=True) - and that's what's actually logged to
     gateway.db, not just returned in the response.
  3. With BOTH the routed Ollama backend's base_url AND Gemini's url
     pointed at unreachable addresses, the request still succeeds - via
     Groq (served_by="groq") - the third tier only gets exercised once the
     first two have both failed for this one request.

Tests 2 and 3 require working GEMINI_API_KEY / GROQ_API_KEY in .env (see
.env.example). If either is missing or invalid, this prints that failure
clearly instead of a raw crash - it does NOT fake a pass.
"""

import asyncio
import sqlite3

import main
from db import DB_PATH, init_db

UNREACHABLE_URL = "http://localhost:1/nothing-listens-here"

# init_db()'s schema migration normally runs inside main.py's FastAPI
# lifespan, which only fires under uvicorn's actual ASGI startup - importing
# main.py directly (as this script does) never triggers it. Call it here so
# gateway.db has whatever columns this version of the code expects (e.g.
# served_by), regardless of whether a real server has run since it was
# added.
init_db()


async def test_normal_request() -> bool:
    print("=== Test 1: normal request through Ollama (should be unaffected) ===")
    try:
        req = main.ChatRequest(prompt="What is the capital of Italy?")
        result = await main.chat(req)
    except Exception as exc:
        print(f"FAIL - unexpected exception: {type(exc).__name__}: {exc}")
        return False

    if result.failed_over or result.served_by != "ollama":
        print(
            f"FAIL - expected served_by='ollama', failed_over=False; "
            f"got served_by={result.served_by!r}, failed_over={result.failed_over}"
        )
        return False
    if result.model_used not in ("qwen2.5:1.5b", "qwen2.5:3b", "cache"):
        print(f"FAIL - unexpected model_used={result.model_used!r}")
        return False

    print(
        f"PASS - model_used={result.model_used}  cache_hit={result.cache_hit}  "
        f"served_by={result.served_by}  failed_over={result.failed_over}  "
        f"latency_ms={result.latency_ms:.1f}"
    )
    return True


async def test_cache_hit_not_failed_over() -> bool:
    print("\n=== Test: cache hit reports failed_over=False (it's the healthy fast path) ===")
    prompt = "What is the capital of Portugal, for the cache-hit test?"
    paraphrase = "What's Portugal's capital city, for the cache-hit test?"
    try:
        await main.chat(main.ChatRequest(prompt=prompt))  # seed the cache
        result = await main.chat(main.ChatRequest(prompt=paraphrase))  # should hit
    except Exception as exc:
        print(f"FAIL - unexpected exception: {type(exc).__name__}: {exc}")
        return False

    if not result.cache_hit or result.served_by != "cache":
        print(
            f"FAIL - expected a cache hit; got cache_hit={result.cache_hit}, "
            f"served_by={result.served_by!r}"
        )
        return False
    if result.failed_over:
        print("FAIL - cache hit reported failed_over=True (a cache hit is not a failover)")
        return False

    print(
        f"PASS - cache_hit={result.cache_hit}  served_by={result.served_by}  "
        f"failed_over={result.failed_over}  latency_ms={result.latency_ms:.1f}"
    )
    return True


async def test_gemini_failover() -> bool:
    print("\n=== Test 2: Ollama fails -> Gemini succeeds ===")
    # This prompt is short with no complexity keywords, so router.route()
    # sends it to qwen2.5:1.5b - patch that specific backend instance only,
    # leaving the 3b one (and the real Ollama service) untouched.
    ollama = main.ollama_backends["qwen2.5:1.5b"]
    original_url = ollama.base_url
    ollama.base_url = UNREACHABLE_URL
    try:
        req = main.ChatRequest(prompt="Name one moon of Jupiter, for the failover test.")
        result = await main.chat(req)
    except Exception as exc:
        print(f"FAIL - failover did not complete: {type(exc).__name__}: {exc}")
        print(
            "       (if this is 'GEMINI_API_KEY is not set', add a real key to "
            ".env - see .env.example - and re-run)"
        )
        return False
    finally:
        ollama.base_url = original_url  # restore immediately, whatever happened

    if not result.failed_over or result.served_by != "gemini":
        print(
            f"FAIL - expected served_by='gemini', failed_over=True; "
            f"got served_by={result.served_by!r}, failed_over={result.failed_over}"
        )
        return False
    if result.model_used != main.GEMINI_MODEL:
        print(f"FAIL - expected model_used={main.GEMINI_MODEL!r}, got {result.model_used!r}")
        return False

    print(
        f"PASS - model_used={result.model_used}  served_by={result.served_by}  "
        f"failed_over={result.failed_over}  latency_ms={result.latency_ms:.1f}"
    )
    print(f"Gemini response (truncated): {result.response[:200]!r}")
    return True


async def test_groq_failover() -> bool:
    print("\n=== Test 3: Ollama AND Gemini both fail -> Groq succeeds ===")
    ollama = main.ollama_backends["qwen2.5:1.5b"]
    original_ollama_url = ollama.base_url
    original_gemini_url = main.gemini_backend.url
    ollama.base_url = UNREACHABLE_URL
    main.gemini_backend.url = UNREACHABLE_URL
    try:
        req = main.ChatRequest(prompt="Name the largest planet, for the double-failover test.")
        result = await main.chat(req)
    except Exception as exc:
        print(f"FAIL - request did not complete: {type(exc).__name__}: {exc}")
        print(
            "       (if this is 'GROQ_API_KEY is not set', add a real key to "
            ".env - see .env.example - and re-run)"
        )
        return False
    finally:
        ollama.base_url = original_ollama_url
        main.gemini_backend.url = original_gemini_url

    if not result.failed_over or result.served_by != "groq":
        print(
            f"FAIL - expected served_by='groq', failed_over=True; "
            f"got served_by={result.served_by!r}, failed_over={result.failed_over}"
        )
        return False
    if result.model_used != main.GROQ_MODEL:
        print(f"FAIL - expected model_used={main.GROQ_MODEL!r}, got {result.model_used!r}")
        return False

    print(
        f"PASS - model_used={result.model_used}  served_by={result.served_by}  "
        f"failed_over={result.failed_over}  latency_ms={result.latency_ms:.1f}"
    )
    print(f"Groq response (truncated): {result.response[:200]!r}")
    return True


def check_db_served_by(expected: str, label: str) -> bool:
    print(f"\n=== Verifying served_by={expected!r} was actually logged to gateway.db ({label}) ===")
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT id, prompt, model_used, served_by, failed_over, latency_ms "
            "FROM requests WHERE served_by = ? ORDER BY id DESC LIMIT 1",
            (expected,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        print(f"FAIL - no served_by={expected!r} row found in gateway.db")
        return False

    row_id, prompt, model_used, served_by, failed_over, latency_ms = row
    print(
        f"PASS - row id={row_id}  model_used={model_used}  served_by={served_by}  "
        f"failed_over={failed_over}  latency_ms={latency_ms:.1f}  prompt={prompt!r}"
    )
    return True


async def main_async() -> None:
    results = {}
    results["normal request (ollama)"] = await test_normal_request()
    results["cache hit not failed_over"] = await test_cache_hit_not_failed_over()
    results["gemini failover"] = await test_gemini_failover()
    results["db logging (gemini)"] = (
        check_db_served_by("gemini", "test 2") if results["gemini failover"] else False
    )
    results["groq failover"] = await test_groq_failover()
    results["db logging (groq)"] = (
        check_db_served_by("groq", "test 3") if results["groq failover"] else False
    )

    print("\n" + "=" * 60)
    print("FAILOVER TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL':<6} {name}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main_async())
