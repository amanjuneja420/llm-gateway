"""
Standalone failover test.
===========================
Run manually, after activating the venv:

    venv\\Scripts\\python test_failover.py

Imports main.py directly (not via HTTP) so it can monkeypatch
main.OLLAMA_URL for exactly one call, simulating a local failure without
touching the real Ollama service, restarting anything, or needing a live
uvicorn server. Two checks:

  (a) a normal prompt still succeeds through Ollama, completely unaffected
      by the failover code path (failed_over=False).
  (b) with OLLAMA_URL pointed at a port nothing listens on, the same
      request shape succeeds anyway - via Gemini failover, with
      failed_over=True in the response - and that failed_over=True is what
      actually gets logged to gateway.db, not just returned in the response.

Test (b) requires a working GEMINI_API_KEY in .env (see .env.example). If
it's missing or invalid, this prints that failure clearly instead of a raw
crash - it does NOT fake a pass.
"""

import asyncio
import sqlite3

import main
from db import DB_PATH


async def test_normal_request() -> bool:
    print("=== Test 1: normal request through Ollama (should be unaffected) ===")
    try:
        req = main.ChatRequest(prompt="What is the capital of Italy?")
        result = await main.chat(req)
    except Exception as exc:
        print(f"FAIL - unexpected exception: {type(exc).__name__}: {exc}")
        return False

    if result.failed_over:
        print("FAIL - failed_over=True on a normal request with Ollama reachable")
        return False
    if result.model_used not in ("qwen2.5:1.5b", "qwen2.5:3b", "cache"):
        print(f"FAIL - unexpected model_used={result.model_used!r}")
        return False

    print(
        f"PASS - model_used={result.model_used}  cache_hit={result.cache_hit}  "
        f"failed_over={result.failed_over}  latency_ms={result.latency_ms:.1f}"
    )
    return True


async def test_failover() -> bool:
    print("\n=== Test 2: simulated local failure -> Gemini failover ===")
    original_url = main.OLLAMA_URL
    main.OLLAMA_URL = "http://localhost:1/api/generate"  # nothing listens here
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
        main.OLLAMA_URL = original_url  # restore immediately, whatever happened

    if not result.failed_over:
        print("FAIL - failed_over=False even though Ollama was unreachable")
        return False
    if result.model_used != main.GEMINI_MODEL:
        print(f"FAIL - expected model_used={main.GEMINI_MODEL!r}, got {result.model_used!r}")
        return False

    print(
        f"PASS - model_used={result.model_used}  failed_over={result.failed_over}  "
        f"latency_ms={result.latency_ms:.1f}"
    )
    print(f"Gemini response (truncated): {result.response[:200]!r}")
    return True


def check_db_logged_failover() -> bool:
    print("\n=== Verifying failed_over=1 was actually logged to gateway.db ===")
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT id, prompt, model_used, failed_over, latency_ms "
            "FROM requests WHERE failed_over = 1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        print("FAIL - no failed_over=1 row found in gateway.db")
        return False

    row_id, prompt, model_used, failed_over, latency_ms = row
    print(
        f"PASS - row id={row_id}  model_used={model_used}  failed_over={failed_over}  "
        f"latency_ms={latency_ms:.1f}  prompt={prompt!r}"
    )
    return True


async def main_async() -> None:
    results = {}
    results["normal request"] = await test_normal_request()
    results["gemini failover"] = await test_failover()
    results["db logging"] = check_db_logged_failover() if results["gemini failover"] else False

    print("\n" + "=" * 60)
    print("FAILOVER TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL':<6} {name}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main_async())
