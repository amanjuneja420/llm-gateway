"""
Standalone rate-limit test.
==============================
Run manually, after activating the venv:

    venv\\Scripts\\python test_rate_limit.py

Two parts, deliberately separated:

  1. Unit-level: exercises RateLimiter/TokenBucket directly - no model
     calls, no network, fully deterministic. Confirms the token-bucket
     algorithm itself: a burst up to capacity succeeds, one more is
     rejected with a sensible retry_after, and after waiting out the
     window it allows requests again.
  2. Wiring-level: confirms main.chat() actually enforces this, not just
     that the algorithm is correct in isolation. The bucket is drained
     directly (instant, no model calls) so the next real request is
     guaranteed to be the one over the limit - this needs only 2 real
     Ollama calls total, not a rapid burst of them.

Why not just fire a burst of real chat() requests and call it done: real
Ollama latency (1-4+ seconds per call, see the benchmark results in
README.md) is comparable to or larger than a short test window, so a
"burst" of real requests isn't actually instantaneous - the bucket
partially refills between requests, and the timing becomes unreliable to
assert on. Testing the algorithm in isolation first, then only checking
the wiring with real calls, avoids that entirely.

Cache state protection: this script does `import main` directly, which
never triggers FastAPI's `lifespan()` - so `main.cache` starts completely
empty, regardless of what's actually persisted in cache_state.npz/.json
on disk (same class of gap as init_db()'s migration needing an explicit
call in test_failover.py, since lifespan() doesn't fire on a direct
import either). Part 2's two real chat() calls are real cache misses
against that empty in-memory cache, and each one calls cache.add() then
cache.save() - which does a full overwrite of cache_state.npz/.json, not
a merge. Left alone, running this test would silently replace a real
persisted cache with just this test's 1-2 throwaway entries. Fixed the
same way gateway.db's state is protected around load_test.py's
adversarial traffic: back up cache_state.npz/.json before running,
restore them (or remove them if they didn't exist) in a `finally`,
regardless of whether the test passes, fails, or crashes.
"""

import asyncio
import os
import shutil
import time

from fastapi import HTTPException

import main
from cache import CACHE_STATE_PATH
from rate_limiter import RateLimiter

TEST_CAPACITY = 3
TEST_WINDOW_SECONDS = 3.0
TEST_CLIENT_KEY = "test-client-key-for-rate-limit-check"

CACHE_NPZ_PATH = f"{CACHE_STATE_PATH}.npz"
CACHE_JSON_PATH = f"{CACHE_STATE_PATH}.json"
CACHE_BACKUP_SUFFIX = ".rate_limit_test_backup"


def backup_cache_state() -> dict[str, bool]:
    """Back up cache_state.npz/.json (if present) before this test's real
    chat() calls can overwrite them. Returns which paths existed, so
    restore_cache_state() knows whether to restore a backup or delete a
    file this test created from nothing."""
    existed = {}
    for path in (CACHE_NPZ_PATH, CACHE_JSON_PATH):
        existed[path] = os.path.exists(path)
        if existed[path]:
            shutil.copy2(path, path + CACHE_BACKUP_SUFFIX)
    return existed


def restore_cache_state(existed: dict[str, bool]) -> None:
    """Undo whatever this test run did to cache_state.npz/.json: restore
    the backed-up original if one existed before, or remove the file this
    test created if there wasn't one - either way, on-disk cache state
    ends up exactly as it was before this test ran."""
    for path, had_existed in existed.items():
        backup_path = path + CACHE_BACKUP_SUFFIX
        if had_existed:
            shutil.move(backup_path, path)
        elif os.path.exists(path):
            os.remove(path)


def test_token_bucket_unit() -> bool:
    print("=== Part 1: RateLimiter/TokenBucket in isolation (no model calls) ===")
    limiter = RateLimiter(capacity=TEST_CAPACITY, window_seconds=TEST_WINDOW_SECONDS)
    all_passed = True

    for i in range(1, TEST_CAPACITY + 1):
        allowed, _ = limiter.check("unit-test-key")
        if not allowed:
            print(f"  [{i}/{TEST_CAPACITY}] FAIL - expected allowed within capacity, got rejected")
            all_passed = False
        else:
            print(f"  [{i}/{TEST_CAPACITY}] PASS - allowed (within capacity)")

    allowed, retry_after = limiter.check("unit-test-key")
    if allowed:
        print("  FAIL - expected the request over capacity to be rejected, it was allowed")
        all_passed = False
    else:
        print(f"  PASS - rejected over capacity, retry_after={retry_after:.2f}s")

    wait_seconds = TEST_WINDOW_SECONDS + 0.5
    print(f"  Waiting {wait_seconds}s for the window to reset...")
    time.sleep(wait_seconds)

    allowed, _ = limiter.check("unit-test-key")
    if not allowed:
        print("  FAIL - expected a check after the window reset to be allowed again")
        all_passed = False
    else:
        print("  PASS - allowed again after the window reset")

    return all_passed


async def test_chat_endpoint_enforces_limit() -> bool:
    print("\n=== Part 2: confirm main.chat() actually enforces this (2 real model calls) ===")
    original_limiter = main.rate_limiter
    test_limiter = RateLimiter(capacity=TEST_CAPACITY, window_seconds=TEST_WINDOW_SECONDS)
    main.rate_limiter = test_limiter
    all_passed = True

    try:
        # Drain the bucket directly - instant, no model calls - so the very
        # next real chat() request is guaranteed to be the one over the limit.
        for _ in range(TEST_CAPACITY):
            test_limiter.check(TEST_CLIENT_KEY)

        print("Firing 1 real request against an already-drained bucket (should be rejected)...")
        req = main.ChatRequest(prompt="Rate limit wiring test, should be rejected.")
        try:
            result = await main.chat(req, x_api_key=TEST_CLIENT_KEY)
            print(f"  FAIL - expected 429, got a successful response instead: {result}")
            all_passed = False
        except HTTPException as exc:
            if exc.status_code == 429:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                print(f"  PASS - got 429, Retry-After={retry_after!r}, detail={exc.detail!r}")
            else:
                print(f"  FAIL - expected 429, got {exc.status_code}: {exc.detail}")
                all_passed = False

        wait_seconds = TEST_WINDOW_SECONDS + 0.5
        print(f"Waiting {wait_seconds}s for the window to reset...")
        await asyncio.sleep(wait_seconds)

        print("Firing 1 more real request (after reset - should succeed)...")
        req = main.ChatRequest(prompt="Rate limit wiring test, after window reset.")
        try:
            result = await main.chat(req, x_api_key=TEST_CLIENT_KEY)
            print(f"  PASS - succeeded again (model_used={result.model_used})")
        except HTTPException as exc:
            print(f"  FAIL - expected success, got {exc.status_code}: {exc.detail}")
            all_passed = False
    finally:
        main.rate_limiter = original_limiter  # restore the production limiter, whatever happened

    return all_passed


async def main_async() -> None:
    cache_existed = backup_cache_state()
    try:
        unit_passed = test_token_bucket_unit()
        wiring_passed = await test_chat_endpoint_enforces_limit()
    finally:
        restore_cache_state(cache_existed)

    print("\n" + "=" * 60)
    print("RATE LIMIT TEST SUMMARY")
    print("=" * 60)
    print(f"  {'PASS' if unit_passed else 'FAIL':<6} token bucket (unit, isolated)")
    print(f"  {'PASS' if wiring_passed else 'FAIL':<6} chat() endpoint enforcement (real requests)")
    print("=" * 60)
    print(
        "\ncache_state.npz/.json restored to their pre-test state - this test's "
        "2 real chat() calls never persist (see module docstring)."
    )


if __name__ == "__main__":
    asyncio.run(main_async())
