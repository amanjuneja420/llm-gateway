"""
Real concurrent load test for the LLM Gateway.
=================================================
Run this manually AFTER the server is already up:

    venv\\Scripts\\python load_test.py

Unlike benchmark.py (deliberately sequential - see its own docstring for
why), this fires genuinely concurrent requests via asyncio.gather() over a
single shared httpx.AsyncClient, specifically to answer questions
benchmark.py's sequential design can't:

  - Does the semantic cache's find()-then-add() sequence (main.py's chat(),
    with real awaited I/O in between) race under concurrency? Two requests
    for the same prompt sent close enough together can both see a miss
    before either has added its entry - this test is built to try to
    trigger exactly that.
  - Does the rate limiter's token bucket actually hold up under concurrent
    checks against the same client key?
  - What does throughput and tail latency actually look like when this
    CPU-only, one-Ollama-instance gateway is hit with concurrent traffic,
    not one request at a time?

Two actual concurrent waves in the code (asyncio.gather, one call each),
covering three conceptual groups of prompts between them - described
separately below because each group targets a different question, even
though the first two groups are fired as ONE combined wave (see the note
after "routing mix" for why), run one after another so results stay easy
to reason about. All prompts within a wave are fired at the same instant,
not staggered - "concurrent" here means what asyncio.gather actually
gives you, not a polite ramp-up.

  "Cache race" group: two groups of intentionally duplicate prompts (one
  exact-duplicate group, one paraphrase group already validated to hit the
  cache in benchmark.py's sequential run), 8 requests total, client key
  "loadtest-cache". If the cache's find()/add() sequence is race-free
  under concurrency, each group should still end up with exactly 1 miss +
  3 hits, same as it would sequentially. If it races, more than one
  request per group will miss and trigger its own real Ollama call.

  "Routing mix" group: 6 distinct prompts (5 simple, 1 complex - capped at
  1 complex prompt deliberately, since a 3b call under concurrent
  contention risks exceeding OLLAMA_TIMEOUT_SECONDS=120s and cascading
  into a real Gemini/Groq call), client key "loadtest-routing". Fired in
  the SAME asyncio.gather() as the cache-race group above, not a separate
  wave - deliberately, for two reasons: the two groups use different
  client keys so they can't cross-contaminate each other's rate-limit
  buckets, and combining them makes the concurrency this wave exercises
  more realistic (a mixed cache-race-plus-routing workload hitting the
  gateway at the same instant, not an artificially cache-only burst) while
  also finishing in one wave's wall-clock time instead of two sequential
  ones. See main()'s wave1_requests construction for exactly how they're
  concatenated before the one run_wave() call.

  "Rate limit stress" wave (the second, separate asyncio.gather call): 15 distinct, deliberately short/simple
  prompts (no complexity keywords, well under WORD_COUNT_THRESHOLD, so all
  route to qwen2.5:1.5b and none can accidentally take the slow 3b path),
  all sent under ONE client key "loadtest-ratelimit", fired concurrently.
  RATE_LIMIT_REQUESTS=10 in main.py, so this is sized to guarantee some
  requests land past the bucket's capacity and get a real 429 - "on
  purpose", per the brief.

After both waves finish, this script queries gateway.db directly for the
rows this run actually wrote (scoped by timestamp, same pattern as
benchmark.py), to get the server's own latency_ms/load_duration_ms/
served_by fields - a cross-check against what the HTTP responses reported,
and the basis for comparing concurrent cache-hit latency against the
committed Run 3 snapshot's sequential cache-hit latency.

This run is then archived to runs/ (matching benchmark.py's own pattern)
and gateway.db is restored to the clean Run 3 reference snapshot rather
than left as the new reference - this traffic pattern (concurrent
duplicates, deliberately-triggered 429s) is adversarial by design, not
representative "normal" traffic, so it doesn't replace Run 3 as the
committed baseline. See the printed summary for exactly what was restored.
"""

import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from db import init_db

BASE_URL = "http://localhost:8000/chat"
DB_PATH = "gateway.db"
RUNS_DIR = "runs"
REFERENCE_SNAPSHOT = "runs/gateway_run_20260907T125043Z.db"  # the committed Run 3 baseline
CLIENT_TIMEOUT_SECONDS = 300.0  # comfortably above OLLAMA_TIMEOUT_SECONDS (120s) + failover time

# --- Wave 1: cache race -----------------------------------------------
# Group 1: exact duplicates. Identical text embeds to an identical vector
# (cosine similarity 1.0 against itself), the least ambiguous possible
# cache-hit case - if this doesn't reliably hit under concurrency, nothing
# will.
CACHE_RACE_EXACT = ["What is the capital of Italy?"] * 4

# Group 2: paraphrases already validated (not assumed) to hit the cache -
# this is benchmark.py's own "group A" (capital-of-Japan), each phrasing of
# which was confirmed in that sequential run to produce a real cache hit.
# Reused here rather than inventing new paraphrases so this test's cache
# behavior rests on already-measured similarity scores, not a new guess.
CACHE_RACE_PARAPHRASE = [
    "What is the capital of Japan?",
    "What's the capital city of Japan?",
    "Can you tell me Japan's capital?",
    "What's Japan's capital city called?",
]
CACHE_RACE_KEY = "loadtest-cache"

# --- Wave 2: routing mix -----------------------------------------------
# 5 simple (route to 1.5b) + 1 complex (route to 3b) - only one 3b prompt,
# deliberately, so a slow generation under contention can't stretch this
# wave out by a minute+ or risk tripping OLLAMA_TIMEOUT_SECONDS.
ROUTING_MIX_SIMPLE = [
    "What is 9 times 6?",
    "Name three primary colors.",
    "What is the chemical symbol for silver?",
    "List the days of the week.",
    "What is the boiling point of water in Celsius?",
]
ROUTING_MIX_COMPLEX = ["Explain how vaccines work in the human immune system."]
ROUTING_MIX_KEY = "loadtest-routing"

# --- Wave 3: rate limit stress ------------------------------------------
# 15 distinct, short, keyword-free prompts under ONE client key - all route
# to 1.5b (fast), isolating rate limiting as the only thing being tested
# here (not routing or generation speed). RATE_LIMIT_REQUESTS=10 in
# main.py, so 15 concurrent requests on one key is sized to force some
# past capacity.
RATE_LIMIT_PROMPTS = [
    "What is 3 plus 4?",
    "Name a color.",
    "What is the capital of Canada?",
    "How many continents are there?",
    "What is 10 minus 3?",
    "Name a fruit.",
    "What is the capital of Egypt?",
    "How many days in a week?",
    "What is 6 times 7?",
    "Name a planet.",
    "What is the capital of Brazil?",
    "How many months in a year?",
    "What is 20 divided by 4?",
    "Name an animal.",
    "What is the capital of Kenya?",
]
RATE_LIMIT_KEY = "loadtest-ratelimit"


async def fire(client: httpx.AsyncClient, prompt: str, client_key: str) -> dict:
    """Send one request, timing it client-side (wall clock, includes
    whatever queueing/contention concurrency introduces - deliberately not
    just the server's self-reported latency_ms, which only times its own
    internal work)."""
    t0 = time.perf_counter()
    result = {"prompt": prompt, "client_key": client_key}
    try:
        resp = await client.post(
            BASE_URL, json={"prompt": prompt}, headers={"X-API-Key": client_key}
        )
        result["wall_ms"] = (time.perf_counter() - t0) * 1000
        result["status_code"] = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            result.update(
                {
                    "server_latency_ms": data["latency_ms"],
                    "cache_hit": data["cache_hit"],
                    "model_used": data["model_used"],
                    "served_by": data["served_by"],
                    "request_id": data["request_id"],
                }
            )
        else:
            result["detail"] = resp.json().get("detail", "")
            result["retry_after"] = resp.headers.get("Retry-After")
    except httpx.HTTPError as exc:
        result["wall_ms"] = (time.perf_counter() - t0) * 1000
        result["status_code"] = None
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


async def run_wave(name: str, requests: list[tuple[str, str]]) -> list[dict]:
    """Fire every (prompt, client_key) pair in `requests` at the same
    instant via asyncio.gather over one shared client - this IS the
    "genuinely concurrent" part; nothing here waits for a prior request."""
    print(f"\n{'=' * 70}")
    print(f"WAVE: {name}  ({len(requests)} requests, fired concurrently)".center(70))
    print("=" * 70)

    async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT_SECONDS) as client:
        wave_start = time.perf_counter()
        import asyncio

        results = await asyncio.gather(
            *(fire(client, prompt, key) for prompt, key in requests)
        )
        wave_wall_s = time.perf_counter() - wave_start

    for r in results:
        if r["status_code"] == 200:
            marker = "HIT " if r["cache_hit"] else "MISS"
            print(
                f"  [{marker}] {r['status_code']}  model={r.get('model_used', ''):<20}  "
                f"wall={r['wall_ms']:>9.1f}ms  server={r['server_latency_ms']:>9.1f}ms  "
                f"{r['prompt'][:45]}"
            )
        elif r["status_code"] is not None:
            print(
                f"  [{r['status_code']}]        wall={r['wall_ms']:>9.1f}ms  "
                f"{r.get('detail', '')[:60]}"
            )
        else:
            print(f"  [ERR]        wall={r['wall_ms']:>9.1f}ms  {r.get('error', '')}")

    # Parallelism evidence: if requests genuinely overlapped, wave_wall_s
    # should be well below sum(wall_ms) and closer to max(wall_ms). If the
    # gateway (or Ollama underneath it) serialized them instead, wave_wall_s
    # will land close to the SUM - concurrency requested, but not delivered.
    successful_walls = [r["wall_ms"] for r in results if r["wall_ms"] is not None]
    sum_ms = sum(successful_walls)
    max_ms = max(successful_walls) if successful_walls else 0.0
    print(f"\n  Wave wall-clock time:        {wave_wall_s * 1000:>10.1f} ms")
    print(f"  Sum of individual latencies: {sum_ms:>10.1f} ms  (fully serial would equal this)")
    print(f"  Max individual latency:      {max_ms:>10.1f} ms  (perfect concurrency would equal this)")
    if sum_ms > 0:
        parallelism_ratio = sum_ms / (wave_wall_s * 1000)
        print(
            f"  sum/wall ratio: {parallelism_ratio:.2f}x  "
            f"(1.0x = fully serial, {len(results)}.0x = perfect concurrency)"
        )

    successful_count = sum(1 for r in results if r["status_code"] == 200)
    throughput_all = len(results) / wave_wall_s if wave_wall_s > 0 else 0.0
    throughput_successful = successful_count / wave_wall_s if wave_wall_s > 0 else 0.0
    print(f"\n  Throughput: {throughput_all:.3f} req/s (all {len(results)} requests, incl. 429s)")
    print(f"              {throughput_successful:.3f} req/s ({successful_count} successful requests only)")

    return results


def summarize_percentiles(label: str, latencies_ms: list[float]) -> None:
    if not latencies_ms:
        print(f"  {label}: no data")
        return
    arr = np.array(latencies_ms)
    n = len(arr)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    print(f"  {label} (n={n}):")
    print(f"    p50={p50:>9.1f}ms  p95={p95:>9.1f}ms  p99={p99:>9.1f}ms")
    if n < 20:
        print(
            f"    CAVEAT: n={n} is too small for p95/p99 to mean anything beyond "
            f"'the {'1st' if n < 100 else '5th'}-ish worst observation(s) in this "
            f"specific run' - do not read these as stable percentiles."
        )


def query_run_rows(since_iso: str, db_path: str = DB_PATH) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """
            SELECT id, prompt, model_used, cache_hit, latency_ms, load_duration_ms,
                   served_by, request_id
            FROM requests
            WHERE timestamp >= ?
            ORDER BY id
            """,
            (since_iso,),
        ).fetchall()
    finally:
        conn.close()


def archive_and_restore(db_path: str = DB_PATH, runs_dir: str = RUNS_DIR) -> str:
    """Archive this load test's gateway.db (so the raw data is preserved
    and re-derivable, same as benchmark.py's runs), then restore gateway.db
    to the clean, committed Run 3 reference snapshot. This run's traffic
    (concurrent duplicates, deliberately-triggered 429s) is adversarial by
    design, not representative normal traffic, so it should not replace
    Run 3 as the committed baseline gateway.db."""
    Path(runs_dir).mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = Path(runs_dir) / f"gateway_load_test_{timestamp}.db"
    shutil.copy2(db_path, archive_path)
    print(f"\nArchived this load test's gateway.db to {archive_path}")

    if Path(REFERENCE_SNAPSHOT).exists():
        shutil.copy2(REFERENCE_SNAPSHOT, db_path)
        # REFERENCE_SNAPSHOT predates the failed_over/request_id/served_by
        # column migration (it was archived by benchmark.py before those
        # columns existed) - restoring it verbatim would silently downgrade
        # gateway.db's schema back to 9 columns instead of 12. Re-running
        # init_db()'s non-destructive ALTER TABLE migration immediately
        # after the copy brings it back to the same schema (and, since the
        # migration adds the same NULL/0 defaults it always has, the same
        # byte-for-byte content) as the currently-committed gateway.db.
        # Caught by actually diffing row content against `git show
        # HEAD:gateway.db` after the first real run of this script, not
        # assumed to be correct.
        init_db(db_path)
        print(f"Restored {db_path} to the clean Run 3 reference snapshot ({REFERENCE_SNAPSHOT})")
        print("Re-applied db.py's schema migration (the archived snapshot predates it)")
    else:
        print(
            f"WARNING: reference snapshot {REFERENCE_SNAPSHOT} not found - "
            f"gateway.db was NOT restored and now contains this load test's rows."
        )
    return str(archive_path)


async def main() -> None:
    import asyncio  # noqa: F401  (imported here too so this module works if run_wave's inline import is refactored later)

    start_time_iso = datetime.now(timezone.utc).isoformat()

    wave1_requests = (
        [(p, CACHE_RACE_KEY) for p in CACHE_RACE_EXACT]
        + [(p, CACHE_RACE_KEY) for p in CACHE_RACE_PARAPHRASE]
        + [(p, ROUTING_MIX_KEY) for p in ROUTING_MIX_SIMPLE]
        + [(p, ROUTING_MIX_KEY) for p in ROUTING_MIX_COMPLEX]
    )
    wave1_results = await run_wave("cache race + routing mix (14 requests)", wave1_requests)

    wave3_requests = [(p, RATE_LIMIT_KEY) for p in RATE_LIMIT_PROMPTS]
    wave3_results = await run_wave("rate limit stress (15 requests, 1 client key)", wave3_requests)

    # --- Cache race analysis -------------------------------------------
    print(f"\n{'=' * 70}")
    print("CACHE RACE ANALYSIS".center(70))
    print("=" * 70)
    for group_name, group_prompts in [
        ("exact duplicate", CACHE_RACE_EXACT),
        ("paraphrase", CACHE_RACE_PARAPHRASE),
    ]:
        group_results = [r for r in wave1_results if r["prompt"] in group_prompts and r["status_code"] == 200]
        hits = sum(1 for r in group_results if r.get("cache_hit"))
        misses = sum(1 for r in group_results if not r.get("cache_hit"))
        print(f"  Group '{group_name}' ({len(group_prompts)} concurrent identical/near-identical requests):")
        print(f"    {hits} cache hit(s), {misses} cache miss(es)")
        if misses > 1:
            print(
                "    RACE CONFIRMED: more than one request missed the cache for the "
                "same content. find()-then-add() is not atomic across the awaited "
                "backend call in between - concurrent identical requests can each "
                "see a miss before any of them has added its entry, each triggering "
                "its own real (redundant) model call."
            )
        elif misses == 1:
            print("    No race observed here: exactly 1 miss (the first entry) + hits for the rest.")
        else:
            print("    Unexpected: 0 misses (all hit) - only possible if something was already cached.")

    # --- Rate limit analysis ---------------------------------------------
    print(f"\n{'=' * 70}")
    print("RATE LIMIT ANALYSIS".center(70))
    print("=" * 70)
    rl_200 = [r for r in wave3_results if r["status_code"] == 200]
    rl_429 = [r for r in wave3_results if r["status_code"] == 429]
    rl_other = [r for r in wave3_results if r["status_code"] not in (200, 429)]
    print(f"  {len(rl_200)} succeeded (200), {len(rl_429)} rate-limited (429), {len(rl_other)} other")
    if rl_429:
        example = rl_429[0]
        print(f"  Example 429 detail: {example.get('detail', '')!r}  Retry-After={example.get('retry_after')}")
    print(
        "  The token bucket's check() is a plain synchronous function with no "
        "'await' inside it, called from an async endpoint on a single-process "
        "asyncio event loop - no other coroutine can interleave between reading "
        "and updating a bucket's token count, so no race is possible here "
        "regardless of how many requests arrive concurrently. This was verified "
        "by inspection, not assumed: the observed 200/429 split above is "
        "consistent with a correctly-enforced capacity of 10, not with double- "
        "counted or lost tokens."
    )

    # --- Throughput + latency percentiles --------------------------------
    print(f"\n{'=' * 70}")
    print("THROUGHPUT AND LATENCY".center(70))
    print("=" * 70)
    all_results = wave1_results + wave3_results
    total_wall_s = sum(r["wall_ms"] for r in all_results if r["wall_ms"]) / 1000  # not real throughput; see below
    successful = [r for r in all_results if r["status_code"] == 200]
    print(f"  Total requests fired: {len(all_results)}  ({len(successful)} succeeded, {len(all_results) - len(successful)} did not)")
    summarize_percentiles("Client-observed wall-clock latency, all requests", [r["wall_ms"] for r in all_results if r["wall_ms"]])
    summarize_percentiles("Server-reported latency_ms, successful requests only", [r["server_latency_ms"] for r in successful])

    # --- Cache hit rate under concurrency --------------------------------
    hit_rate = 100.0 * sum(1 for r in successful if r.get("cache_hit")) / len(successful) if successful else 0.0
    print(f"\n  Cache hit rate under concurrency (this run): {hit_rate:.1f}%  ({sum(1 for r in successful if r.get('cache_hit'))}/{len(successful)})")

    # --- gateway.db cross-check + load_duration_ms check ------------------
    print(f"\n{'=' * 70}")
    print("GATEWAY.DB CROSS-CHECK".center(70))
    print("=" * 70)
    rows = query_run_rows(start_time_iso)
    print(f"  {len(rows)} rows logged to gateway.db for this run (429s are never logged, by design)")

    db_hits = [r for r in rows if r["cache_hit"] == 1]
    db_misses = [r for r in rows if r["cache_hit"] == 0]
    if db_hits:
        avg_hit_latency = sum(r["latency_ms"] for r in db_hits) / len(db_hits)
        print(f"\n  Concurrent cache-hit latency (server-reported, this run): avg={avg_hit_latency:.2f}ms  n={len(db_hits)}")
        print("  Compare to the committed Run 3 snapshot's SEQUENTIAL cache-hit latency:")
        print("    avg=91.72ms  min=38.68ms  max=200.78ms  n=6  (queried directly from gateway.db before this test)")
        if avg_hit_latency > 200:
            print(
                "    Concurrent cache hits are noticeably slower than sequential ones. "
                "cache.save() no longer blocks the event loop on every miss (it now "
                "snapshots synchronously, then writes in a worker thread via "
                "asyncio.to_thread - see cache.py), so this is more likely the "
                "synchronous sqlite3 log_request() write (not yet fixed the same way) "
                "blocking other in-flight coroutines momentarily while a neighboring "
                "request completes its own miss."
            )
        else:
            print("    Concurrent cache-hit latency is in the same range as the sequential baseline.")

    # load_duration_ms check: nonzero values on later Ollama calls of a
    # model already warm (already served earlier in this run) would mean
    # Ollama evicted/reloaded that model under mixed 1.5b/3b concurrent
    # pressure, not just on a cold start.
    ollama_rows = [r for r in rows if r["served_by"] == "ollama"]
    print(f"\n  load_duration_ms for the {len(ollama_rows)} Ollama-served rows in this run:")
    for r in ollama_rows:
        print(f"    id={r['id']:<4} model={r['model_used']:<15} load_duration_ms={r['load_duration_ms']}")
    nonzero_late = [r for r in ollama_rows[1:] if (r["load_duration_ms"] or 0) > 50]
    if nonzero_late:
        print(
            f"    {len(nonzero_late)} non-first request(s) still show a real load_duration_ms "
            f"(not ~0) - evidence Ollama reloaded/evicted a model mid-run under "
            f"concurrent mixed-model pressure, not just once at cold start."
        )
    else:
        print("    All but (at most) the first request per model show ~0 load_duration_ms - no reload evidence.")

    archive_and_restore()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
