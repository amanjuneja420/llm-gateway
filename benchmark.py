"""
Benchmark / test script for the LLM Gateway.
==============================================
Run this manually AFTER the server is already up:

    venv\\Scripts\\python benchmark.py

Fires a batch of prompts at the running gateway (mix of short/simple prompts,
longer/complex prompts, and several near-duplicate paraphrase groups spread
out through the batch so real cache hits happen), then queries gateway.db
directly and prints a summary: total requests, cache hit rate, average
latency for cache hits vs misses, and average latency per model.

Requests are sent sequentially (not concurrently) - this is CPU-only
inference on one machine, so concurrent requests would just queue behind
each other on the Ollama side and produce misleading per-request latency
numbers.

After printing the summary, this script also archives the resulting
gateway.db to runs/gateway_run_<timestamp>.db, since the normal workflow is
to wipe gateway.db before the next run. Without this, aggregating multiple
runs later (see aggregate_benchmark_runs.py) means re-typing numbers from
printed logs by hand instead of querying preserved raw data.
"""

import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000/chat"
DB_PATH = "gateway.db"
RUNS_DIR = "runs"

# 30 prompts total. Comments mark the paraphrase groups (near-duplicates of
# an earlier prompt in the list) - these are the ones expected to produce
# cache hits. Groups are deliberately spread out rather than sent back to
# back, closer to how duplicate questions would actually arrive in traffic.
PROMPTS = [
    "What is the capital of Japan?",  # group A (capital-of-Japan) #1
    "Explain how photosynthesis works in plants.",  # group C (photosynthesis) #1
    "What is 15 times 7?",
    "What's the capital city of Japan?",  # group A #2 -> expect cache hit
    "Name three primary colors.",
    "Compare the advantages and disadvantages of remote work versus office work.",  # group D (remote-work) #1
    "What is the largest planet in our solar system?",  # group B (largest-planet) #1
    "Why do seasons change throughout the year?",
    "Can you tell me Japan's capital?",  # group A #3 -> expect cache hit
    "What year did World War II end?",
    "What are the pros and cons of remote work compared to working in an office?",  # group D #2 -> expect cache hit
    "What is the chemical symbol for gold?",
    "Which planet is the biggest in our solar system?",  # group B #2 -> expect cache hit
    "What are the steps to bake a basic loaf of bread from scratch?",
    "List the days of the week.",
    "Can you explain the process of photosynthesis in plants?",  # group C #2 -> expect cache hit
    "Translate 'hello' to Spanish.",
    "Analyze the causes of the fall of the Roman Empire.",
    "Tell me the largest planet in the solar system.",  # group B #3 -> expect cache hit
    "Summarize the plot of Romeo and Juliet in a few sentences.",
    "What is the difference between machine learning and deep learning?",
    "What's Japan's capital city called?",  # group A #4 -> expect cache hit
    "Evaluate the pros and cons of electric vehicles compared to gasoline cars.",
    "How does photosynthesis work?",  # group C #3 -> expect cache hit (shorter phrasing)
    "What is the tallest mountain in the world?",  # group E (tallest-mountain) #1
    (
        "I've been thinking about switching careers into software "
        "engineering, and I want to understand what skills I should focus "
        "on and how long it might realistically take."
    ),
    "What is the boiling point of water in Celsius?",
    "How do seasons change during the year?",  # paraphrase of the "why do seasons change" prompt
    "What's the highest mountain on Earth?",  # group E #2 -> expect cache hit
    "Contrast the benefits of solar power and wind power for home energy use.",
]


def run_benchmark() -> str:
    """Fire every prompt in PROMPTS at the gateway sequentially. Returns the
    UTC timestamp (ISO format) captured just before the first request, used
    to scope the SQLite summary query to only this run's rows."""
    start_time_iso = datetime.now(timezone.utc).isoformat()

    print(f"Firing {len(PROMPTS)} prompts at {BASE_URL} ...\n")
    failures = 0
    with httpx.Client(timeout=180.0) as client:
        for i, prompt in enumerate(PROMPTS, start=1):
            t0 = time.perf_counter()
            try:
                resp = client.post(BASE_URL, json={"prompt": prompt})
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                # A single slow/failed request (e.g. the gateway's own
                # httpx call to Ollama exceeding its timeout under heavy
                # background CPU load) shouldn't take down the whole batch.
                # No row gets logged to gateway.db for a failed request
                # (main.py never reaches log_request on this path), so the
                # summary's totals will simply be a bit lower than 30 for
                # runs that hit this - that's reported honestly, not hidden.
                failures += 1
                print(f"[{i:>2}/{len(PROMPTS)}] FAIL  {type(exc).__name__}: {exc}  {prompt[:50]}")
                continue

            elapsed = time.perf_counter() - t0

            hit_marker = "HIT " if data["cache_hit"] else "MISS"
            print(
                f"[{i:>2}/{len(PROMPTS)}] {hit_marker}  "
                f"model={data['model_used']:<13}  "
                f"latency={data['latency_ms']:>9.1f}ms  "
                f"(wall={elapsed*1000:>9.1f}ms)  "
                f"{prompt[:60]}"
            )

    if failures:
        print(f"\n{failures} of {len(PROMPTS)} requests failed (see FAIL lines above).")

    return start_time_iso


def print_summary(since_iso: str, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT model_used, cache_hit, latency_ms
            FROM requests
            WHERE timestamp >= ?
            ORDER BY id
            """,
            (since_iso,),
        ).fetchall()
    finally:
        conn.close()

    total = len(rows)
    if total == 0:
        print("No rows found for this run - is the server logging correctly?")
        return

    hits = [r for r in rows if r[1] == 1]
    misses = [r for r in rows if r[1] == 0]

    hit_rate = 100.0 * len(hits) / total
    avg_hit_latency = sum(r[2] for r in hits) / len(hits) if hits else 0.0
    avg_miss_latency = sum(r[2] for r in misses) / len(misses) if misses else 0.0

    by_model: dict[str, list[float]] = {}
    for model_used, _cache_hit, latency_ms in rows:
        by_model.setdefault(model_used, []).append(latency_ms)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY".center(60))
    print("=" * 60)
    print(f"{'Total requests':<35}{total:>25}")
    print(f"{'Cache hits':<35}{len(hits):>25}")
    print(f"{'Cache misses':<35}{len(misses):>25}")
    print(f"{'Cache hit rate':<35}{hit_rate:>24.1f}%")
    print("-" * 60)
    print(f"{'Avg latency - cache hits':<35}{avg_hit_latency:>21.1f} ms")
    print(f"{'Avg latency - cache misses':<35}{avg_miss_latency:>21.1f} ms")
    if avg_hit_latency > 0:
        speedup = avg_miss_latency / avg_hit_latency
        print(f"{'Speedup (miss / hit)':<35}{speedup:>23.1f}x")
    print("-" * 60)
    print("Avg latency by model_used:")
    for model_used in sorted(by_model):
        latencies = by_model[model_used]
        avg = sum(latencies) / len(latencies)
        print(f"  {model_used:<25}  n={len(latencies):<4}  avg={avg:>10.1f} ms")
    print("=" * 60)


def archive_db(db_path: str = DB_PATH, runs_dir: str = RUNS_DIR) -> str:
    """Copy gateway.db to runs/gateway_run_<timestamp>.db so this run's raw
    data survives the next run wiping gateway.db. Returns the archive path."""
    Path(runs_dir).mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = Path(runs_dir) / f"gateway_run_{timestamp}.db"
    shutil.copy2(db_path, archive_path)
    print(f"\nArchived this run's gateway.db to {archive_path}")
    return str(archive_path)


if __name__ == "__main__":
    since = run_benchmark()
    print_summary(since)
    archive_db()
