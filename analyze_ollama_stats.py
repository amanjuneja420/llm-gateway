"""
Ollama timing breakdown.
=========================
Standalone analysis script - run manually after benchmark.py, against the
gateway.db produced by that run. Answers one specific question, per model:
was the earlier ~48s average for qwen2.5:3b caused by Ollama repeatedly
reloading the model between requests (load_duration), or is it genuinely
just slow token generation on this CPU (eval_duration)?

Only looks at cache misses (cache_hit = 0) served by Ollama specifically,
since cache hits never call Ollama and have no load/eval numbers to
report - and, since Phase 2's failover chain, a cache miss doesn't
necessarily mean Ollama either: Gemini and Groq are also possible values
of served_by. A row with served_by IS NULL predates that column
entirely (added in Phase 2) - at that point in the project there was no
failover chain yet, so every such row is guaranteed to be a genuine
Ollama call and is included on that basis, not excluded just because the
column happens to be empty. Rows actually served_by='gemini'/'groq' are
excluded and reported separately below, rather than silently folded into
these Ollama-specific load/eval percentages, which mean something
different (or nothing - Gemini reports neither) for a cloud backend.
"""

import sqlite3

DB_PATH = "gateway.db"


def main(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        all_misses = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE cache_hit = 0"
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT model_used, latency_ms, load_duration_ms, eval_count, eval_duration_ms
            FROM requests
            WHERE cache_hit = 0 AND (served_by = 'ollama' OR served_by IS NULL)
            ORDER BY id
            """
        ).fetchall()
        failover_misses = conn.execute(
            "SELECT served_by, COUNT(*) FROM requests "
            "WHERE cache_hit = 0 AND served_by IN ('gemini', 'groq') "
            "GROUP BY served_by"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No Ollama cache-miss rows found in gateway.db - run benchmark.py first.")
        return

    if failover_misses:
        excluded = ", ".join(f"{count} {backend}" for backend, count in failover_misses)
        print(
            f"Excluded {excluded} cache-miss row(s) served by cloud failover, not Ollama - "
            f"load_duration/eval_duration mean something different (or nothing) for those "
            f"backends. {len(rows)}/{all_misses} total cache misses were Ollama and are "
            f"analyzed below.\n"
        )

    by_model: dict[str, list[tuple]] = {}
    for row in rows:
        by_model.setdefault(row[0], []).append(row)

    print("=" * 78)
    print("OLLAMA TIMING BREAKDOWN (cache misses only - real model calls)".center(78))
    print("=" * 78)

    for model in sorted(by_model):
        entries = by_model[model]
        n = len(entries)

        avg_latency = sum(e[1] for e in entries) / n
        avg_load = sum(e[2] or 0 for e in entries) / n
        avg_eval_count = sum(e[3] or 0 for e in entries) / n
        avg_eval_duration = sum(e[4] or 0 for e in entries) / n

        tokens_per_sec = (
            avg_eval_count / (avg_eval_duration / 1000) if avg_eval_duration > 0 else 0.0
        )
        load_frac = 100.0 * avg_load / avg_latency if avg_latency > 0 else 0.0
        eval_frac = 100.0 * avg_eval_duration / avg_latency if avg_latency > 0 else 0.0

        print(f"\nModel: {model}   (n={n} requests)")
        print(f"  {'avg total latency_ms':<32}{avg_latency:>15.1f}")
        print(f"  {'avg load_duration_ms':<32}{avg_load:>15.1f}")
        print(f"  {'avg eval_count (tokens)':<32}{avg_eval_count:>15.1f}")
        print(f"  {'avg eval_duration_ms':<32}{avg_eval_duration:>15.1f}")
        print(f"  {'tokens/sec':<32}{tokens_per_sec:>15.2f}")
        print(f"  {'load_duration / latency':<32}{load_frac:>14.1f}%")
        print(f"  {'eval_duration / latency':<32}{eval_frac:>14.1f}%")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
