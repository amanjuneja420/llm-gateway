"""
Aggregate benchmark.py results across multiple independent runs.
===================================================================
This is not itself a benchmark - it's a small reporting script over numbers
already printed by separate benchmark.py runs (each against a freshly
restarted server: empty cache, fresh gateway.db). Run manually, by hand,
after collecting a set of runs, to see how much the cache-hit speedup
multiplier and cache hit rate actually vary run-to-run on a shared,
CPU-only machine where background load is not controlled for.

RUNS below is filled in directly from each run's printed BENCHMARK SUMMARY
block (see benchmark_output.log / benchmark_run_*.log for the raw output).
"""

RUNS = [
    # label, total_requests, cache_hits, avg_hit_ms, avg_miss_ms
    ("Run A (section 5, first clean run)", 30, 6, 35.3, 27489.6),
    ("Run B (post-instrumentation re-run)", 30, 6, 90.5, 36146.5),
    ("Run 1 (multi-run set)", 30, 6, 63.0, 43214.6),
    ("Run 2 (multi-run set, 1 request timed out)", 29, 6, 61.3, 39310.3),
    ("Run 3 (multi-run set)", 30, 6, 91.7, 40389.6),
]


def main() -> None:
    rows = []
    for label, total, hits, avg_hit, avg_miss in RUNS:
        hit_rate = 100.0 * hits / total
        speedup = avg_miss / avg_hit
        rows.append((label, total, hit_rate, avg_hit, avg_miss, speedup))

    print("=" * 100)
    print("COMBINED BENCHMARK RESULTS ACROSS RUNS".center(100))
    print("=" * 100)
    header = f"{'Run':<44}{'n':>4}{'hit rate':>10}{'avg hit ms':>13}{'avg miss ms':>14}{'speedup':>10}"
    print(header)
    print("-" * 100)
    for label, total, hit_rate, avg_hit, avg_miss, speedup in rows:
        print(
            f"{label:<44}{total:>4}{hit_rate:>9.1f}%{avg_hit:>13.1f}{avg_miss:>14.1f}{speedup:>9.1f}x"
        )
    print("-" * 100)

    speedups = [r[5] for r in rows]
    hit_rates = [r[2] for r in rows]

    def stats(values):
        return min(values), max(values), sum(values) / len(values)

    sp_min, sp_max, sp_avg = stats(speedups)
    hr_min, hr_max, hr_avg = stats(hit_rates)

    print(f"{'Speedup (miss/hit) across runs':<44}"
          f"min={sp_min:>7.1f}x   max={sp_max:>7.1f}x   avg={sp_avg:>7.1f}x")
    print(f"{'Cache hit rate across runs':<44}"
          f"min={hr_min:>6.1f}%   max={hr_max:>6.1f}%   avg={hr_avg:>6.1f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
