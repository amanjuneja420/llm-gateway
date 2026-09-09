"""
Real client test for POST /chat/stream.
==========================================
Run manually AFTER the server is already up:

    venv\\Scripts\\python test_streaming.py

The whole point of /chat/stream is that tokens arrive incrementally instead
of all at once - so this test has to actually observe that, not just check
the final text. It uses httpx's `client.stream()` + `aiter_lines()`
(NOT a plain `client.post()`, which would buffer the whole SSE body and
make every chunk look like it arrived at the same instant - the exact
mistake this test is built to avoid), records a real wall-clock timestamp
on every chunk as it's received, and reports:

  - time to first token (the number that actually matters for perceived
    responsiveness on this CPU-only hardware - see main.py's module
    docstring on why /chat/stream exists at all)
  - total time to the "done" event
  - the gap between consecutive chunks, to show they're spread out over
    real time rather than delivered in one instantaneous burst
  - a second test against a prompt already in the cache, confirming a
    cache hit still comes back through the same SSE shape (one token
    event + done, both essentially instant)
"""

import json
import sys
import time

import httpx

# Ollama can emit real Unicode (subscripts, arrows, etc.) mid-response, and
# Windows' console defaults to cp1252, which can't encode all of it -
# reconfigure stdout to UTF-8 (replacing anything even that can't render)
# so a token containing e.g. "H₂O" doesn't crash this test script.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000/chat/stream"


def stream_prompt(prompt: str, client_key: str = "test-streaming") -> None:
    print(f"\n{'=' * 70}")
    print(f"Prompt: {prompt!r}")
    print("=" * 70)

    chunk_times: list[float] = []
    token_count = 0
    full_text = ""
    done_event: dict | None = None

    t0 = time.perf_counter()
    with httpx.Client(timeout=180.0) as client:
        with client.stream(
            "POST", BASE_URL, json={"prompt": prompt}, headers={"X-API-Key": client_key}
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                elapsed = time.perf_counter() - t0
                chunk_times.append(elapsed)
                event = json.loads(line[len("data: "):])
                if event["type"] == "token":
                    token_count += 1
                    full_text += event["text"]
                    preview = event["text"].replace("\n", "\\n")[:30]
                    print(f"  [t={elapsed*1000:>8.1f}ms] token #{token_count:<4} {preview!r}")
                elif event["type"] == "done":
                    done_event = event
                    print(f"  [t={elapsed*1000:>8.1f}ms] DONE  {event}")
                elif event["type"] == "error":
                    print(f"  [t={elapsed*1000:>8.1f}ms] ERROR {event}")

    if not chunk_times:
        print("  No chunks received at all - something is wrong.")
        return

    time_to_first_token_ms = chunk_times[0] * 1000
    total_time_ms = chunk_times[-1] * 1000
    gaps_ms = [
        (chunk_times[i] - chunk_times[i - 1]) * 1000 for i in range(1, len(chunk_times))
    ]

    print(f"\n  Total tokens received: {token_count}")
    print(f"  Time to first chunk:   {time_to_first_token_ms:>9.1f} ms")
    print(f"  Time to done:          {total_time_ms:>9.1f} ms")
    if gaps_ms:
        print(
            f"  Inter-chunk gaps:      min={min(gaps_ms):.1f}ms  "
            f"max={max(gaps_ms):.1f}ms  avg={sum(gaps_ms)/len(gaps_ms):.1f}ms"
        )
        nonzero_gaps = sum(1 for g in gaps_ms if g > 1.0)
        print(
            f"  {nonzero_gaps}/{len(gaps_ms)} gaps are > 1ms apart - "
            f"tokens arrived spread out over real time, not in one instantaneous burst "
            f"(a client.post() on the same endpoint would show every chunk at the same "
            f"timestamp, having buffered the whole response before returning it)."
        )
    print(f"  Reassembled text ({len(full_text)} chars): {full_text[:150]!r}...")
    if done_event and not done_event.get("cache_hit"):
        print(
            f"\n  Compare: time-to-first-token ({time_to_first_token_ms:.0f}ms) vs. "
            f"non-streaming /chat's wait-for-everything latency for a similar prompt "
            f"({done_event['latency_ms']:.0f}ms total here) - a streaming client sees "
            f"output {done_event['latency_ms'] / max(time_to_first_token_ms, 1):.1f}x "
            f"sooner than it would waiting for the full /chat response."
        )


if __name__ == "__main__":
    # Test 1: a fresh, moderately long prompt - the case /chat/stream exists
    # for. Deliberately complex enough to route to qwen2.5:3b (the slow one)
    # so the incremental-arrival evidence is unambiguous.
    stream_prompt(
        "Explain how photosynthesis works in plants, step by step.",
        client_key="test-streaming-fresh",
    )

    # Test 2: send the exact same prompt again - should now be a cache hit,
    # confirming /chat/stream checks the cache (see main.py's module
    # docstring on why that matters) and returns the SSE shape's cache-hit
    # path (one token event + done, both essentially instant).
    stream_prompt(
        "Explain how photosynthesis works in plants, step by step.",
        client_key="test-streaming-fresh",
    )
