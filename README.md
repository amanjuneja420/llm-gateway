# LLM Gateway MVP

A scoped-down local LLM gateway: one FastAPI endpoint that routes prompts between two
local Ollama models with a heuristic, semantically caches similar prompts to skip
redundant model calls, logs every request to SQLite, and fails over to Gemini
(cloud) if the routed local model call fails. Built and tested primarily against
local Ollama models on a CPU-only Windows machine — no GPU — with Gemini as the one
cloud dependency, used only as a fallback.

**Explicitly out of scope** (by design, not by omission): rate limiting, Redis/Postgres,
a full circuit-breaker state machine (see "Gemini failover" below for what's actually
implemented instead - simple, not stateful), load balancing across replicas, streaming
responses, Docker/Kubernetes, a trained ML router, any UI/dashboard.

## Architecture

```
                POST /chat {"prompt": "..."}
                            |
                            v
                 1. Embed prompt (all-MiniLM-L6-v2)
                            |
                            v
              2. Semantic cache lookup (cosine sim, numpy)
                     /                      \
              similarity >= 0.92        similarity < 0.92
                    |                          |
                    v                          v
          return cached response      3. Heuristic router picks
          (cache_hit=true,               qwen2.5:1.5b or qwen2.5:3b
           no model call)                       |
                    |                            v
                    |                  4. Call Ollama /api/generate
                    |                       (fails? -> 4b. call Gemini instead,
                    |                        failed_over=true - see below)
                    |                            |
                    |                            v
                    |                  5. Store (embedding, prompt,
                    |                     response) in cache
                    |                            |
                    +-------------+--------------+
                                  |
                                  v
                    6. Log request to SQLite (gateway.db)
                                  |
                                  v
        Return {response, model_used, cache_hit, latency_ms, failed_over}
```

Small modules, each independently runnable/testable:

- [`router.py`](router.py) — the heuristic model router (no I/O, pure function)
- [`cache.py`](cache.py) — the semantic cache (embedding + cosine similarity)
- [`db.py`](db.py) — SQLite schema + logging helper
- [`main.py`](main.py) — FastAPI app that wires the router, cache, Ollama, and Gemini failover together behind `POST /chat`
- [`benchmark.py`](benchmark.py) — standalone load/test script, run manually against the live server
- [`analyze_ollama_stats.py`](analyze_ollama_stats.py) — standalone analysis script, breaks down Ollama's own load/eval timing per model from `gateway.db`
- [`aggregate_benchmark_runs.py`](aggregate_benchmark_runs.py) — combines several `benchmark.py` runs' printed summaries into one table with min/max/avg speedup and hit rate
- [`test_failover.py`](test_failover.py) — standalone test script for the Gemini failover path, run manually
- `runs/` — raw `gateway.db` snapshot from each `benchmark.py` run (`gateway_run_<UTC timestamp>.db`), archived automatically before the next run wipes the live `gateway.db`
- `.env` (not committed, see `.env.example`) — holds `GEMINI_API_KEY`, loaded at startup via `python-dotenv`

## The routing heuristic — and why it's a heuristic, not a trained model

`router.py` decides which model handles a prompt using two cheap, fully inspectable
signals computed directly on the prompt text — **no model, no training data, no
learned weights**:

1. **Word count** — prompts longer than `WORD_COUNT_THRESHOLD = 20` words are treated
   as "long" and routed to the bigger model. 20 was chosen by eyeballing example
   prompts: it's roughly where a prompt stops being a single simple question and
   becomes a paragraph-style, multi-clause request.
2. **Complexity keywords** — a fixed set (`explain`, `compare`, `contrast`, `steps`,
   `why`, `how does`, `difference between`, `pros and cons`, `analyze`, `summarize`,
   `evaluate`) that correlate with multi-step or open-ended reasoning tasks, catching
   short-but-hard prompts like "Why is the sky blue?" that word count alone would miss.

If either signal fires, the prompt goes to `qwen2.5:3b`; otherwise `qwen2.5:1.5b`.

**Why a heuristic instead of a trained classifier:** given the time box, a heuristic is
instant to run (no inference cost of its own), fully transparent (every routing
decision can be explained by pointing at the exact line of code that made it), and
needs zero training data or evaluation harness to trust. A trained router could route
more accurately in principle, but that requires labeled examples of which model
*should* have handled a given prompt and its own accuracy evaluation — infrastructure
this MVP doesn't have time to build today. The honest framing for an interview:
*"I used an explicit heuristic and can justify every constant in it; a trained router
would be the natural next iteration if I had labeled routing data."*

## The semantic cache and its similarity threshold

`cache.py` embeds every prompt with `sentence-transformers/all-MiniLM-L6-v2`, keeps
all `(embedding, prompt, response, model_used)` tuples in a plain Python list (no
FAISS, no vector DB), and on each new prompt computes cosine similarity against every
cached embedding with a single numpy dot product (embeddings are stored pre-normalized,
so cosine similarity is just a dot product). If the best match scores at or above
`SIMILARITY_THRESHOLD = 0.92`, that cached response is returned instantly and no model
call is made.

**How 0.92 was chosen:** measured directly, not guessed. Using this exact model:

| Prompt pair | Cosine similarity |
|---|---|
| "What is the capital of France?" vs "What's France's capital city?" (paraphrase) | 0.944 |
| "What is the capital of France?" vs "Can you tell me the capital of France" (paraphrase) | 0.934 |
| "What is the capital of France?" vs "What is the population of France?" (related, different question) | 0.709 |
| "What is the capital of France?" vs "What is the capital of Germany?" (related, different answer) | 0.663 |
| "What is the capital of France?" vs "How do I bake sourdough bread?" (unrelated) | 0.133 |

Genuine paraphrases land in the 0.93–0.95 range; the nearest "related but actually a
different question" pairs land around 0.66–0.71. 0.92 sits comfortably in the gap
between those two clusters — high enough that a subtly different question (different
country, different attribute) won't incorrectly hit the cache, low enough that real
paraphrasing reliably hits. It's a single named constant at the top of `cache.py`, so
it's trivial to retune if more data suggests otherwise.

## Gemini failover

The gateway's two Ollama models are the primary backends; Gemini is a **cloud
fallback for when the routed local call fails**, not a third routing option the
heuristic ever picks directly.

**What counts as a failure, precisely:** in `chat()`, the call to `call_ollama()` is
wrapped in a bare `except Exception`. This catches everything the local call can
throw, including:
- A timeout past `OLLAMA_TIMEOUT_SECONDS = 120.0` (the same constant `call_ollama`'s
  own `httpx.AsyncClient` already used for its timeout - failover reuses it rather
  than introducing a second, different threshold to reason about). httpx raises
  `ConnectTimeout` or `ReadTimeout` in this case.
- A connection failure (Ollama not running, wrong port, network refused) -
  `httpx.ConnectError`.
- An HTTP error status from Ollama itself (`resp.raise_for_status()` raising
  `HTTPStatusError` on a 4xx/5xx).
- Any other exception the call raises, on the theory that "local call didn't produce
  a usable response" should fail over regardless of the exact exception type.

On any of the above, the *same request* (same prompt) is immediately retried against
`GEMINI_MODEL` (currently `gemini-3.8-flash` - see the note in `main.py` on why not
the originally planned `gemini-1.5-flash`, which is deprecated and 404s on the live
API). The response is returned normally, with `model_used` set to the Gemini model
name and a new `failed_over: true` field on `ChatResponse`. `db.py` logs
`failed_over` as its own column (0/1) on every request, so failovers are queryable
after the fact, not just visible in the moment.

**This is explicitly not a circuit breaker.** There's no failure counter, no
open/half-open/closed state, no cooldown window before trying Ollama again. Every
request independently tries the local model first; only that one request's failure
decides whether it falls back to Gemini. If Ollama recovers a millisecond later, the
very next request goes straight back to it. A real circuit breaker (trip after N
consecutive failures, stop even trying the local model for a cooldown period, half-open
probe requests) is meaningfully more machinery than this MVP's time box covers, and is
called out explicitly in this README's scope, not silently skipped.

**Security note on the API key:** `call_gemini()` sends `GEMINI_API_KEY` as the
`x-goog-api-key` HTTP header, never as a URL query parameter. httpx exceptions (and
anything that logs them) include the request URL in their message - a key-in-URL
would leak the secret into ordinary error output the moment a Gemini call ever
failed. A header never appears in that message.

**Setup:** copy `.env.example` to `.env` and put a real key in it:
```bash
copy .env.example .env
```
Then edit `.env` and replace `your_key_here` with a real Gemini API key. `.env` is
gitignored and is never committed; `main.py` loads it at startup via
`python-dotenv`'s `load_dotenv()`.

**Verification ([`test_failover.py`](test_failover.py)):** imports `main.py` directly
(no live server needed) so it can monkeypatch `main.OLLAMA_URL` to an unreachable port
for exactly one call, simulating a local failure without touching the real Ollama
service. Real output from an actual run, key redacted from nothing because the script
never prints it:

```
=== Test 1: normal request through Ollama (should be unaffected) ===
PASS - model_used=qwen2.5:1.5b  cache_hit=False  failed_over=False  latency_ms=1283.1

=== Test 2: simulated local failure -> Gemini failover ===
Ollama call failed (ConnectError: All connection attempts failed) - failing over to Gemini
PASS - model_used=gemini-3.8-flash  failed_over=True  latency_ms=5629.9
Gemini response (truncated): 'Europa'

=== Verifying failed_over=1 was actually logged to gateway.db ===
PASS - row id=34  model_used=gemini-3.8-flash  failed_over=1  latency_ms=5629.9  prompt='Name one moon of Jupiter, for the failover test.'

============================================================
FAILOVER TEST SUMMARY
============================================================
  PASS   normal request
  PASS   gemini failover
  PASS   db logging
============================================================
```

Note the path this took to get here: the first run of this test correctly detected
the simulated failure and correctly attempted failover, but failed at the last step
with `GEMINI_API_KEY is not set` (no `.env` existed yet) - itself useful evidence
that the failure-detection and failover-attempt logic work independently of whether
Gemini itself succeeds. The second run, still with the originally planned
`gemini-1.5-flash`, got as far as a real HTTP call to Gemini and back a `404 Not
Found` - that model has been deprecated since this project was scoped, confirmed by
querying `GET /v1beta/models` for this key and finding it absent from the list. Only
after switching to `gemini-3.8-flash` did the full path succeed end to end. All three
states are reported here rather than only the final passing one, because a debugging
path that actually happened is more honest evidence than a clean run that skips it.

## How to run

All commands from the project root (`llm-gateway/`), using the existing venv.

**1. Make sure Ollama is running with both models pulled:**
```bash
ollama list
# should show qwen2.5:1.5b and qwen2.5:3b
```

**2. (Optional, only needed for Gemini failover) Set up `.env`:**
```bash
copy .env.example .env
```
Edit `.env` and put a real Gemini API key in place of `your_key_here`. Without this,
everything except the Gemini failover path works exactly the same - a failed local
call will itself fail (with a clear "GEMINI_API_KEY is not set" error) instead of
successfully failing over.

**3. Start the gateway server:**
```bash
venv\Scripts\python -m uvicorn main:app --port 8000
```
First startup takes ~20-25s while the embedding model loads into memory. Leave this
running in its own terminal.

**4. (Optional) Verify the Gemini failover path:**
```bash
venv\Scripts\python test_failover.py
```
Imports `main.py` directly rather than hitting the live server - see "Gemini
failover" above for what it checks and its real output.

**5. In a second terminal, run the benchmark:**
```bash
venv\Scripts\python benchmark.py
```
This fires 30 prompts (a mix of short/simple, long/complex, and several
paraphrase groups spread through the batch) at the running server, then queries
`gateway.db` and prints a summary: total requests, cache hit rate, average latency
for cache hits vs. misses, and average latency broken down by which model handled
each request. That printed block is the real, reproducible number set for this
project — see the run below.

**6. (Optional) Break down Ollama's own load/eval timing per model:**
```bash
venv\Scripts\python analyze_ollama_stats.py
```
Answers whether a slow model is due to reload overhead (`load_duration_ms`) or
genuine token-generation time (`eval_duration_ms`) — see the section below.

**7. (Optional) Inspect the raw log directly:**
```bash
venv\Scripts\python -c "import sqlite3; [print(r) for r in sqlite3.connect('gateway.db').execute('SELECT * FROM requests ORDER BY id')]"
```

## Benchmark results (5 independent runs)

The cache-hit speedup multiplier turned out to vary noticeably run-to-run on this
shared, CPU-only machine (778x in the first clean run, 400x in a later one) — that's
background CPU load changing the *absolute* miss latency, not noise in the cache
itself (hit latency stays consistently in the tens-to-hundreds of milliseconds
regardless of load). To report an honest number instead of cherry-picking one run,
`benchmark.py` was run 5 separate times, each against a freshly restarted server
(empty cache, fresh `gateway.db`, same 30-prompt set every time, `keep_alive: 30m`
on every Ollama call so models stay resident):

```
====================================================================================================
                               COMBINED BENCHMARK RESULTS ACROSS RUNS
====================================================================================================
Run                                            n  hit rate   avg hit ms   avg miss ms   speedup
----------------------------------------------------------------------------------------------------
Run A (section 5, first clean run)            30     20.0%         35.3       27489.6    778.7x
Run B (post-instrumentation re-run)           30     20.0%         90.5       36146.5    399.4x
Run 1 (multi-run set)                         30     20.0%         63.0       43214.6    685.9x
Run 2 (multi-run set, 1 request timed out)    29     20.7%         61.3       39310.3    641.3x
Run 3 (multi-run set)                         30     20.0%         91.7       40389.6    440.5x
----------------------------------------------------------------------------------------------------
Speedup (miss/hit) across runs              min=  399.4x   max=  778.7x   avg=  589.2x
Cache hit rate across runs                  min=  20.0%   max=  20.7%   avg=  20.1%
====================================================================================================
```

**Headline result: cache hits were consistently 400x-780x faster than live model
generation across 5 runs (average: ~589x).** Cache hit rate was stable at ~20% every
time (the same 30-prompt set with the same 6 intended paraphrase hits, run 2 lost one
request to a timeout unrelated to the cache — see below — leaving 29 logged instead of
30, hence 20.7% instead of 20.0%). The exact speedup multiple depends on how loaded the
CPU is at the moment (it swings the *miss* latency around, from ~27s to ~43s average
across these runs) but the qualitative result — a cache hit costs tens to low-hundreds
of milliseconds regardless of load, a real model call costs tens of seconds — held in
every single run. This table is reproduced by
[`aggregate_benchmark_runs.py`](aggregate_benchmark_runs.py); the 3 newest runs' raw
per-request output is in [`benchmark_run_1.log`](benchmark_run_1.log),
[`benchmark_run_2.log`](benchmark_run_2.log), [`benchmark_run_3.log`](benchmark_run_3.log).

One robustness issue surfaced by running the batch 5 times instead of once: on a
sufficiently loaded run, a single `qwen2.5:3b` call can exceed `main.py`'s own 120s
httpx timeout to Ollama (run 2, request 23) and return a 500. `benchmark.py` now
catches this per-request and logs a `FAIL` line instead of crashing the whole batch —
a real gap this repeated-run exercise caught that a single run wouldn't have.

**Known limitation on reproducibility, stated plainly rather than hidden:** the
archiving mechanism below (`runs/gateway_run_<timestamp>.db`) didn't exist yet when
Runs A, B, 1, 2, and 3 were executed, so all five are transcribed from that session's
printed summaries/logs rather than re-derivable from a preserved raw `gateway.db`.
Run 3's raw data happened to still be sitting in `gateway.db` (nothing had wiped it
yet) when the archiving script was added, so it was copied into `runs/` retroactively
as `gateway_run_20260907T125043Z.db` - Runs A, B, 1, and 2 have no raw snapshot and
never will. From the *next* `benchmark.py` invocation onward, the script copies
`gateway.db` to `runs/gateway_run_<UTC timestamp>.db` itself immediately after
printing its summary, before the next run's fresh restart wipes it - so every future
run is fully re-derivable from raw per-request rows (including the
`load_duration_ms` / `eval_count` / `eval_duration_ms` instrumentation), not just its
printed summary.

### Detail from the most recent single run (Run 3)

```
============================================================
                     BENCHMARK SUMMARY
============================================================
Total requests                                            30
Cache hits                                                 6
Cache misses                                              24
Cache hit rate                                         20.0%
------------------------------------------------------------
Avg latency - cache hits                            91.7 ms
Avg latency - cache misses                       40389.6 ms
Speedup (miss / hit)                                 440.3x
------------------------------------------------------------
Avg latency by model_used:
  cache                      n=6     avg=      91.7 ms
  qwen2.5:1.5b               n=11    avg=    3386.1 ms
  qwen2.5:3b                 n=13    avg=   71700.3 ms
============================================================
```

Takeaways that held across every run, not just this one:

- **Routing worked as designed**: short/simple prompts consistently went to
  `qwen2.5:1.5b` (low single-digit seconds), longer/keyword-bearing prompts to
  `qwen2.5:3b` (tens of seconds to over a minute) — an order-of-magnitude latency
  difference between the two models on this CPU, which is exactly the cost the
  router exists to manage (don't pay the 3b tax for "What is 15 times 7?").
- **Not every paraphrase hit** — e.g. "How does photosynthesis work?" and "What's the
  highest mountain on Earth?" scored below 0.92 against their near-duplicate and fell
  through to a real model call, in every run. This is expected and, if anything,
  reassuring: 0.92 is a deliberately conservative threshold that favors correctness
  (not returning a wrong cached answer) over maximizing the hit rate. Lowering it
  would catch these looser paraphrases at some risk of false-positive hits on subtly
  different questions — see the similarity table above for the actual gap this
  constant is tuned against.
- Absolute latencies vary run-to-run on a shared CPU — background load on the machine
  matters more than anything in the gateway's own code. The routing split and the
  cache speedup range are the numbers that generalize; a single run's raw milliseconds
  don't, which is the whole reason for reporting a range across 5 runs above instead
  of one run's number.

### Is the 3b model slow because of reload overhead, or genuinely slow generation?

Ollama's `/api/generate` response includes `load_duration` (time spent loading the
model into memory), `eval_count` (tokens generated), and `eval_duration` (time spent
generating those tokens) — all in nanoseconds except `eval_count`. `main.py` extracts
these on every real model call and logs them to `gateway.db` as `load_duration_ms`,
`eval_count`, and `eval_duration_ms` (NULL on cache hits, since no model call was
made). [`analyze_ollama_stats.py`](analyze_ollama_stats.py) queries this and breaks
it down per model — numbers below are from Run 3, the most recent run:

```
==============================================================================
        OLLAMA TIMING BREAKDOWN (cache misses only - real model calls)
==============================================================================

Model: qwen2.5:1.5b   (n=11 requests)
  avg total latency_ms                     3386.1
  avg load_duration_ms                        7.8
  avg eval_count (tokens)                    35.7
  avg eval_duration_ms                     1824.2
  tokens/sec                                19.59
  load_duration / latency                    0.2%
  eval_duration / latency                   53.9%

Model: qwen2.5:3b   (n=13 requests)
  avg total latency_ms                    71700.3
  avg load_duration_ms                       11.1
  avg eval_count (tokens)                   592.8
  avg eval_duration_ms                    69811.7
  tokens/sec                                 8.49
  load_duration / latency                    0.0%
  eval_duration / latency                   97.4%
==============================================================================
```

**Answer: it's genuinely slow generation, not reload overhead — consistently, across
runs.** `load_duration_ms` is under 12ms on average for both models in every run
checked so far — under 0.2% of total latency — meaning `keep_alive` is working and
the model is not being unloaded/reloaded between requests. Essentially all of the
wall-clock time (54-63% for 1.5b, 97%+ for 3b - the remainder is Ollama's own
prompt-eval time plus network/HTTP overhead not captured here) is `eval_duration`:
actual token-by-token generation. The 3b model also runs at roughly half the
throughput of the 1.5b model (~8-10 vs. ~20-22 tokens/sec across runs) and tends to
generate longer responses for this prompt set - both effects compound, which is why
3b's average latency is so much higher and so much more variable run-to-run (longer
responses have more opportunity to be slowed down by background CPU load). This is
the expected shape for CPU-only inference on integrated graphics: no GPU to
parallelize matrix multiplies, so generation speed scales roughly with parameter
count and response length, not with how often the model gets swapped in and out of
memory.
