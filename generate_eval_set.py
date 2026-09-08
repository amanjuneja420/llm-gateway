"""
Router evaluation harness - generation step only.
=====================================================
Runs a diverse set of prompts through BOTH qwen2.5:1.5b and qwen2.5:3b
directly, bypassing router.py entirely - this script doesn't route, it
generates both models' answers to every prompt so a human can judge them
side by side afterward. Writes prompt / response_1.5b / response_3b /
which_is_better to a CSV, with which_is_better left EMPTY on purpose.

This script does NOT judge anything - "which_is_better" (expected values:
"1.5b", "3b", or "tie") is filled in by hand, separately, after this runs.
Nothing here scores or picks a winner.

Doesn't touch the live gateway (main.py), the cache, or gateway.db - just
direct calls via the same OllamaBackend class the gateway itself uses (see
backends.py), for consistency with the rest of the project.

Run manually, after Ollama is up with both models pulled:
    venv\\Scripts\\python generate_eval_set.py

Expect this to take a while: 50 prompts x 2 models = 100 real generations,
and qwen2.5:3b alone has run anywhere from ~20s to well over a minute per
prompt on this CPU-only machine in earlier benchmark runs (see README).
Progress prints per prompt as it goes.
"""

import asyncio
import csv
import time

from backends import OllamaBackend

OUTPUT_PATH = "eval_set.csv"

model_15b = OllamaBackend(model="qwen2.5:1.5b")
model_3b = OllamaBackend(model="qwen2.5:3b")

# 50 prompts total: the 20 distinct underlying questions from benchmark.py's
# 30-prompt set (deduplicated - that set's paraphrase groups exist to test
# the semantic cache, which is irrelevant here, so only one phrasing per
# question is kept) plus 30 new prompts covering domains benchmark.py
# doesn't touch (math, biology, CS/data structures, economics, literature),
# still mixing short/simple and long/complex.
PROMPTS = [
    # --- from benchmark.py, deduplicated to one phrasing per question ---
    "What is the capital of Japan?",
    "Explain how photosynthesis works in plants.",
    "What is 15 times 7?",
    "Name three primary colors.",
    "Compare the advantages and disadvantages of remote work versus office work.",
    "What is the largest planet in our solar system?",
    "Why do seasons change throughout the year?",
    "What year did World War II end?",
    "What is the chemical symbol for gold?",
    "What are the steps to bake a basic loaf of bread from scratch?",
    "List the days of the week.",
    "Translate 'hello' to Spanish.",
    "Analyze the causes of the fall of the Roman Empire.",
    "Summarize the plot of Romeo and Juliet in a few sentences.",
    "What is the difference between machine learning and deep learning?",
    "Evaluate the pros and cons of electric vehicles compared to gasoline cars.",
    "What is the tallest mountain in the world?",
    (
        "I've been thinking about switching careers into software "
        "engineering, and I want to understand what skills I should focus "
        "on and how long it might realistically take."
    ),
    "What is the boiling point of water in Celsius?",
    "Contrast the benefits of solar power and wind power for home energy use.",
    # --- new: short/simple ---
    "What is the square root of 144?",
    "Name the closest star to Earth.",
    "What is the freezing point of water in Fahrenheit?",
    "How many continents are there?",
    "What is the currency used in Japan?",
    "Spell the word 'necessary' correctly.",
    "What is the chemical formula for water?",
    "Name two prime numbers between 10 and 20.",
    "What language is primarily spoken in Brazil?",
    "How many days are there in a leap year?",
    "What is the capital of Australia?",
    "Convert 100 Fahrenheit to Celsius.",
    "Name a mammal that lays eggs.",
    "What is the plural of 'cactus'?",
    "How many sides does a hexagon have?",
    # --- new: long/complex ---
    "Explain why the sky appears blue during the day and red or orange at sunset.",
    "Compare the economic systems of capitalism and socialism.",
    "What are the steps involved in photosynthesis at the cellular level?",
    "Why do some programming languages use garbage collection while others require manual memory management?",
    "Summarize the causes and consequences of the Industrial Revolution.",
    "Explain the difference between a stack and a queue data structure, with an example use case for each.",
    "Analyze why the Roman aqueduct system was considered an engineering marvel for its time.",
    "Compare and contrast supervised and unsupervised machine learning approaches.",
    "Explain how vaccines work to build immunity against diseases.",
    "Why is it important to normalize a database, and what problems can arise if you don't?",
    "Describe the steps to set up a basic REST API using Python and FastAPI.",
    "Evaluate the tradeoffs between renewable energy sources like solar and wind versus fossil fuels.",
    "Explain the difference between TCP and UDP and when you would use each one.",
    "Why do interest rates affect inflation, and how do central banks use this relationship?",
    'Compare the plots and themes of the novels "1984" and "Brave New World."',
]


async def generate_for_model(backend: OllamaBackend, prompt: str) -> str:
    try:
        result = await backend.generate(prompt)
        return result.text
    except Exception as exc:
        # Recorded in the CSV rather than crashing the whole run - a human
        # judging the sheet will see this cell and can just skip that row.
        return f"[ERROR: {type(exc).__name__}: {exc}]"


async def generate_eval_set() -> None:
    total_calls = len(PROMPTS) * 2
    print(f"Generating eval set: {len(PROMPTS)} prompts x 2 models = {total_calls} real generations")
    print("This can take a while on CPU-only hardware, especially for qwen2.5:3b.\n")

    rows = []
    run_start = time.perf_counter()
    for i, prompt in enumerate(PROMPTS, start=1):
        t0 = time.perf_counter()
        response_15b = await generate_for_model(model_15b, prompt)
        response_3b = await generate_for_model(model_3b, prompt)
        elapsed = time.perf_counter() - t0
        print(f"[{i:>2}/{len(PROMPTS)}] {elapsed:>6.1f}s  {prompt[:70]}")
        rows.append(
            {
                "prompt": prompt,
                "response_1.5b": response_15b,
                "response_3b": response_3b,
                "which_is_better": "",
            }
        )

    total_elapsed = time.perf_counter() - run_start

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["prompt", "response_1.5b", "response_3b", "which_is_better"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH} in {total_elapsed:.1f}s total.")
    print(
        "which_is_better is intentionally empty - fill it in by hand with "
        "'1.5b', '3b', or 'tie'. This script only generates; it does not judge."
    )


if __name__ == "__main__":
    asyncio.run(generate_eval_set())
