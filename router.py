"""
Heuristic prompt router.
=========================
IMPORTANT: this is an explicit HEURISTIC, not a trained classifier. There is
no model here, no training data, no learned weights — just two cheap,
inspectable signals computed directly on the prompt text:

  1. word count       -> long prompts tend to need more reasoning/context
  2. complexity words  -> certain keywords ("explain", "compare", "steps",
                           "why", ...) correlate with multi-step / open-ended
                           reasoning tasks, regardless of length

This was a deliberate choice over a trained router given the time box: it's
instant (no inference cost of its own), fully transparent, and easy to justify
line by line. A learned router could route more accurately in principle, but
it needs labeled training data and its own evaluation, which this MVP does
not have time for. If this project grew, the natural next step would be to
log (prompt, heuristic_decision, actual_latency/quality) pairs from
production traffic and train a lightweight classifier on top of that data.

Tune the two constants below to change routing behavior; nothing else in this
file needs to change to retune it.
"""

import re

# Prompts with strictly more words than this are considered "long" and get
# routed to the bigger model. 20 words is roughly the length where a prompt
# stops being a single simple question and starts being a paragraph-style
# request (multiple clauses, some context-setting) - chosen by eyeballing a
# handful of short vs. long example prompts, not derived from data.
WORD_COUNT_THRESHOLD = 20

# Presence of any of these words (case-insensitive, whole-word match)
# strongly suggests the prompt wants multi-step reasoning, a structured
# explanation, or a comparison - the kind of task the bigger model handles
# more reliably even when the prompt itself is short (e.g. "Why is the sky
# blue?" is only 5 words but is not a "simple" prompt).
COMPLEXITY_KEYWORDS = {
    "explain",
    "compare",
    "contrast",
    "steps",
    "step-by-step",
    "why",
    "how does",
    "difference between",
    "pros and cons",
    "analyze",
    "summarize",
    "evaluate",
}

SIMPLE_MODEL = "qwen2.5:1.5b"
COMPLEX_MODEL = "qwen2.5:3b"

# Compiled once at import, not per call: a single word-boundary-delimited
# alternation of every keyword/phrase above. This is a genuine whole-word
# match, matching the comment's claim above it (which the previous plain
# `keyword in prompt_lower` substring check did not - "footsteps" contains
# "steps" as a substring and would incorrectly trip this, despite having
# nothing to do with instructions or reasoning). \b bounds the OUTER edges
# of a multi-word phrase like "how does" too (Python's \b is defined on
# \w = [A-Za-z0-9_] boundaries, not literal spaces), so a phrase still
# matches as a whole unit; re.escape() on each keyword makes the literal
# hyphens in "step-by-step" safe to include in the pattern.
_COMPLEXITY_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(keyword) for keyword in COMPLEXITY_KEYWORDS) + r")\b"
)


def _has_complexity_keyword(prompt_lower: str) -> bool:
    return _COMPLEXITY_PATTERN.search(prompt_lower) is not None


def route(prompt: str) -> str:
    """
    Decide which Ollama model should handle this prompt.

    Pure function, no I/O - easy to unit test and easy to explain: given a
    prompt, we only look at its word count and whether it contains a
    complexity keyword. That's the entire heuristic.
    """
    word_count = len(prompt.split())
    prompt_lower = prompt.lower()

    if word_count > WORD_COUNT_THRESHOLD or _has_complexity_keyword(prompt_lower):
        return COMPLEX_MODEL

    return SIMPLE_MODEL


if __name__ == "__main__":
    # Quick manual sanity check - run `python router.py` to eyeball decisions.
    samples = [
        "What is 2+2?",
        "What's the capital of France?",
        "Why is the sky blue?",
        "Explain how a transformer neural network works in detail.",
        "Compare Python and JavaScript for backend web development.",
        "List the first five prime numbers.",
        (
            "I'm building a small side project and I want to understand the "
            "tradeoffs between using SQLite and Postgres for a low-traffic "
            "internal tool, can you walk me through it"
        ),
        # Whole-word matching regression cases - a plain substring check
        # would incorrectly route both of these to the complex model, since
        # "steps" and "why" appear as substrings of longer words here.
        "I heard footsteps outside my window last night.",  # "steps" is a substring only - should stay SIMPLE
        "What are the steps to bake bread?",  # genuine "steps" usage - should still route COMPLEX
    ]
    for s in samples:
        print(f"[{route(s):>13}]  ({len(s.split()):>2} words)  {s}")
