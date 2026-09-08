"""
Cost estimate: what would this session's token volume have cost on paid
hosted APIs?
=============================================================================
Illustrative only - NOT a live pricing integration. The pricing table below
is a hardcoded snapshot that WILL go stale; it exists to make one point
concrete using this project's own real numbers: routing simple prompts to a
small local model, and caching near-duplicates, avoids real per-token cost
that an all-paid-API setup would incur.

The two reference models named when this module was scoped - GPT-4o-mini
and Claude Haiku 3.5 - have both since been superseded on their providers'
own current pricing pages, checked directly rather than assumed (same "the
model you had in mind might not be the current one" lesson as this
project's earlier Gemini and Groq model selections):

  - OpenAI GPT-4o-mini no longer appears in OpenAI's pricing tables (only
    in one passing note about a legacy web-search-tool billing exception,
    not as a priced model in its own right). Replaced with GPT-5.6 Luna,
    the cheapest model in OpenAI's current "flagship models" table - the
    closest thing to a current "mini" tier.
    Source: https://developers.openai.com (Pricing | OpenAI API page),
    checked 2026-09-08. $1.20 / 1M output tokens (standard tier, short
    context).
  - Claude Haiku 3.5 is listed on Anthropic's own pricing page as
    "retired, except on Bedrock and Google Cloud". Replaced with Claude
    Haiku 4.5, the current Haiku model.
    Source: https://platform.claude.com/en/docs/about-claude/pricing
    (Pricing - Claude Platform Docs), checked 2026-09-08. $5 / 1M output
    tokens (standard tier).

Only OUTPUT-token pricing is used below. eval_count in gateway.db is
Ollama's (and, for cloud failover rows, Groq's) generated-token count; this
project never logs prompt/input token counts, so an apples-to-apples
input+output estimate isn't possible from this data alone. That's a real
limitation of this estimate, not hidden: it under-counts what a paid API
would actually charge (input tokens are never free), so treat the numbers
below as a floor, not a full bill.
"""

import sqlite3

from db import DB_PATH

# {label: price per single output token, in USD}
REFERENCE_PRICING_PER_TOKEN = {
    "OpenAI GPT-5.6 Luna": 1.20 / 1_000_000,
    "Anthropic Claude Haiku 4.5": 5.00 / 1_000_000,
}


def main(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT served_by, eval_count FROM requests WHERE eval_count IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No rows with eval_count found in gateway.db - run benchmark.py or the server first.")
        return

    total_tokens = sum(r[1] for r in rows)
    # served_by is NULL for rows logged before Phase 2 added the column
    # (e.g. the committed Run 3 snapshot) - those are all real Ollama calls
    # from before served_by existed, so NULL counts as local here too.
    local_tokens = sum(r[1] for r in rows if r[0] in (None, "ollama"))
    cloud_tokens = total_tokens - local_tokens

    print("=" * 64)
    print("COST ESTIMATE: token volume on paid hosted APIs vs. local".center(64))
    print("=" * 64)
    print(f"{'Total tokens generated (eval_count)':<42}{total_tokens:>20,}")
    print(f"{'  served by local Ollama (actual cost: $0)':<42}{local_tokens:>20,}")
    print(f"{'  served by cloud failover (Gemini/Groq)':<42}{cloud_tokens:>20,}")
    print("-" * 64)
    print("If ALL of this volume had instead gone to a paid API:")
    print()
    for label, price_per_token in REFERENCE_PRICING_PER_TOKEN.items():
        cost = total_tokens * price_per_token
        print(f"  {label:<42}${cost:>9.4f}")
    print()
    print(f"  {'Local Ollama (what actually happened)':<42}${0:>9.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
