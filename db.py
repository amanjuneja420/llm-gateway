"""
SQLite request logging.
========================
One table, zero setup beyond a local .db file. Uses the stdlib `sqlite3`
module synchronously - each insert is a single fast local write, so it's not
worth the complexity of an async wrapper for an MVP handling one request at
a time.
"""

import sqlite3
from datetime import datetime, timezone

DB_PATH = "gateway.db"

# Prompts are truncated to this many characters before being stored, per the
# spec - we only need enough of the prompt to recognize it later, not a full
# transcript.
PROMPT_LOG_CHARS = 200


# Columns added after the table's original CREATE, each migrated in
# non-destructively below if an older gateway.db (e.g. the Run 3 snapshot
# committed to the repo) doesn't have them yet. Keeping this as a dict makes
# adding the next column later a one-line change instead of a new ad hoc
# ALTER TABLE block.
_MIGRATED_COLUMNS = {
    "failed_over": "INTEGER NOT NULL DEFAULT 0",
    "request_id": "TEXT",
    "served_by": "TEXT",
}


def init_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prompt TEXT NOT NULL,
                model_used TEXT NOT NULL,
                cache_hit INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                load_duration_ms REAL,
                eval_count INTEGER,
                eval_duration_ms REAL,
                failed_over INTEGER NOT NULL DEFAULT 0,
                request_id TEXT,
                served_by TEXT
            )
            """
        )
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
        for column, sql_type in _MIGRATED_COLUMNS.items():
            if column not in existing_cols:
                conn.execute(f"ALTER TABLE requests ADD COLUMN {column} {sql_type}")
        conn.commit()
    finally:
        conn.close()


def log_request(
    prompt: str,
    model_used: str,
    cache_hit: bool,
    latency_ms: float,
    load_duration_ms: float | None = None,
    eval_count: int | None = None,
    eval_duration_ms: float | None = None,
    failed_over: bool = False,
    request_id: str | None = None,
    served_by: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    """Log one request. model_used is 'cache' on a cache hit, otherwise the
    actual model name that produced the response (e.g. 'qwen2.5:1.5b',
    'gemini-2.5-flash', or 'openai/gpt-oss-20b' on Groq). load_duration_ms,
    eval_count, and eval_duration_ms come from the backend's own response
    where it reports them (Ollama: all three; Groq: eval_count and
    eval_duration_ms; Gemini: neither) and are NULL otherwise. failed_over
    is True only when served_by is "gemini" or "groq" - NOT simply
    "whenever served_by isn't 'ollama'", since served_by="cache" isn't
    "ollama" either but a cache hit is the healthy fast path, not a
    failover (see main.py's chat(), which sets this explicitly rather than
    relying on that broader-sounding rule, for exactly this reason). Kept
    alongside served_by for backward compatibility with the simpler
    two-tier Phase 1 framing; served_by is the more precise field: "ollama"
    | "gemini" | "groq" | "cache", recording which backend in the
    three-tier failover chain actually served this response. request_id is
    the UUID main.py generated for this request - the same value it
    returns in the response - so this row can be matched back to a
    specific request instead of just a timestamp."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO requests (
                timestamp, prompt, model_used, cache_hit, latency_ms,
                load_duration_ms, eval_count, eval_duration_ms, failed_over,
                request_id, served_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                prompt[:PROMPT_LOG_CHARS],
                model_used,
                int(cache_hit),
                latency_ms,
                load_duration_ms,
                eval_count,
                eval_duration_ms,
                int(failed_over),
                request_id,
                served_by,
            ),
        )
        conn.commit()
    finally:
        conn.close()
