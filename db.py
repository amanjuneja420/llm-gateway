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
                eval_duration_ms REAL
            )
            """
        )
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
    db_path: str = DB_PATH,
) -> None:
    """Log one request. model_used is 'cache' on a cache hit, otherwise the
    actual Ollama model name (e.g. 'qwen2.5:1.5b'). load_duration_ms,
    eval_count, and eval_duration_ms come straight from Ollama's own
    response payload and are left NULL on cache hits, since no model call
    was made."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO requests (
                timestamp, prompt, model_used, cache_hit, latency_ms,
                load_duration_ms, eval_count, eval_duration_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        conn.commit()
    finally:
        conn.close()
