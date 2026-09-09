"""
Read-only dashboard over gateway.db.
======================================
Run with:

    venv\\Scripts\\streamlit run dashboard.py

Reads gateway.db directly - no dependency on the gateway server being up,
no writes, no new columns or migrations. Everything shown here is derivable
from the same `requests` table every other script in this project already
queries (benchmark.py's summary, cost_estimate.py, analyze_ollama_stats.py).

No new dependencies beyond streamlit itself, per the brief - pandas is
imported below only because Streamlit already depends on it (it's what
st.bar_chart/st.line_chart actually consume under the hood), not because
this file chose to add it as a separate dependency.

Two things this dashboard deliberately does NOT paper over, stated here so
reading the charts doesn't require re-deriving them from main.py/db.py:

1. **`served_by` is NULL for every row logged before Phase 2's failover
   chain existed** (that column was added by a later schema migration -
   see db.py's `_MIGRATED_COLUMNS`). The committed reference gateway.db
   (Run 3) predates it entirely, so its "model usage split" chart below
   will show 100% of rows as "not recorded (pre-migration)", not because
   something is broken, but because that's what actually happened. Point
   this dashboard at a gateway.db that has real Phase 2+ traffic in it
   (e.g. one of the runs/gateway_run_*.db archives from after Phase 2) to
   see a populated ollama/gemini/groq split.
2. **Rate-limited (429) requests are never logged to gateway.db at all**
   - see main.py's chat(): the 429 is raised before log_request() is ever
   reached, the same as the all-backends-failed 502 path. There is no
   query against this table that can recover a real count of rate-limited
   requests; the metric below is always a labeled zero explaining why,
   not a real (and currently-zero-by-coincidence) count. Getting a real
   number would mean adding a second log call to chat()'s 429 branch,
   which isn't something to do silently as a side effect of building a
   read-only dashboard - flagged instead of faked.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = "gateway.db"

st.set_page_config(page_title="LLM Gateway Dashboard", layout="wide")
st.title("LLM Gateway — request log dashboard")
st.caption(f"Reading directly from `{DB_PATH}` — read-only, no writes, no server required.")


@st.cache_data(ttl=5)
def load_data(db_path: str) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM requests ORDER BY id", conn)
    finally:
        conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


df = load_data(DB_PATH)

if df.empty:
    st.warning(f"No rows found in `{DB_PATH}` — run benchmark.py or send a few /chat requests first.")
    st.stop()

st.caption(f"{len(df)} rows loaded, spanning {df['timestamp'].min()} to {df['timestamp'].max()}")

# --- Top-line metrics -----------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
total_requests = len(df)
cache_hit_rate = 100.0 * df["cache_hit"].sum() / total_requests
avg_latency_ms = df["latency_ms"].mean()

col1.metric("Total requests logged", total_requests)
col2.metric("Cache hit rate", f"{cache_hit_rate:.1f}%")
col3.metric("Avg latency (all rows)", f"{avg_latency_ms:,.0f} ms")
col4.metric(
    "Rate-limited (429) requests logged",
    "0 (by design)",
    help=(
        "429s are rejected before log_request() is ever called in main.py's "
        "chat() - the same as the all-backends-failed 502 path. This number "
        "can never be anything but 0 from gateway.db alone; it is not a real "
        "measurement that happens to be zero. See this file's module "
        "docstring."
    ),
)

st.divider()

# --- Request volume over time ---------------------------------------------
st.subheader("Request volume over time")
span = df["timestamp"].max() - df["timestamp"].min()
if span < pd.Timedelta(hours=2):
    freq, freq_label = "1min", "minute"
elif span < pd.Timedelta(days=2):
    freq, freq_label = "1h", "hour"
else:
    freq, freq_label = "1D", "day"
volume = df.set_index("timestamp").resample(freq).size().rename("requests")
st.caption(f"Bucketed by {freq_label} (chosen automatically from this data's {span} span).")
st.bar_chart(volume)

st.divider()

left, right = st.columns(2)

# --- Model usage split ------------------------------------------------------
with left:
    st.subheader("Model usage split (model_used)")
    model_counts = df["model_used"].value_counts()
    st.bar_chart(model_counts)

    st.subheader("served_by breakdown")
    served_by_display = df["served_by"].fillna("not recorded (pre-migration)")
    served_by_counts = served_by_display.value_counts()
    st.bar_chart(served_by_counts)
    if (df["served_by"].isna()).all():
        st.caption(
            "Every row here predates the served_by column (added in Phase 2's "
            "failover chain) - this isn't broken, it's what actually happened "
            "for this specific gateway.db. See this file's module docstring."
        )

# --- Cache hit rate + latency by model --------------------------------------
with right:
    st.subheader("Cache hit rate")
    hit_counts = df["cache_hit"].map({1: "hit", 0: "miss"}).value_counts()
    st.bar_chart(hit_counts)

    st.subheader("Average latency by model_used")
    avg_by_model = df.groupby("model_used")["latency_ms"].mean().sort_values(ascending=False)
    st.bar_chart(avg_by_model)

st.divider()

# --- Raw data ---------------------------------------------------------------
with st.expander("Raw data (last 100 rows)"):
    st.dataframe(df.tail(100), use_container_width=True)
