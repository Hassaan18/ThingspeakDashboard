"""
Interactive ThingSpeak dashboard for the Amertate channel.
Deployable as-is on Streamlit Community Cloud.

Local run:
    streamlit run streamlit_app.py

Requires a ThingSpeak read API key, provided via Streamlit secrets:
    .streamlit/secrets.toml (local)          -> THINGSPEAK_READ_API_KEY = "..."
    Streamlit Cloud "Secrets" settings (prod) -> same key/value
"""

import time
from datetime import datetime, time as dt_time, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

CHANNEL_ID = "3240736"
BASE_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}"

# ThingSpeak silently truncates any single feeds.json request to this many rows,
# regardless of the start/end range requested.
PAGE_SIZE_CAP = 8000

# ThingSpeak's API works in UTC; this is the timezone shown to the user everywhere
# else (input pickers, displayed timestamps).
LOCAL_TZ = "Europe/Helsinki"

st.set_page_config(page_title="Amertate ThingSpeak Dashboard", layout="wide")

READ_API_KEY = st.secrets.get("THINGSPEAK_READ_API_KEY")
if not READ_API_KEY:
    st.error(
        "Missing ThingSpeak API key. Add THINGSPEAK_READ_API_KEY to "
        ".streamlit/secrets.toml (local) or this app's Secrets settings (Streamlit Cloud)."
    )
    st.stop()


def _format_for_api(ts: pd.Timestamp) -> str:
    """ThingSpeak's start/end params are interpreted as UTC, so convert first."""
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def fetch_channel_info() -> dict:
    """Get channel metadata, including the real names of field1..field8."""
    resp = requests.get(f"{BASE_URL}/feeds.json", params={"api_key": READ_API_KEY, "results": 1})
    resp.raise_for_status()
    channel = resp.json()["channel"]
    return {f"field{i}": channel[f"field{i}"] for i in range(1, 9) if f"field{i}" in channel}


def fetch_feeds(start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> list[dict]:
    """Fetch a single page of feed entries for the [start, end] window."""
    params = {"api_key": READ_API_KEY}
    if start is not None:
        params["start"] = _format_for_api(start)
    if end is not None:
        params["end"] = _format_for_api(end)

    resp = requests.get(f"{BASE_URL}/feeds.json", params=params)
    resp.raise_for_status()
    return resp.json()["feeds"]


def fetch_all_feeds(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """
    Fetch every feed entry in [start, end], paging backwards past ThingSpeak's
    per-request row cap. Each page is requested with the same `start` and a
    shrinking `end`, so pages are pulled from newest to oldest and merged,
    deduping by entry_id (the boundary timestamp is re-requested on purpose
    to avoid losing entries that share the same second-resolution timestamp).
    """
    feeds_by_id: dict[int, dict] = {}
    current_end = end

    while True:
        page = fetch_feeds(start=start, end=current_end)
        if not page:
            break

        new_count = 0
        for feed in page:
            entry_id = feed["entry_id"]
            if entry_id not in feeds_by_id:
                feeds_by_id[entry_id] = feed
                new_count += 1

        if len(page) < PAGE_SIZE_CAP:
            break  # this page wasn't capped, so we've reached the `start` boundary

        earliest = min(pd.to_datetime(f["created_at"]) for f in page)
        if earliest >= current_end or new_count == 0:
            break  # safety: no forward progress, avoid an infinite loop

        current_end = earliest
        time.sleep(0.2)  # be polite to the API when paging many chunks

    return sorted(feeds_by_id.values(), key=lambda f: f["entry_id"])


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discard known-bad sensor readings (turn them into NaN so the rest of the
    row is kept). Matched by field display name, since field numbers vary by
    channel:
      - Gen RPM: values above 300 are implausible for this generator.
      - Wind Speed / Wind Direction: -1 is the sensor's error/no-reading value.
    """
    for col in df.columns:
        lower = col.lower()
        if "rpm" in lower:
            df.loc[df[col] > 300, col] = float("nan")
        elif "wind" in lower and ("spd" in lower or "speed" in lower or "dir" in lower):
            df.loc[df[col] == -1, col] = float("nan")
    return df


def build_dataframe(feeds: list[dict], field_names: dict) -> pd.DataFrame:
    df = pd.DataFrame(feeds)
    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_convert(LOCAL_TZ)
    df = df.rename(columns=field_names)

    value_cols = list(field_names.values())
    for col in value_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("created_at").sort_index()
    df = df[[c for c in value_cols if c in df.columns]]
    return apply_quality_filters(df)


@st.cache_data(ttl=60)
def load_field_names() -> dict:
    return fetch_channel_info()


@st.cache_data(ttl=60)
def load_data(start: pd.Timestamp, end: pd.Timestamp, field_names: dict) -> pd.DataFrame:
    feeds = fetch_all_feeds(start, end)
    return build_dataframe(feeds, field_names)


st.title("Amertate ThingSpeak Dashboard")

field_names = load_field_names()
all_fields = list(field_names.values())

with st.sidebar:
    st.header("Filters")
    st.caption("All times are Finland time (Europe/Helsinki).")

    now_local = pd.Timestamp.now(tz=LOCAL_TZ)
    today = now_local.date()
    six_hours_ago = now_local - timedelta(hours=6)

    if "default_start_date" not in st.session_state:
        st.session_state.default_start_date = six_hours_ago.date()
    if "default_start_time" not in st.session_state:
        st.session_state.default_start_time = six_hours_ago.time().replace(microsecond=0)
    if "default_end_date" not in st.session_state:
        st.session_state.default_end_date = today
    if "default_end_time" not in st.session_state:
        st.session_state.default_end_time = now_local.time().replace(microsecond=0)

    st.caption("Start")
    start_col1, start_col2 = st.columns(2)
    start_date = start_col1.date_input(
        "Start date",
        value=st.session_state.default_start_date,
        max_value=today,
        key="start_date",
        label_visibility="collapsed",
    )
    start_time = start_col2.time_input(
        "Start time",
        value=st.session_state.default_start_time,
        key="start_time",
        label_visibility="collapsed",
    )

    st.caption("End")
    end_col1, end_col2 = st.columns(2)
    end_date = end_col1.date_input(
        "End date",
        value=st.session_state.default_end_date,
        max_value=today,
        key="end_date",
        label_visibility="collapsed",
    )
    end_time = end_col2.time_input(
        "End time",
        value=st.session_state.default_end_time,
        key="end_time",
        label_visibility="collapsed",
    )

    selected_fields = st.multiselect(
        "Fields to display",
        options=all_fields,
        default=all_fields,
    )

    chart_mode = st.radio("Chart layout", ["Stacked (own scale)", "Overlaid (one chart)"])

    refresh = st.button("Refresh data", use_container_width=True)

if refresh:
    load_data.clear()

start_ts = pd.Timestamp(datetime.combine(start_date, start_time), tz=LOCAL_TZ)
end_ts = pd.Timestamp(datetime.combine(end_date, end_time), tz=LOCAL_TZ)

if end_ts <= start_ts:
    st.sidebar.error("End must be after start.")
    st.stop()

with st.spinner("Fetching data from ThingSpeak..."):
    df = load_data(start_ts, end_ts, field_names)

if df.empty:
    st.warning("No data returned for the selected date range.")
    st.stop()

if not selected_fields:
    st.info("Select at least one field in the sidebar to see charts.")
    st.stop()

st.caption(f"{len(df):,} readings from {df.index.min()} to {df.index.max()}")

col1, col2, col3 = st.columns(3)
col1.metric("Readings", f"{len(df):,}")
col2.metric("First reading", str(df.index.min()))
col3.metric("Last reading", str(df.index.max()))

visible_df = df[selected_fields]

if chart_mode == "Stacked (own scale)":
    fig = make_subplots(
        rows=len(selected_fields),
        cols=1,
        shared_xaxes=True,
        subplot_titles=selected_fields,
        vertical_spacing=0.4 / max(len(selected_fields), 1),
    )
    for i, col in enumerate(selected_fields, start=1):
        fig.add_trace(
            go.Scatter(
                x=visible_df.index,
                y=visible_df[col],
                mode="lines+markers",
                name=col,
                line=dict(width=1, dash="dot"),
                marker=dict(size=6),
            ),
            row=i,
            col=1,
        )
    fig.update_layout(height=250 * len(selected_fields), showlegend=False)
    fig.update_xaxes(showticklabels=True)
else:
    fig = go.Figure()
    for col in selected_fields:
        fig.add_trace(
            go.Scatter(
                x=visible_df.index,
                y=visible_df[col],
                mode="lines+markers",
                name=col,
                line=dict(width=1, dash="dot"),
                marker=dict(size=6),
            )
        )
    fig.update_layout(height=600, legend=dict(orientation="h", y=1.02))

st.plotly_chart(fig, use_container_width=True)

with st.expander("Summary statistics"):
    st.dataframe(visible_df.describe().T, use_container_width=True)

with st.expander("Raw data"):
    st.dataframe(visible_df, use_container_width=True)
    st.download_button(
        "Download CSV",
        visible_df.to_csv().encode("utf-8"),
        file_name="thingspeak_data.csv",
        mime="text/csv",
    )
