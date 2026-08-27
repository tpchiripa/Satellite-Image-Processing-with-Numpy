"""
GeoWatch Streamlit dashboard (Milestone 4).

Three tabs for this milestone: Overview, Live Event Map, Fire Monitor.
Vegetation Monitor, Land Disturbance, and AOI Monitoring tabs are
roadmap items — they depend on detection modules (vegetation.py,
disturbance.py) that don't exist yet; adding empty tabs for them now
would misrepresent capability GeoWatch doesn't have.

Data flow, deliberately layered so the dashboard never talks to FIRMS
or PostGIS directly:

    FIRMSProvider (src/ingestion/firms.py)
        -> GeoWatchEvent objects
        -> PostgresEventStore (src/monitoring/postgres_store.py), or
           InMemoryEventStore if no database is configured
        -> this dashboard, which only ever reads through EventStore's
           interface and renders via map_view.build_event_map()

Run with:
    streamlit run app/dashboard.py

Environment variables (see .env.example):
    FIRMS_MAP_KEY   optional; without it, the dashboard shows an empty
                    state with instructions rather than crashing.
    DATABASE_URL    optional; without it, falls back to a
                    session-scoped InMemoryEventStore (no persistence
                    between dashboard restarts).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# `streamlit run app/dashboard.py` puts only this file's own directory
# (app/) on sys.path, not the repo root -- so `from src...` below would
# fail with ModuleNotFoundError regardless of the working directory the
# command is run from. This mirrors what conftest.py does for pytest,
# which doesn't apply here since Streamlit doesn't use conftest.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.detection.wildfire import BurnSeverity
from src.geospatial.aoi import AOI
from src.ingestion.firms import FIRMSError, FIRMSProvider
from src.monitoring.events import InMemoryEventStore
from src.monitoring.map_view import build_event_map
from src.monitoring.timeseries import TrendDirection, analyze_event_time_series
from src.reporting.reports import generate_intelligence_report
from src.types import EventType, GeoWatchEvent

DEFAULT_AOI = AOI(label="Southern Africa (default AOI)", west=10.0, south=-35.0, east=40.0, north=-10.0)


# --- Resource setup (cached across reruns within a session) -------------


@st.cache_resource
def get_event_store(database_url: str | None):
    """Return a PostgresEventStore if database_url is set and reachable,
    otherwise fall back to an in-memory store. Cached so the dashboard
    doesn't reconnect on every widget interaction — Streamlit reruns the
    whole script on every user action, so uncached setup would reconnect
    constantly. Takes database_url as an explicit argument (rather than
    reading os.environ internally) so Streamlit's cache is correctly
    keyed on it — an argument-less cached function would return the same
    cached resource forever, even if the environment changed."""
    if database_url:
        try:
            from src.monitoring.postgres_store import PostgresEventStore

            store = PostgresEventStore(database_url)
            store.count()  # cheap query to confirm the connection actually works
            return store, True
        except Exception:
            pass
    return InMemoryEventStore(), False


@st.cache_resource
def get_firms_provider(map_key: str | None):
    """Return a FIRMSProvider if map_key is set, else None. Takes map_key
    as an explicit argument for the same cache-correctness reason as
    get_event_store() above."""
    if not map_key:
        return None
    try:
        return FIRMSProvider(map_key=map_key)
    except FIRMSError:
        return None


# --- Data operations ------------------------------------------------------


def fetch_and_store_live_fires(aoi: AOI, days: int, sensor: str) -> tuple[int, str]:
    """Query FIRMS for the given AOI/day range, convert to events, store
    any not already present. Returns (new_event_count, message)."""
    provider = get_firms_provider(os.environ.get("FIRMS_MAP_KEY"))
    if provider is None:
        return 0, "No FIRMS_MAP_KEY configured — see .env.example to enable live data."

    store, _ = get_event_store(os.environ.get("DATABASE_URL"))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    try:
        rows = provider.search_observations(
            bbox=aoi.as_bbox_tuple(), start_date=start, end_date=end, sensor=sensor
        )
    except FIRMSError as exc:
        return 0, f"FIRMS query failed: {exc}"

    events = provider.rows_to_events(rows, sensor=sensor)
    added = 0
    for event in events:
        try:
            store.add_event(event)
            added += 1
        except ValueError:
            pass  # already stored from a previous fetch
    return added, f"Fetched {len(events)} detection(s), {added} new."


def get_wildfire_events(store) -> list[GeoWatchEvent]:
    """All stored events of type WILDFIRE, newest observation first."""
    events = [e for e in store.list_events() if e.event_type == EventType.WILDFIRE]
    events.sort(key=lambda e: e.observation_time or e.detected_at, reverse=True)
    return events


# --- Tabs -------------------------------------------------------------


def render_overview_tab(store, db_connected: bool) -> None:
    st.subheader("System overview")

    events = store.list_events()
    wildfire_events = [e for e in events if e.event_type == EventType.WILDFIRE]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total events stored", len(events))
    col2.metric("Active fire observations", len(wildfire_events))
    col3.metric("Monitored AOIs", 1)  # single default AOI in this milestone
    col4.metric(
        "Storage backend",
        "PostgreSQL/PostGIS" if db_connected else "In-memory (not persistent)",
    )

    if not db_connected:
        st.warning(
            "No database connection configured (or connection failed) — using a "
            "temporary in-memory store. Events will not persist across dashboard "
            "restarts. Set DATABASE_URL and run `docker compose up -d db` to "
            "enable persistence. See .env.example."
        )

    if wildfire_events:
        latest = wildfire_events[0]
        latest_time = latest.observation_time or latest.detected_at
        st.caption(f"Most recent observation: {latest_time.strftime('%Y-%m-%d %H:%M UTC')}")
    else:
        st.info(
            "No events stored yet. Go to the **Fire Monitor** tab and click "
            "**Fetch latest fire data** to pull observations from NASA FIRMS."
        )

    st.divider()
    st.subheader("Intelligence report")
    st.caption(
        "Summarizes all currently stored events — counts, confidence, evidence "
        "levels — with limitations text generated from the actual data included, "
        "not a fixed disclaimer. See src/reporting/reports.py."
    )

    if not events:
        st.info("Generate a report once at least one event is stored.")
        return

    report = generate_intelligence_report(
        aoi=DEFAULT_AOI,
        period_start=min(e.observation_time or e.detected_at for e in events),
        period_end=max(e.observation_time or e.detected_at for e in events),
        events=events,
    )

    col1, col2 = st.columns(2)
    col1.metric("Report covers", f"{report.total_events} event(s)")
    col2.metric("Average confidence", f"{report.average_confidence:.2f}" if report.average_confidence else "—")

    with st.expander("Limitations (always shown alongside any figures above)"):
        for limitation in report.limitations:
            st.caption(f"• {limitation}")

    dl_col1, dl_col2 = st.columns(2)
    dl_col1.download_button(
        "Download report (JSON)",
        data=report.to_json(),
        file_name=f"geowatch_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )
    dl_col2.download_button(
        "Download event table (CSV)",
        data=report.to_csv(),
        file_name=f"geowatch_events_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


def render_live_map_tab(store) -> None:
    st.subheader("Live event map")
    st.caption(
        "Markers are color-coded by evidence level — blue means OBSERVED "
        "(a raw satellite detection), matching the language used throughout "
        "GeoWatch. See the README's Responsible Use section."
    )

    events = store.list_events()

    col1, col2 = st.columns(2)
    with col1:
        event_type_filter = st.multiselect(
            "Event type",
            options=[t.value for t in EventType],
            default=[EventType.WILDFIRE.value] if events else [],
        )
    with col2:
        min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)

    filtered = [
        e
        for e in events
        if (not event_type_filter or e.event_type.value in event_type_filter)
        and e.confidence.value >= min_confidence
    ]

    st.caption(f"Showing {len(filtered)} of {len(events)} stored event(s).")

    fmap = build_event_map(filtered)
    st.iframe(fmap._repr_html_(), height=520)


def render_fire_monitor_tab(store) -> None:
    st.subheader("Fire monitor")

    with st.expander("Fetch latest fire data from NASA FIRMS", expanded=bool(os.environ.get("FIRMS_MAP_KEY"))):
        provider = get_firms_provider(os.environ.get("FIRMS_MAP_KEY"))
        if provider is None:
            st.info(
                "No FIRMS_MAP_KEY configured. Get a free key at "
                "https://firms.modaps.eosdis.nasa.gov/api/area/ and set it as "
                "an environment variable to enable live fetching — see .env.example."
            )
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                days = st.number_input("Days back", min_value=1, max_value=10, value=1)
            with col2:
                sensor = st.selectbox(
                    "Sensor",
                    options=["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT"],
                )
            with col3:
                st.write("")
                st.write("")
                fetch_clicked = st.button("Fetch latest fire data", type="primary")

            if fetch_clicked:
                with st.spinner("Querying NASA FIRMS..."):
                    added, message = fetch_and_store_live_fires(DEFAULT_AOI, int(days), sensor)
                if added > 0:
                    st.success(message)
                else:
                    st.info(message)
                st.rerun()

    wildfire_events = get_wildfire_events(store)

    if not wildfire_events:
        st.info("No active-fire observations stored yet.")
        return

    st.caption(f"{len(wildfire_events)} stored active-fire observation(s).")

    table_rows = []
    for e in wildfire_events:
        obs_time = e.observation_time or e.detected_at
        table_rows.append(
            {
                "Observed (UTC)": obs_time.strftime("%Y-%m-%d %H:%M"),
                "Latitude": round(e.latitude, 4),
                "Longitude": round(e.longitude, 4),
                "Confidence": round(e.confidence.value, 2),
                "Source": e.source,
                "FRP (MW)": e.metadata.get("fire_radiative_power_mw", "—"),
                "Evidence level": e.evidence_level.value,
            }
        )
    st.dataframe(table_rows, width="stretch", hide_index=True)

    avg_confidence = sum(e.confidence.value for e in wildfire_events) / len(wildfire_events)
    st.caption(f"Average confidence across stored observations: {avg_confidence:.2f}")

    st.divider()
    st.subheader("Activity over time")
    st.caption(
        "Daily detection counts from stored observations. Trend direction is a "
        "simple heuristic (comparing the first half of the period to the second), "
        "not a statistical test — see src/monitoring/timeseries.py."
    )

    obs_times = [e.observation_time or e.detected_at for e in wildfire_events]
    period_start = min(obs_times)
    period_end = max(obs_times) + timedelta(days=1)  # inclusive of the last observation's day

    ts_summary = analyze_event_time_series(wildfire_events, period_start, period_end, bucket_days=1)

    if len(ts_summary.points) >= 2:
        chart_data = {
            p.period_start.strftime("%Y-%m-%d"): p.event_count for p in ts_summary.points
        }
        st.bar_chart(chart_data)

        trend_labels = {
            TrendDirection.INCREASING: "📈 Increasing",
            TrendDirection.DECREASING: "📉 Decreasing",
            TrendDirection.STABLE: "➡️ Stable",
            TrendDirection.INSUFFICIENT_DATA: "Not enough data for a trend (need 4+ days)",
        }
        st.caption(
            f"Trend: {trend_labels[ts_summary.trend_direction]} "
            f"— based on {len(ts_summary.points)} day(s) of data."
        )
    else:
        st.caption("All observations fall on a single day — nothing to chart yet.")


# --- Entry point ---------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="GeoWatch", page_icon="🛰️", layout="wide")
    st.title("🛰️ GeoWatch")
    st.caption("Open satellite environmental intelligence platform — monitoring our changing planet.")

    store, db_connected = get_event_store(os.environ.get("DATABASE_URL"))

    tab_overview, tab_map, tab_fire = st.tabs(["Overview", "Live Event Map", "Fire Monitor"])
    with tab_overview:
        render_overview_tab(store, db_connected)
    with tab_map:
        render_live_map_tab(store)
    with tab_fire:
        render_fire_monitor_tab(store)


if __name__ == "__main__":
    main()
