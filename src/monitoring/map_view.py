"""
Interactive event map (Phase 16).

Builds a Folium map of GeoWatchEvents, color-coded by evidence level so
the map itself communicates certainty, not just location. This is a
standalone function (not tied to Streamlit) so it's reusable both from
a notebook (Milestone 3) and from the dashboard (Milestone 4) without
duplicating logic.
"""

from __future__ import annotations

from typing import Optional

import folium

from src.types import EvidenceLevel, GeoWatchEvent

# Color mapping communicates evidence level visually, consistent with the
# EvidenceLevel.display_phrase() language used everywhere else in GeoWatch.
_EVIDENCE_COLORS = {
    EvidenceLevel.OBSERVED: "blue",
    EvidenceLevel.DETECTED: "orange",
    EvidenceLevel.INFERRED: "purple",
    EvidenceLevel.PREDICTED: "gray",
    EvidenceLevel.CONFIRMED: "green",
}

DEFAULT_CENTER = (0.0, 0.0)
DEFAULT_ZOOM_NO_EVENTS = 2
DEFAULT_ZOOM_WITH_EVENTS = 6


def _popup_html(event: GeoWatchEvent) -> str:
    """Build a responsibly-worded popup: uses summary_line(), never a raw label."""
    lines = [
        f"<b>{event.summary_line()}</b>",
        f"Source: {event.source}",
        f"Confidence: {event.confidence.value:.2f} ({event.confidence.basis})",
    ]
    if event.observation_time:
        lines.append(f"Observed: {event.observation_time.isoformat()}")
    if event.severity:
        lines.append(f"Severity: {event.severity}")
    return "<br>".join(lines)


def build_event_map(
    events: list[GeoWatchEvent],
    center: Optional[tuple[float, float]] = None,
    zoom_start: Optional[int] = None,
) -> folium.Map:
    """Build an interactive Folium map of GeoWatch events.

    Args:
        events: list of GeoWatchEvent to plot. An empty list is handled
            gracefully — GeoWatch must not assume continuous data
            availability (see project responsible-use principles).
        center: (latitude, longitude) to center the map on. Defaults to
            the mean position of the given events, or (0, 0) if the list
            is empty.
        zoom_start: initial zoom level. Defaults to a wide view if there
            are no events, or a regional view if there are.

    Returns:
        A folium.Map with one color-coded marker per event.
    """
    if center is None:
        if events:
            center = (
                sum(e.latitude for e in events) / len(events),
                sum(e.longitude for e in events) / len(events),
            )
        else:
            center = DEFAULT_CENTER

    if zoom_start is None:
        zoom_start = DEFAULT_ZOOM_WITH_EVENTS if events else DEFAULT_ZOOM_NO_EVENTS

    fmap = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")

    for event in events:
        color = _EVIDENCE_COLORS.get(event.evidence_level, "gray")
        folium.CircleMarker(
            location=(event.latitude, event.longitude),
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(_popup_html(event), max_width=300),
            tooltip=event.summary_line(),
        ).add_to(fmap)

    return fmap
