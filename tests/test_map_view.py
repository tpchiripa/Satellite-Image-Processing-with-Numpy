"""Tests for src/monitoring/map_view.py"""

from __future__ import annotations

from datetime import datetime, timezone

import folium
import pytest

from src.monitoring.map_view import build_event_map
from src.types import ConfidenceScore, EvidenceLevel, EventType, GeoWatchEvent


def _make_event(event_id: str, lat: float, lon: float, evidence_level=EvidenceLevel.OBSERVED) -> GeoWatchEvent:
    return GeoWatchEvent(
        event_id=event_id,
        event_type=EventType.WILDFIRE,
        latitude=lat,
        longitude=lon,
        detected_at=datetime.now(timezone.utc),
        observation_time=datetime.now(timezone.utc),
        source="test-fixture",
        evidence_level=evidence_level,
        confidence=ConfidenceScore(value=0.8, basis="test", evidence_level=evidence_level),
    )


class TestBuildEventMap:
    def test_returns_folium_map(self) -> None:
        result = build_event_map([])
        assert isinstance(result, folium.Map)

    def test_empty_event_list_does_not_crash(self) -> None:
        fmap = build_event_map([])
        html = fmap.get_root().render()
        assert html  # renders successfully with no events

    def test_map_centers_on_event_mean_position(self) -> None:
        events = [_make_event("e1", 0.0, 0.0), _make_event("e2", 10.0, 10.0)]
        fmap = build_event_map(events)
        assert fmap.location == [5.0, 5.0]

    def test_explicit_center_overrides_default(self) -> None:
        events = [_make_event("e1", 0.0, 0.0)]
        fmap = build_event_map(events, center=(50.0, 50.0))
        assert fmap.location == [50.0, 50.0]

    def test_markers_present_in_rendered_html(self) -> None:
        events = [_make_event("e1", -33.9, 18.4)]
        fmap = build_event_map(events)
        html = fmap.get_root().render()
        assert "-33.9" in html
        assert "18.4" in html

    def test_summary_line_appears_in_popup(self) -> None:
        events = [_make_event("e1", -33.9, 18.4, evidence_level=EvidenceLevel.OBSERVED)]
        fmap = build_event_map(events)
        html = fmap.get_root().render()
        assert "wildfire" in html.lower()

    def test_multiple_events_produce_multiple_markers(self) -> None:
        events = [
            _make_event("e1", -33.9, 18.4),
            _make_event("e2", -34.0, 18.5),
            _make_event("e3", -34.1, 18.6),
        ]
        fmap = build_event_map(events)
        html = fmap.get_root().render()
        # Each event's distinct longitude should appear once as a marker coordinate.
        assert html.count("18.4") >= 1
        assert html.count("18.5") >= 1
        assert html.count("18.6") >= 1

    def test_no_events_uses_wide_default_zoom(self) -> None:
        fmap = build_event_map([])
        assert fmap.options["zoom"] == 2

    def test_events_use_regional_default_zoom(self) -> None:
        fmap = build_event_map([_make_event("e1", -33.9, 18.4)])
        assert fmap.options["zoom"] == 6
