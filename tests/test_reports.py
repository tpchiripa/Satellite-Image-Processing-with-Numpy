"""Tests for src/reporting/reports.py"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from src.detection.wildfire import SeverityThresholds, detect_wildfire
from src.geospatial.aoi import AOI
from src.reporting.reports import generate_intelligence_report
from src.types import ConfidenceScore, EventStatus, EventType, EvidenceLevel, GeoWatchEvent


def _make_event(
    event_id: str,
    confidence: float = 0.6,
    event_type: EventType = EventType.WILDFIRE,
    source: str = "NASA FIRMS (VIIRS_NOAA20_NRT)",
) -> GeoWatchEvent:
    return GeoWatchEvent(
        event_id=event_id,
        event_type=event_type,
        latitude=-18.5,
        longitude=16.2,
        detected_at=datetime(2026, 8, 25, 1, 0, tzinfo=timezone.utc),
        observation_time=datetime(2026, 8, 25, 1, 0, tzinfo=timezone.utc),
        source=source,
        evidence_level=EvidenceLevel.OBSERVED,
        confidence=ConfidenceScore(value=confidence, basis="test", evidence_level=EvidenceLevel.OBSERVED),
        status=EventStatus.NEW,
    )


DEFAULT_AOI = AOI(label="Test AOI", west=10.0, south=-35.0, east=40.0, north=-10.0)
PERIOD_START = datetime(2026, 8, 24, tzinfo=timezone.utc)
PERIOD_END = datetime(2026, 8, 25, tzinfo=timezone.utc)


class TestGenerateIntelligenceReportBasics:
    def test_empty_events_produces_valid_report(self) -> None:
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[])
        assert report.total_events == 0
        assert report.average_confidence is None
        assert report.highest_confidence_event_id is None
        assert report.events == []

    def test_aoi_fields_captured(self) -> None:
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[])
        assert report.aoi_label == "Test AOI"
        assert report.aoi_bbox == (10.0, -35.0, 40.0, -10.0)

    def test_period_captured_as_iso_strings(self) -> None:
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[])
        assert report.period_start == PERIOD_START.isoformat()
        assert report.period_end == PERIOD_END.isoformat()

    def test_generated_at_is_recent(self) -> None:
        before = datetime.now(timezone.utc)
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[])
        generated = datetime.fromisoformat(report.generated_at)
        assert generated >= before


class TestEventAggregation:
    def test_total_events_matches_input(self) -> None:
        events = [_make_event("e1"), _make_event("e2"), _make_event("e3")]
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events)
        assert report.total_events == 3

    def test_events_by_type_counted_correctly(self) -> None:
        events = [_make_event("e1"), _make_event("e2")]
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events)
        assert report.events_by_type == {"wildfire": 2}

    def test_events_by_evidence_level_counted_correctly(self) -> None:
        events = [_make_event("e1"), _make_event("e2")]
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events)
        assert report.events_by_evidence_level == {"observed": 2}

    def test_average_confidence_computed(self) -> None:
        events = [_make_event("e1", confidence=0.4), _make_event("e2", confidence=0.8)]
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events)
        assert report.average_confidence == pytest.approx(0.6)

    def test_highest_confidence_event_identified(self) -> None:
        events = [_make_event("e1", confidence=0.3), _make_event("e2", confidence=0.9)]
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events)
        assert report.highest_confidence_event_id == "e2"

    def test_event_summaries_included(self) -> None:
        events = [_make_event("e1", confidence=0.7)]
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events)
        assert len(report.events) == 1
        assert report.events[0].event_id == "e1"
        assert report.events[0].confidence == 0.7
        assert "Observed:" in report.events[0].summary


class TestWildfireResultIntegration:
    def test_requires_pixel_resolution_when_result_provided(self) -> None:
        dnbr = np.array([[0.5, 0.05], [0.9, 0.05]])
        result = detect_wildfire(dnbr)
        with pytest.raises(ValueError):
            generate_intelligence_report(
                DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[], wildfire_result=result
            )

    def test_burned_area_computed_from_result(self) -> None:
        dnbr = np.ones((10, 10)) * 0.9  # all high severity
        result = detect_wildfire(dnbr)
        report = generate_intelligence_report(
            DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[], wildfire_result=result, pixel_resolution_m=10.0
        )
        # 100 pixels * 100 m^2 / 10000 = 1.0 hectare, 100% affected
        assert report.burned_area_hectares == pytest.approx(1.0)
        assert report.burned_area_percentage == pytest.approx(100.0)

    def test_severity_breakdown_included(self) -> None:
        dnbr = np.array([0.05, 0.9])  # one unburned, one high
        result = detect_wildfire(dnbr)
        report = generate_intelligence_report(
            DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[], wildfire_result=result, pixel_resolution_m=10.0
        )
        assert report.severity_breakdown["unburned"] == 1
        assert report.severity_breakdown["high"] == 1

    def test_custom_thresholds_reflected_in_report(self) -> None:
        custom = SeverityThresholds(unburned_max=0.05, low_max=0.15, moderate_low_max=0.3, moderate_high_max=0.5)
        dnbr = np.array([0.2])
        result = detect_wildfire(dnbr, thresholds=custom)
        report = generate_intelligence_report(
            DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[], wildfire_result=result, pixel_resolution_m=10.0
        )
        assert report.severity_thresholds_used["low_max"] == 0.15

    def test_no_wildfire_result_leaves_severity_fields_none(self) -> None:
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[])
        assert report.burned_area_hectares is None
        assert report.severity_breakdown is None


class TestLimitations:
    def test_limitations_nonempty_even_with_no_events(self) -> None:
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[])
        assert len(report.limitations) > 0

    def test_limitations_mention_source_when_events_present(self) -> None:
        events = [_make_event("e1", source="NASA FIRMS (VIIRS_NOAA20_NRT)")]
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events)
        assert any("NASA FIRMS" in lim for lim in report.limitations)

    def test_limitations_mention_thresholds_when_severity_result_present(self) -> None:
        dnbr = np.array([0.5])
        result = detect_wildfire(dnbr)
        report = generate_intelligence_report(
            DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[], wildfire_result=result, pixel_resolution_m=10.0
        )
        assert any("dNBR thresholds" in lim for lim in report.limitations)

    def test_limitations_never_omit_scope_disclaimer(self) -> None:
        # The "wildfire monitoring only" disclaimer must always be present,
        # regardless of what other data is included.
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[])
        assert any("only wildfire monitoring" in lim for lim in report.limitations)

    def test_no_confirmed_evidence_level_claims(self) -> None:
        # Guards against ever silently upgrading language to sound more
        # certain than the underlying evidence supports.
        events = [_make_event("e1")]
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events)
        report_text = report.to_json()
        assert '"evidence_level": "confirmed"' not in report_text.lower().replace(" ", "")


class TestSerialization:
    def test_to_dict_returns_plain_dict(self) -> None:
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[_make_event("e1")])
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["total_events"] == 1

    def test_to_json_roundtrips(self) -> None:
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[_make_event("e1")])
        parsed = json.loads(report.to_json())
        assert parsed["aoi_label"] == "Test AOI"
        assert parsed["total_events"] == 1

    def test_to_json_is_valid_json_with_severity_data(self) -> None:
        dnbr = np.array([0.5])
        result = detect_wildfire(dnbr)
        report = generate_intelligence_report(
            DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[], wildfire_result=result, pixel_resolution_m=10.0
        )
        # Should not raise - severity_breakdown values must be JSON-safe ints
        json.loads(report.to_json())

    def test_to_csv_empty_events_returns_empty_string(self) -> None:
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events=[])
        assert report.to_csv() == ""

    def test_to_csv_contains_event_rows(self) -> None:
        events = [_make_event("e1"), _make_event("e2")]
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events)
        csv_text = report.to_csv()
        assert "e1" in csv_text
        assert "e2" in csv_text
        assert "event_id" in csv_text  # header present

    def test_to_csv_row_count_matches_event_count(self) -> None:
        events = [_make_event(f"e{i}") for i in range(5)]
        report = generate_intelligence_report(DEFAULT_AOI, PERIOD_START, PERIOD_END, events)
        lines = [l for l in report.to_csv().strip().split("\n") if l]
        assert len(lines) == 6  # header + 5 rows
