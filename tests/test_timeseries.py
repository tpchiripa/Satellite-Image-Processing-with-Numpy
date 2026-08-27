"""Tests for src/monitoring/timeseries.py"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.monitoring.timeseries import (
    TrendDirection,
    analyze_event_time_series,
    compute_event_time_series,
    compute_trend_direction,
)
from src.types import ConfidenceScore, EventStatus, EventType, EvidenceLevel, GeoWatchEvent


def _make_event(event_id: str, observed_at: datetime, confidence: float = 0.6) -> GeoWatchEvent:
    return GeoWatchEvent(
        event_id=event_id,
        event_type=EventType.WILDFIRE,
        latitude=-18.0,
        longitude=16.0,
        detected_at=observed_at,
        observation_time=observed_at,
        source="NASA FIRMS (test)",
        evidence_level=EvidenceLevel.OBSERVED,
        confidence=ConfidenceScore(value=confidence, basis="test", evidence_level=EvidenceLevel.OBSERVED),
        status=EventStatus.NEW,
    )


BASE = datetime(2026, 8, 1, tzinfo=timezone.utc)


class TestComputeEventTimeSeries:
    def test_empty_events_produces_empty_buckets(self) -> None:
        points = compute_event_time_series([], BASE, BASE + timedelta(days=3), bucket_days=1)
        assert len(points) == 3
        assert all(p.event_count == 0 for p in points)
        assert all(p.mean_confidence is None for p in points)

    def test_events_bucketed_correctly(self) -> None:
        events = [
            _make_event("e1", BASE + timedelta(hours=1)),
            _make_event("e2", BASE + timedelta(hours=2)),
            _make_event("e3", BASE + timedelta(days=1, hours=1)),
        ]
        points = compute_event_time_series(events, BASE, BASE + timedelta(days=2), bucket_days=1)
        assert points[0].event_count == 2
        assert points[1].event_count == 1

    def test_bucket_boundaries_half_open(self) -> None:
        # An event exactly at a bucket boundary belongs to the bucket it starts, not the previous one.
        events = [_make_event("e1", BASE + timedelta(days=1))]
        points = compute_event_time_series(events, BASE, BASE + timedelta(days=2), bucket_days=1)
        assert points[0].event_count == 0
        assert points[1].event_count == 1

    def test_final_bucket_may_be_narrower(self) -> None:
        points = compute_event_time_series(
            [], BASE, BASE + timedelta(days=2, hours=12), bucket_days=1
        )
        assert len(points) == 3
        assert points[2].period_end - points[2].period_start == timedelta(hours=12)

    def test_mean_confidence_computed_per_bucket(self) -> None:
        events = [
            _make_event("e1", BASE + timedelta(hours=1), confidence=0.4),
            _make_event("e2", BASE + timedelta(hours=2), confidence=0.8),
        ]
        points = compute_event_time_series(events, BASE, BASE + timedelta(days=1), bucket_days=1)
        assert points[0].mean_confidence == pytest.approx(0.6)

    def test_evidence_level_breakdown_per_bucket(self) -> None:
        events = [_make_event("e1", BASE + timedelta(hours=1))]
        points = compute_event_time_series(events, BASE, BASE + timedelta(days=1), bucket_days=1)
        assert points[0].events_by_evidence_level == {"observed": 1}

    def test_nonpositive_bucket_days_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_event_time_series([], BASE, BASE + timedelta(days=1), bucket_days=0)

    def test_period_end_before_start_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_event_time_series([], BASE, BASE - timedelta(days=1))

    def test_multi_day_bucket_width(self) -> None:
        events = [
            _make_event("e1", BASE + timedelta(hours=1)),
            _make_event("e2", BASE + timedelta(days=2, hours=1)),
        ]
        points = compute_event_time_series(events, BASE, BASE + timedelta(days=4), bucket_days=2)
        assert len(points) == 2
        assert points[0].event_count == 1
        assert points[1].event_count == 1


class TestComputeTrendDirection:
    def test_insufficient_data_below_four_buckets(self) -> None:
        points = compute_event_time_series([], BASE, BASE + timedelta(days=3), bucket_days=1)
        assert compute_trend_direction(points) == TrendDirection.INSUFFICIENT_DATA

    def test_stable_when_all_zero(self) -> None:
        points = compute_event_time_series([], BASE, BASE + timedelta(days=6), bucket_days=1)
        assert compute_trend_direction(points) == TrendDirection.STABLE

    def test_increasing_trend_detected(self) -> None:
        events = [_make_event(f"e{i}", BASE + timedelta(days=5, hours=i)) for i in range(10)]
        points = compute_event_time_series(events, BASE, BASE + timedelta(days=6), bucket_days=1)
        assert compute_trend_direction(points) == TrendDirection.INCREASING

    def test_decreasing_trend_detected(self) -> None:
        events = [_make_event(f"e{i}", BASE + timedelta(hours=i)) for i in range(10)]
        points = compute_event_time_series(events, BASE, BASE + timedelta(days=6), bucket_days=1)
        assert compute_trend_direction(points) == TrendDirection.DECREASING

    def test_stable_when_change_within_threshold(self) -> None:
        events = (
            [_make_event(f"a{i}", BASE + timedelta(hours=i)) for i in range(5)]
            + [_make_event(f"b{i}", BASE + timedelta(days=5, hours=i)) for i in range(5)]
        )
        points = compute_event_time_series(events, BASE, BASE + timedelta(days=6), bucket_days=1)
        # Equal counts in both halves -> genuinely stable.
        assert compute_trend_direction(points) == TrendDirection.STABLE

    def test_increase_from_zero_is_increasing(self) -> None:
        events = [_make_event(f"e{i}", BASE + timedelta(days=5, hours=i)) for i in range(3)]
        points = compute_event_time_series(events, BASE, BASE + timedelta(days=6), bucket_days=1)
        assert compute_trend_direction(points) == TrendDirection.INCREASING

    def test_custom_threshold_changes_sensitivity(self) -> None:
        events = [_make_event(f"e{i}", BASE + timedelta(days=5, hours=i)) for i in range(6)] + [
            _make_event(f"f{i}", BASE + timedelta(hours=i)) for i in range(5)
        ]
        points = compute_event_time_series(events, BASE, BASE + timedelta(days=6), bucket_days=1)
        # A small change might read as stable with a high threshold, but
        # increasing with a very low one.
        assert compute_trend_direction(points, relative_change_threshold=0.01) == TrendDirection.INCREASING


class TestAnalyzeEventTimeSeries:
    def test_returns_evidence_level_detected(self) -> None:
        summary = analyze_event_time_series([], BASE, BASE + timedelta(days=4))
        assert summary.evidence_level == EvidenceLevel.DETECTED

    def test_total_events_matches_input(self) -> None:
        events = [_make_event(f"e{i}", BASE + timedelta(hours=i)) for i in range(7)]
        summary = analyze_event_time_series(events, BASE, BASE + timedelta(days=4))
        assert summary.total_events == 7

    def test_bucket_days_recorded(self) -> None:
        summary = analyze_event_time_series([], BASE, BASE + timedelta(days=4), bucket_days=2)
        assert summary.bucket_days == 2

    def test_trend_direction_included(self) -> None:
        events = [_make_event(f"e{i}", BASE + timedelta(days=5, hours=i)) for i in range(10)]
        summary = analyze_event_time_series(events, BASE, BASE + timedelta(days=6))
        assert summary.trend_direction == TrendDirection.INCREASING

    def test_points_cover_full_period(self) -> None:
        summary = analyze_event_time_series([], BASE, BASE + timedelta(days=5), bucket_days=1)
        assert len(summary.points) == 5
        assert summary.points[0].period_start == BASE
        assert summary.points[-1].period_end == BASE + timedelta(days=5)
