"""
Event time-series aggregation for an AOI (Phase 12, event-based slice).

Unlike vegetation-change tracking (which needs repeated raster imagery
GeoWatch doesn't have a live provider for yet), event time series work
on data GeoWatch already has for real: stored GeoWatchEvent objects
with real observation timestamps, most concretely NASA FIRMS fire
detections. This module is genuinely wireable into the live dashboard
today — see app/dashboard.py's Fire Monitor tab.

Two things this module is careful about, continuing the same discipline
as every other detection module:

1. TREND DIRECTION IS A SIMPLE HEURISTIC, NOT A STATISTICAL TEST.
   compute_trend_direction() compares mean event counts between the
   first and second half of the time series. It does not perform
   significance testing, does not account for satellite revisit-rate
   variation, and a short or sparse time series can easily produce a
   misleading "increasing"/"decreasing" label from noise alone. Always
   report bucket count and period length alongside any trend label —
   see TimeSeriesSummary.

2. EVIDENCE LEVEL. Aggregating and counting existing events is a
   deterministic calculation — EvidenceLevel.DETECTED, not INFERRED.
   The trend label itself is also DETECTED (it's arithmetic on the
   data), not a claim about *why* activity changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from src.types import EvidenceLevel, GeoWatchEvent


class TrendDirection(str, Enum):
    """Simple heuristic trend label — see module docstring for caveats."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class TimeSeriesPoint:
    """Event activity within one time bucket."""

    period_start: datetime
    period_end: datetime
    event_count: int
    mean_confidence: Optional[float]
    events_by_evidence_level: dict[str, int]


@dataclass
class TimeSeriesSummary:
    """Full output of a time-series analysis run.

    evidence_level is always DETECTED — see module docstring.
    trend_direction is included alongside bucket_count and
    total_events so downstream consumers can judge how much weight the
    trend label actually deserves, rather than presenting it in isolation.
    """

    points: list[TimeSeriesPoint]
    bucket_days: int
    total_events: int
    trend_direction: TrendDirection
    evidence_level: EvidenceLevel = field(default=EvidenceLevel.DETECTED)


def compute_event_time_series(
    events: list[GeoWatchEvent],
    period_start: datetime,
    period_end: datetime,
    bucket_days: int = 1,
) -> list[TimeSeriesPoint]:
    """Bucket events into fixed-width time periods and summarize each bucket.

    Args:
        events: events to bucket. Caller is responsible for having
            already filtered these to the relevant AOI/event type — this
            function buckets by time only.
        period_start: start of the overall time range to cover.
        period_end: end of the overall time range to cover. Must be
            after period_start.
        bucket_days: width of each time bucket, in days. Must be positive.

    Returns:
        List of TimeSeriesPoint, one per bucket, covering
        [period_start, period_end) in order. The final bucket may be
        narrower than bucket_days if the range doesn't divide evenly.
        Buckets with no events still appear, with event_count=0 and
        mean_confidence=None — a gap in activity is real information,
        not something to silently omit.

    Raises:
        ValueError: if bucket_days is not positive, or period_end is
            not after period_start.
    """
    if bucket_days <= 0:
        raise ValueError(f"compute_event_time_series: bucket_days must be positive, got {bucket_days}")
    if period_end <= period_start:
        raise ValueError(
            f"compute_event_time_series: period_end ({period_end}) must be after "
            f"period_start ({period_start})"
        )

    bucket_width = timedelta(days=bucket_days)
    points: list[TimeSeriesPoint] = []

    bucket_start = period_start
    while bucket_start < period_end:
        bucket_end = min(bucket_start + bucket_width, period_end)

        bucket_events = [
            e
            for e in events
            if bucket_start <= (e.observation_time or e.detected_at) < bucket_end
        ]

        mean_confidence = (
            round(sum(e.confidence.value for e in bucket_events) / len(bucket_events), 3)
            if bucket_events
            else None
        )

        evidence_breakdown: dict[str, int] = {}
        for e in bucket_events:
            evidence_breakdown[e.evidence_level.value] = (
                evidence_breakdown.get(e.evidence_level.value, 0) + 1
            )

        points.append(
            TimeSeriesPoint(
                period_start=bucket_start,
                period_end=bucket_end,
                event_count=len(bucket_events),
                mean_confidence=mean_confidence,
                events_by_evidence_level=evidence_breakdown,
            )
        )

        bucket_start = bucket_end

    return points


def compute_trend_direction(
    points: list[TimeSeriesPoint], relative_change_threshold: float = 0.2
) -> TrendDirection:
    """Compare mean event counts between the first and second half of a
    time series to produce a simple trend label.

    This is NOT a statistical trend test — see module docstring. It's a
    coarse heuristic intended for a quick dashboard summary, not a claim
    that should be treated as validated.

    Args:
        points: time series, as returned by compute_event_time_series().
            Order matters — this function does not re-sort.
        relative_change_threshold: minimum relative change between the
            first-half and second-half mean counts to call it increasing
            or decreasing, rather than stable. Defaults to 0.2 (20%).

    Returns:
        TrendDirection.INSUFFICIENT_DATA if fewer than 4 buckets are
        provided (too few to meaningfully split in half), otherwise
        INCREASING, DECREASING, or STABLE.
    """
    if len(points) < 4:
        return TrendDirection.INSUFFICIENT_DATA

    midpoint = len(points) // 2
    first_half = points[:midpoint]
    second_half = points[midpoint:]

    first_mean = sum(p.event_count for p in first_half) / len(first_half)
    second_mean = sum(p.event_count for p in second_half) / len(second_half)

    if first_mean == 0 and second_mean == 0:
        return TrendDirection.STABLE

    # Avoid division by zero when first_mean is 0 but second_mean isn't:
    # treat any activity appearing from nothing as a relative increase.
    if first_mean == 0:
        return TrendDirection.INCREASING if second_mean > 0 else TrendDirection.STABLE

    relative_change = (second_mean - first_mean) / first_mean

    if relative_change > relative_change_threshold:
        return TrendDirection.INCREASING
    if relative_change < -relative_change_threshold:
        return TrendDirection.DECREASING
    return TrendDirection.STABLE


def analyze_event_time_series(
    events: list[GeoWatchEvent],
    period_start: datetime,
    period_end: datetime,
    bucket_days: int = 1,
    relative_change_threshold: float = 0.2,
) -> TimeSeriesSummary:
    """Run the full time-series pipeline: bucket events, then compute a trend label.

    This is the main entry point consumers should use — it ties together
    compute_event_time_series() and compute_trend_direction() into a
    single auditable result.

    Args:
        events: events to analyze.
        period_start: start of the time range to cover.
        period_end: end of the time range to cover.
        bucket_days: passed to compute_event_time_series().
        relative_change_threshold: passed to compute_trend_direction().

    Returns:
        TimeSeriesSummary with the bucketed points, total event count,
        and trend direction.
    """
    points = compute_event_time_series(events, period_start, period_end, bucket_days=bucket_days)
    trend = compute_trend_direction(points, relative_change_threshold=relative_change_threshold)

    return TimeSeriesSummary(
        points=points,
        bucket_days=bucket_days,
        total_events=sum(p.event_count for p in points),
        trend_direction=trend,
    )
