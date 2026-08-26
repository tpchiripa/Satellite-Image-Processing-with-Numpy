"""
Automated intelligence report generation (Phase 18/19).

Ties together what GeoWatch actually has by Milestone 5: stored events
(currently only WILDFIRE, from NASA FIRMS), and optionally a specific
WildfireDetectionResult from imagery-based dNBR analysis (Milestone 2).
It deliberately does NOT claim capabilities GeoWatch doesn't have yet —
there is no vegetation-decline or land-disturbance section here, because
those detection modules don't exist. Adding empty/fake sections for
them would misrepresent the system, which runs directly against this
project's own Responsible Use principle.

A report's `limitations` field is generated dynamically from what data
actually went into it, not a fixed disclaimer boilerplate — a report
built only from FIRMS events gets different limitations than one that
also includes an imagery-based severity analysis.

Exports to JSON (primary format) and CSV (event table only). PDF export
is a roadmap item — see README.md.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.detection.wildfire import WildfireDetectionResult
from src.geospatial.aoi import AOI
from src.geospatial.area import AreaStats, compute_area_stats
from src.types import GeoWatchEvent


@dataclass
class EventSummary:
    """Compact, JSON-safe representation of one event for the report's event table."""

    event_id: str
    event_type: str
    evidence_level: str
    confidence: float
    latitude: float
    longitude: float
    observed_at: Optional[str]
    source: str
    summary: str


@dataclass
class IntelligenceReport:
    """A GeoWatch intelligence report for one AOI over one monitoring period.

    This is the ONLY place report content should be assembled — dashboard
    or notebook code should call generate_intelligence_report() rather
    than hand-building report-shaped dicts, so the limitations/evidence
    discipline stays centralized.
    """

    aoi_label: str
    aoi_bbox: tuple[float, float, float, float]
    period_start: str
    period_end: str
    generated_at: str

    total_events: int
    events_by_type: dict[str, int]
    events_by_evidence_level: dict[str, int]
    average_confidence: Optional[float]
    highest_confidence_event_id: Optional[str]

    burned_area_hectares: Optional[float]
    burned_area_percentage: Optional[float]
    severity_breakdown: Optional[dict[str, int]]
    severity_thresholds_used: Optional[dict[str, float]]

    events: list[EventSummary]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """Event table only, as CSV. The summary/severity fields above
        don't fit a flat event-per-row format, so they're intentionally
        left out of the CSV — use to_json() for the full report."""
        buffer = io.StringIO()
        if not self.events:
            return ""
        writer = csv.DictWriter(buffer, fieldnames=list(asdict(self.events[0]).keys()))
        writer.writeheader()
        for event in self.events:
            writer.writerow(asdict(event))
        return buffer.getvalue()


def _event_to_summary(event: GeoWatchEvent) -> EventSummary:
    observed_at = event.observation_time or event.detected_at
    return EventSummary(
        event_id=event.event_id,
        event_type=event.event_type.value,
        evidence_level=event.evidence_level.value,
        confidence=round(event.confidence.value, 3),
        latitude=event.latitude,
        longitude=event.longitude,
        observed_at=observed_at.isoformat() if observed_at else None,
        source=event.source,
        summary=event.summary_line(),
    )


def _build_limitations(
    events: list[GeoWatchEvent],
    wildfire_result: Optional[WildfireDetectionResult],
) -> list[str]:
    """Generate limitations text specific to what data this report actually
    contains, rather than a fixed boilerplate disclaimer."""
    limitations: list[str] = []

    sources = sorted({e.source for e in events})
    if sources:
        limitations.append(
            "Event data in this report comes from: " + ", ".join(sources) + ". "
            "These are near-real-time satellite detections, not an exhaustive "
            "record — detection depends on satellite overpass timing, cloud "
            "cover, and fire size/intensity relative to sensor resolution."
        )

    evidence_levels = sorted({e.evidence_level.value for e in events})
    if evidence_levels:
        limitations.append(
            "All events in this report carry evidence level(s): "
            + ", ".join(evidence_levels)
            + ". None are independently CONFIRMED. See this project's "
            "Responsible Use principles before treating any figure here "
            "as a verified fact."
        )

    if wildfire_result is not None:
        t = wildfire_result.thresholds
        limitations.append(
            f"Burn-severity classification uses dNBR thresholds "
            f"(unburned<={t.unburned_max}, low<={t.low_max}, "
            f"moderate_low<={t.moderate_low_max}, moderate_high<={t.moderate_high_max}, "
            f"high>{t.moderate_high_max}) — an adaptation of USGS/FIREMON "
            f"conventions for unscaled reflectance, not a universally "
            f"validated standard. Different thresholds would yield a "
            f"different reported affected area."
        )

    limitations.append(
        "This report covers only wildfire monitoring. GeoWatch does not yet "
        "implement vegetation-decline, land-disturbance, or mining-related "
        "detection — no claims are made about non-fire environmental change "
        "in this AOI."
    )

    return limitations


def generate_intelligence_report(
    aoi: AOI,
    period_start: datetime,
    period_end: datetime,
    events: list[GeoWatchEvent],
    wildfire_result: Optional[WildfireDetectionResult] = None,
    pixel_resolution_m: Optional[float] = None,
) -> IntelligenceReport:
    """Assemble an IntelligenceReport from stored events and, optionally, a
    specific imagery-based wildfire detection result.

    Args:
        aoi: the Area of Interest this report covers.
        period_start: start of the monitoring period.
        period_end: end of the monitoring period.
        events: events to include (caller is responsible for having
            already filtered these to the AOI/period — this function
            summarizes what it's given rather than re-filtering).
        wildfire_result: optional WildfireDetectionResult (see
            src/detection/wildfire.py) from a specific pre/post-fire
            imagery comparison, if one was run for this AOI/period.
        pixel_resolution_m: required if wildfire_result is provided —
            needed to convert its pixel mask into real hectares.

    Returns:
        A complete IntelligenceReport.

    Raises:
        ValueError: if wildfire_result is provided without pixel_resolution_m.
    """
    if wildfire_result is not None and pixel_resolution_m is None:
        raise ValueError(
            "generate_intelligence_report: pixel_resolution_m is required "
            "when wildfire_result is provided, to convert the pixel mask "
            "into real area units."
        )

    events_by_type: dict[str, int] = {}
    events_by_evidence_level: dict[str, int] = {}
    for e in events:
        events_by_type[e.event_type.value] = events_by_type.get(e.event_type.value, 0) + 1
        events_by_evidence_level[e.evidence_level.value] = (
            events_by_evidence_level.get(e.evidence_level.value, 0) + 1
        )

    average_confidence = (
        round(sum(e.confidence.value for e in events) / len(events), 3) if events else None
    )
    highest_confidence_event_id = (
        max(events, key=lambda e: e.confidence.value).event_id if events else None
    )

    burned_area_hectares: Optional[float] = None
    burned_area_percentage: Optional[float] = None
    severity_breakdown: Optional[dict[str, int]] = None
    severity_thresholds_used: Optional[dict[str, float]] = None

    if wildfire_result is not None:
        area_stats: AreaStats = compute_area_stats(
            wildfire_result.burned_mask, pixel_resolution_m=pixel_resolution_m
        )
        burned_area_hectares = round(area_stats.affected_area_hectares, 2)
        burned_area_percentage = round(area_stats.affected_percentage, 2)
        severity_breakdown = wildfire_result.severity_counts
        t = wildfire_result.thresholds
        severity_thresholds_used = {
            "unburned_max": t.unburned_max,
            "low_max": t.low_max,
            "moderate_low_max": t.moderate_low_max,
            "moderate_high_max": t.moderate_high_max,
        }

    return IntelligenceReport(
        aoi_label=aoi.label,
        aoi_bbox=aoi.as_bbox_tuple(),
        period_start=period_start.isoformat(),
        period_end=period_end.isoformat(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_events=len(events),
        events_by_type=events_by_type,
        events_by_evidence_level=events_by_evidence_level,
        average_confidence=average_confidence,
        highest_confidence_event_id=highest_confidence_event_id,
        burned_area_hectares=burned_area_hectares,
        burned_area_percentage=burned_area_percentage,
        severity_breakdown=severity_breakdown,
        severity_thresholds_used=severity_thresholds_used,
        events=[_event_to_summary(e) for e in events],
        limitations=_build_limitations(events, wildfire_result),
    )
