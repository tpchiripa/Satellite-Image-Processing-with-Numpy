"""
Core shared types for GeoWatch.

This module exists to make one product principle impossible to violate
by accident: GeoWatch must never present a model's guess as a verified
fact. Every detection, classification, or scored result in this system
carries an explicit EvidenceLevel so that downstream code (dashboard,
reports, AI explainer) can render language appropriate to how certain
we actually are — instead of every function silently degrading into
a confident-sounding string.

Rule of thumb used throughout GeoWatch:
    OBSERVED   -> "A satellite recorded this."
    DETECTED   -> "A deterministic calculation (NDVI, dNBR, etc.) found this."
    INFERRED   -> "A pattern suggests this, combining multiple signals."
    PREDICTED  -> "A trained model estimated this; it has a known error rate."
    CONFIRMED  -> "This has independent verification beyond satellite data."

No module in this codebase should hardcode phrases like "illegal mining
detected." Language like "potential mining-related land disturbance"
should be generated FROM the EvidenceLevel + confidence score, not typed
out ad hoc in fifteen different places.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class EvidenceLevel(str, Enum):
    """How certain GeoWatch is about a given piece of information.

    Ordered loosely from "raw fact" to "independently verified," but
    note PREDICTED and INFERRED are not strictly ranked against each
    other — they represent different *kinds* of uncertainty, not a
    single confidence scale.
    """

    OBSERVED = "observed"     # Raw satellite/sensor observation, no interpretation.
    DETECTED = "detected"     # Deterministic calculation applied to observations (NDVI, dNBR...).
    INFERRED = "inferred"     # Multiple signals combined into a pattern-level judgment.
    PREDICTED = "predicted"   # Output of a trained ML model.
    CONFIRMED = "confirmed"   # Independently verified beyond satellite evidence.

    def display_phrase(self) -> str:
        """Human-facing qualifier to prefix any claim at this evidence level."""
        return {
            EvidenceLevel.OBSERVED: "Observed:",
            EvidenceLevel.DETECTED: "Detected:",
            EvidenceLevel.INFERRED: "Potential:",
            EvidenceLevel.PREDICTED: "Model estimate:",
            EvidenceLevel.CONFIRMED: "Confirmed:",
        }[self]


class EventType(str, Enum):
    """Extensible event taxonomy. Add new types here, not new tables."""

    WILDFIRE = "wildfire"
    VEGETATION_DECLINE = "vegetation_decline"
    DEFORESTATION = "deforestation"
    LAND_DISTURBANCE = "land_disturbance"
    POTENTIAL_MINING = "potential_mining"
    FLOOD = "flood"
    OTHER = "other"


class EventStatus(str, Enum):
    NEW = "new"
    ONGOING = "ongoing"
    RESOLVED = "resolved"
    UNDER_REVIEW = "under_review"


@dataclass
class Evidence:
    """One supporting fact behind a claim, with its own evidence level.

    A single Event can (and usually should) be backed by several Evidence
    entries of different levels — e.g. an OBSERVED active-fire detection
    plus a DETECTED dNBR calculation plus an INFERRED persistence pattern.
    """

    description: str
    level: EvidenceLevel
    source: str
    observed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceScore:
    """A bounded, explainable confidence score — never a bare float in isolation."""

    value: float  # 0.0 - 1.0
    basis: str    # short human explanation of what drove this number
    evidence_level: EvidenceLevel

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"ConfidenceScore.value must be in [0, 1], got {self.value}")


@dataclass
class GeoWatchEvent:
    """Generic environmental event record (see Phase 10 in the project spec).

    This shape is intentionally source-agnostic: a wildfire event and a
    potential-mining event both use this same structure, distinguished by
    event_type. Do not create parallel per-domain event classes — extend
    this one, and extend EventType instead.
    """

    event_id: str
    event_type: EventType
    latitude: float
    longitude: float
    detected_at: datetime
    observation_time: Optional[datetime]
    source: str
    evidence_level: EvidenceLevel
    confidence: ConfidenceScore
    status: EventStatus = EventStatus.NEW
    severity: Optional[str] = None
    geometry: Optional[dict[str, Any]] = None  # GeoJSON-like geometry, kept generic for now
    evidence: list[Evidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_line(self) -> str:
        """A single, responsibly-worded sentence describing this event.

        This is the ONLY place event summary text should be generated from,
        so that the evidence-level discipline is enforced in one location
        instead of scattered across the dashboard and report code.
        """
        qualifier = self.evidence_level.display_phrase()
        return f"{qualifier} {self.event_type.value.replace('_', ' ')} near ({self.latitude:.3f}, {self.longitude:.3f})"
