"""
Wildfire detection: burned-area masking and burn-severity classification
from dNBR (see src/remote_sensing/nbr.py for the underlying calculation).

Two things this module is deliberately careful about:

1. SEVERITY THRESHOLDS ARE METHODOLOGY, NOT FACT. The defaults below
   follow the commonly cited USGS/FIREMON (Key & Benson) dNBR breakpoints,
   but that scheme was calibrated on specific sensors and a *1000-scaled*
   NBR convention. GeoWatch computes NBR on unscaled reflectance (roughly
   -1 to 1), so these defaults are a reasonable adaptation, not a
   validated standard — they are fully overridable via SeverityThresholds,
   and every result is tagged with the thresholds actually used so it's
   auditable later.

2. EVIDENCE LEVEL. Burned-area/severity results here are the output of a
   deterministic calculation applied to satellite-derived reflectance —
   that's EvidenceLevel.DETECTED, not INFERRED or CONFIRMED. Do not
   upgrade the evidence level of anything returned from this module
   without a genuinely independent additional signal (see
   src/detection/disturbance.py for where INFERRED-level reasoning
   belongs, once it exists).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.types import EvidenceLevel


class BurnSeverity(str, Enum):
    """Burn severity classes, ordered from no signal to most severe."""

    NO_DATA = "no_data"
    UNBURNED = "unburned"
    LOW = "low"
    MODERATE_LOW = "moderate_low"
    MODERATE_HIGH = "moderate_high"
    HIGH = "high"


@dataclass
class SeverityThresholds:
    """dNBR breakpoints used to classify burn severity.

    Values are inclusive upper bounds: a pixel is assigned to the first
    class whose bound it does not exceed. Anything above
    moderate_high_max is classified HIGH.

    Defaults are an adaptation of the USGS/FIREMON (Key & Benson) scheme
    for unscaled (non-*1000) NBR. Override these for your own sensor,
    ecosystem, or validated study rather than treating the defaults as
    ground truth — see module docstring.
    """

    unburned_max: float = 0.10
    low_max: float = 0.27
    moderate_low_max: float = 0.44
    moderate_high_max: float = 0.66

    def __post_init__(self) -> None:
        bounds = [self.unburned_max, self.low_max, self.moderate_low_max, self.moderate_high_max]
        if bounds != sorted(bounds):
            raise ValueError(
                "SeverityThresholds must be strictly increasing: "
                f"unburned_max={self.unburned_max}, low_max={self.low_max}, "
                f"moderate_low_max={self.moderate_low_max}, "
                f"moderate_high_max={self.moderate_high_max}"
            )


@dataclass
class WildfireDetectionResult:
    """Full output of a burned-area/severity detection run.

    evidence_level is always DETECTED — see module docstring. thresholds
    is included so any downstream report or dashboard can disclose
    exactly which methodology produced this result.
    """

    burned_mask: np.ndarray
    severity_labels: np.ndarray
    severity_counts: dict[str, int]
    thresholds: SeverityThresholds
    evidence_level: EvidenceLevel = field(default=EvidenceLevel.DETECTED)


def detect_burned_area(dnbr: np.ndarray, burn_threshold: float = 0.10) -> np.ndarray:
    """Produce a boolean burned-area mask from a dNBR array.

    Args:
        dnbr: array-like of dNBR values (see compute_dnbr()).
        burn_threshold: minimum dNBR to consider a pixel burned. Defaults
            to 0.10, matching SeverityThresholds.unburned_max so this
            function and classify_burn_severity() agree on the burned/
            unburned boundary by default.

    Returns:
        Boolean ndarray, same shape as dnbr. True = burned. No-data
        (NaN) pixels are False — absence of information is not evidence
        of burning.

    Raises:
        ValueError: if burn_threshold is not finite.
    """
    if not np.isfinite(burn_threshold):
        raise ValueError(f"detect_burned_area: burn_threshold must be finite, got {burn_threshold}")

    arr = np.asarray(dnbr, dtype=np.float64)
    # NaN comparisons are always False, so NaN pixels correctly fall out
    # of the mask without special-casing.
    return arr >= burn_threshold


def classify_burn_severity(
    dnbr: np.ndarray, thresholds: SeverityThresholds | None = None
) -> np.ndarray:
    """Classify each pixel's burn severity from its dNBR value.

    Args:
        dnbr: array-like of dNBR values.
        thresholds: SeverityThresholds to use. Defaults to
            SeverityThresholds() if not provided.

    Returns:
        Object ndarray of the same shape as dnbr, with each element a
        BurnSeverity value. NaN input pixels become BurnSeverity.NO_DATA.
    """
    if thresholds is None:
        thresholds = SeverityThresholds()

    arr = np.asarray(dnbr, dtype=np.float64)
    # NOTE: np.full(shape, BurnSeverity.NO_DATA, dtype=object) silently
    # truncates str-subclassed Enum values to a plain fixed-width string
    # (a real numpy gotcha, caught by this module's own test suite) —
    # np.empty + .fill() avoids the dtype-inference path that causes it.
    result = np.empty(arr.shape, dtype=object)
    result.fill(BurnSeverity.NO_DATA)

    valid = ~np.isnan(arr)
    result[valid & (arr <= thresholds.unburned_max)] = BurnSeverity.UNBURNED
    result[valid & (arr > thresholds.unburned_max) & (arr <= thresholds.low_max)] = BurnSeverity.LOW
    result[
        valid & (arr > thresholds.low_max) & (arr <= thresholds.moderate_low_max)
    ] = BurnSeverity.MODERATE_LOW
    result[
        valid & (arr > thresholds.moderate_low_max) & (arr <= thresholds.moderate_high_max)
    ] = BurnSeverity.MODERATE_HIGH
    result[valid & (arr > thresholds.moderate_high_max)] = BurnSeverity.HIGH

    return result


def summarize_severity_counts(severity_labels: np.ndarray) -> dict[str, int]:
    """Count pixels per BurnSeverity class.

    Args:
        severity_labels: object ndarray of BurnSeverity values, as
            returned by classify_burn_severity().

    Returns:
        Dict mapping each BurnSeverity's string value to its pixel count.
        Always includes every BurnSeverity key, even if the count is 0.
    """
    flat = np.asarray(severity_labels, dtype=object).ravel()
    counts = {severity.value: 0 for severity in BurnSeverity}
    for label in flat:
        counts[label.value] += 1
    return counts


def detect_wildfire(
    dnbr: np.ndarray,
    burn_threshold: float = 0.10,
    thresholds: SeverityThresholds | None = None,
) -> WildfireDetectionResult:
    """Run the full burned-area + severity detection pipeline on a dNBR array.

    This is the main entry point Milestone 2 consumers should use — it
    ties together detect_burned_area(), classify_burn_severity(), and
    summarize_severity_counts() into a single auditable result.

    Args:
        dnbr: array-like of dNBR values.
        burn_threshold: passed to detect_burned_area().
        thresholds: passed to classify_burn_severity(). Defaults to
            SeverityThresholds() if not provided.

    Returns:
        WildfireDetectionResult with the burned mask, per-pixel severity
        labels, severity class counts, and the thresholds actually used.
    """
    if thresholds is None:
        thresholds = SeverityThresholds()

    burned_mask = detect_burned_area(dnbr, burn_threshold=burn_threshold)
    severity_labels = classify_burn_severity(dnbr, thresholds=thresholds)
    severity_counts = summarize_severity_counts(severity_labels)

    return WildfireDetectionResult(
        burned_mask=burned_mask,
        severity_labels=severity_labels,
        severity_counts=severity_counts,
        thresholds=thresholds,
    )
