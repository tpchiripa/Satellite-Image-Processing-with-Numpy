"""
Vegetation change detection from dNDVI (Phase 7).

Structurally mirrors src/detection/wildfire.py — same two principles apply:

1. DECLINE THRESHOLDS ARE METHODOLOGY, NOT FACT. The defaults below are
   a reasonable, illustrative breakpoint scheme, not a validated
   ecological standard. What counts as "significant" vegetation decline
   genuinely depends on ecosystem type, season, and baseline variability
   — a savanna's normal seasonal NDVI swing can exceed a rainforest's
   drought-stress signal. Override VegetationChangeThresholds for your
   own AOI/ecosystem rather than treating the defaults as ground truth.

2. EVIDENCE LEVEL. Vegetation-change results here are the output of a
   deterministic calculation (dNDVI) applied to satellite-derived
   reflectance — that's EvidenceLevel.DETECTED, not INFERRED or
   CONFIRMED. A sustained multi-period decline pattern that combines
   this with other signals (e.g. persistence over time, proximity to
   known disturbance) would be INFERRED-level reasoning, which belongs
   in src/detection/disturbance.py — not here.

Unlike burn severity, vegetation change is NOT one-directional: a
negative dNDVI means vegetation increased (regrowth, a wet season, crop
growth), which is genuinely useful information, not just "no decline."
classify_vegetation_change() reports this explicitly as IMPROVEMENT
rather than folding it into a generic "stable" bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.types import EvidenceLevel


class VegetationChangeClass(str, Enum):
    """Vegetation change classes, ordered from most improved to most declined."""

    NO_DATA = "no_data"
    IMPROVEMENT = "improvement"
    STABLE = "stable"
    SLIGHT_DECLINE = "slight_decline"
    MODERATE_DECLINE = "moderate_decline"
    SEVERE_DECLINE = "severe_decline"


@dataclass
class VegetationChangeThresholds:
    """dNDVI breakpoints used to classify vegetation change.

    Values are inclusive upper bounds: a pixel is assigned to the first
    class whose bound it does not exceed. Anything above
    moderate_decline_max is classified SEVERE_DECLINE; anything at or
    below improvement_max is classified IMPROVEMENT.

    Defaults are illustrative, not a validated ecological standard —
    see module docstring. Override for your own ecosystem/season.
    """

    improvement_max: float = -0.10
    stable_max: float = 0.05
    slight_decline_max: float = 0.15
    moderate_decline_max: float = 0.30

    def __post_init__(self) -> None:
        bounds = [
            self.improvement_max,
            self.stable_max,
            self.slight_decline_max,
            self.moderate_decline_max,
        ]
        if bounds != sorted(bounds):
            raise ValueError(
                "VegetationChangeThresholds must be strictly increasing: "
                f"improvement_max={self.improvement_max}, stable_max={self.stable_max}, "
                f"slight_decline_max={self.slight_decline_max}, "
                f"moderate_decline_max={self.moderate_decline_max}"
            )


@dataclass
class VegetationDeclineResult:
    """Full output of a vegetation-change analysis run.

    evidence_level is always DETECTED — see module docstring. thresholds
    is included so any downstream report or dashboard can disclose
    exactly which methodology produced this result.
    """

    decline_mask: np.ndarray
    change_labels: np.ndarray
    change_counts: dict[str, int]
    mean_ndvi_change: float
    thresholds: VegetationChangeThresholds
    evidence_level: EvidenceLevel = field(default=EvidenceLevel.DETECTED)


def detect_vegetation_decline(dndvi: np.ndarray, decline_threshold: float = 0.05) -> np.ndarray:
    """Produce a boolean decline mask from a dNDVI array.

    Args:
        dndvi: array-like of dNDVI values (see compute_dndvi()). Positive
            values indicate NDVI decreased from baseline to recent.
        decline_threshold: minimum dNDVI to consider a pixel declining.
            Defaults to 0.05, matching
            VegetationChangeThresholds.stable_max so this function and
            classify_vegetation_change() agree on the decline boundary
            by default.

    Returns:
        Boolean ndarray, same shape as dndvi. True = declining. No-data
        (NaN) pixels are False — absence of information is not evidence
        of decline.

    Raises:
        ValueError: if decline_threshold is not finite.
    """
    if not np.isfinite(decline_threshold):
        raise ValueError(
            f"detect_vegetation_decline: decline_threshold must be finite, got {decline_threshold}"
        )

    arr = np.asarray(dndvi, dtype=np.float64)
    # NaN comparisons are always False, so NaN pixels correctly fall out
    # of the mask without special-casing.
    return arr >= decline_threshold


def classify_vegetation_change(
    dndvi: np.ndarray, thresholds: VegetationChangeThresholds | None = None
) -> np.ndarray:
    """Classify each pixel's vegetation change from its dNDVI value.

    Args:
        dndvi: array-like of dNDVI values.
        thresholds: VegetationChangeThresholds to use. Defaults to
            VegetationChangeThresholds() if not provided.

    Returns:
        Object ndarray of the same shape as dndvi, with each element a
        VegetationChangeClass value. NaN input pixels become
        VegetationChangeClass.NO_DATA.
    """
    if thresholds is None:
        thresholds = VegetationChangeThresholds()

    arr = np.asarray(dndvi, dtype=np.float64)
    # np.empty + .fill() rather than np.full(): np.full() silently
    # truncates str-subclassed Enum fill values even with dtype=object
    # explicit (a real numpy gotcha, first caught in wildfire.py's test
    # suite) — this avoids the dtype-inference path that causes it.
    result = np.empty(arr.shape, dtype=object)
    result.fill(VegetationChangeClass.NO_DATA)

    valid = ~np.isnan(arr)
    result[valid & (arr <= thresholds.improvement_max)] = VegetationChangeClass.IMPROVEMENT
    result[
        valid & (arr > thresholds.improvement_max) & (arr <= thresholds.stable_max)
    ] = VegetationChangeClass.STABLE
    result[
        valid & (arr > thresholds.stable_max) & (arr <= thresholds.slight_decline_max)
    ] = VegetationChangeClass.SLIGHT_DECLINE
    result[
        valid & (arr > thresholds.slight_decline_max) & (arr <= thresholds.moderate_decline_max)
    ] = VegetationChangeClass.MODERATE_DECLINE
    result[valid & (arr > thresholds.moderate_decline_max)] = VegetationChangeClass.SEVERE_DECLINE

    return result


def summarize_change_counts(change_labels: np.ndarray) -> dict[str, int]:
    """Count pixels per VegetationChangeClass.

    Args:
        change_labels: object ndarray of VegetationChangeClass values, as
            returned by classify_vegetation_change().

    Returns:
        Dict mapping each VegetationChangeClass's string value to its
        pixel count. Always includes every class key, even if 0.
    """
    flat = np.asarray(change_labels, dtype=object).ravel()
    counts = {change_class.value: 0 for change_class in VegetationChangeClass}
    for label in flat:
        counts[label.value] += 1
    return counts


def analyze_vegetation_change(
    dndvi: np.ndarray,
    decline_threshold: float | None = None,
    thresholds: VegetationChangeThresholds | None = None,
) -> VegetationDeclineResult:
    """Run the full vegetation-change analysis pipeline on a dNDVI array.

    This is the main entry point consumers should use — it ties together
    detect_vegetation_decline(), classify_vegetation_change(), and
    summarize_change_counts() into a single auditable result.

    Args:
        dndvi: array-like of dNDVI values.
        decline_threshold: passed to detect_vegetation_decline(). Defaults
            to thresholds.stable_max if not provided, keeping the boolean
            mask and the classification in agreement by default.
        thresholds: passed to classify_vegetation_change(). Defaults to
            VegetationChangeThresholds() if not provided.

    Returns:
        VegetationDeclineResult with the decline mask, per-pixel change
        labels, change class counts, mean NDVI change, and the thresholds
        actually used.
    """
    if thresholds is None:
        thresholds = VegetationChangeThresholds()
    if decline_threshold is None:
        decline_threshold = thresholds.stable_max

    decline_mask = detect_vegetation_decline(dndvi, decline_threshold=decline_threshold)
    change_labels = classify_vegetation_change(dndvi, thresholds=thresholds)
    change_counts = summarize_change_counts(change_labels)

    arr = np.asarray(dndvi, dtype=np.float64)
    valid_values = arr[~np.isnan(arr)]
    mean_change = float(np.mean(valid_values)) if valid_values.size > 0 else float("nan")

    return VegetationDeclineResult(
        decline_mask=decline_mask,
        change_labels=change_labels,
        change_counts=change_counts,
        mean_ndvi_change=mean_change,
        thresholds=thresholds,
    )
