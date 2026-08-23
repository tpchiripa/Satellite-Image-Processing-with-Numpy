"""Tests for src/detection/wildfire.py"""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.wildfire import (
    BurnSeverity,
    SeverityThresholds,
    classify_burn_severity,
    detect_burned_area,
    detect_wildfire,
    summarize_severity_counts,
)
from src.types import EvidenceLevel


class TestSeverityThresholds:
    def test_default_thresholds_are_valid(self) -> None:
        thresholds = SeverityThresholds()
        assert thresholds.unburned_max < thresholds.low_max < thresholds.moderate_low_max < thresholds.moderate_high_max

    def test_non_increasing_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError):
            SeverityThresholds(unburned_max=0.5, low_max=0.3, moderate_low_max=0.6, moderate_high_max=0.8)

    def test_custom_thresholds_accepted(self) -> None:
        thresholds = SeverityThresholds(
            unburned_max=0.05, low_max=0.2, moderate_low_max=0.4, moderate_high_max=0.6
        )
        assert thresholds.low_max == 0.2


class TestDetectBurnedArea:
    def test_above_threshold_is_burned(self) -> None:
        dnbr = np.array([0.5])
        mask = detect_burned_area(dnbr, burn_threshold=0.1)
        assert mask[0] == True  # noqa: E712

    def test_below_threshold_is_unburned(self) -> None:
        dnbr = np.array([0.05])
        mask = detect_burned_area(dnbr, burn_threshold=0.1)
        assert mask[0] == False  # noqa: E712

    def test_exactly_at_threshold_is_burned(self) -> None:
        dnbr = np.array([0.1])
        mask = detect_burned_area(dnbr, burn_threshold=0.1)
        assert mask[0] == True  # noqa: E712

    def test_nan_pixel_is_not_burned(self) -> None:
        dnbr = np.array([np.nan])
        mask = detect_burned_area(dnbr, burn_threshold=0.1)
        assert mask[0] == False  # noqa: E712

    def test_negative_dnbr_is_unburned(self) -> None:
        # Negative dNBR = vegetation increase/regrowth, not burning.
        dnbr = np.array([-0.3])
        mask = detect_burned_area(dnbr, burn_threshold=0.1)
        assert mask[0] == False  # noqa: E712

    def test_nonfinite_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            detect_burned_area(np.array([0.5]), burn_threshold=np.nan)

    def test_2d_array(self) -> None:
        dnbr = np.array([[0.5, 0.05], [np.nan, 0.9]])
        mask = detect_burned_area(dnbr, burn_threshold=0.1)
        assert mask.shape == (2, 2)
        assert mask[0, 0] == True and mask[0, 1] == False  # noqa: E712
        assert mask[1, 0] == False and mask[1, 1] == True  # noqa: E712


class TestClassifyBurnSeverity:
    def test_unburned_classification(self) -> None:
        result = classify_burn_severity(np.array([0.05]))
        assert result[0] == BurnSeverity.UNBURNED

    def test_low_severity_classification(self) -> None:
        result = classify_burn_severity(np.array([0.2]))
        assert result[0] == BurnSeverity.LOW

    def test_moderate_low_classification(self) -> None:
        result = classify_burn_severity(np.array([0.35]))
        assert result[0] == BurnSeverity.MODERATE_LOW

    def test_moderate_high_classification(self) -> None:
        result = classify_burn_severity(np.array([0.55]))
        assert result[0] == BurnSeverity.MODERATE_HIGH

    def test_high_severity_classification(self) -> None:
        result = classify_burn_severity(np.array([0.9]))
        assert result[0] == BurnSeverity.HIGH

    def test_nan_gives_no_data(self) -> None:
        result = classify_burn_severity(np.array([np.nan]))
        assert result[0] == BurnSeverity.NO_DATA

    def test_boundary_values_use_lower_class(self) -> None:
        # Thresholds are inclusive upper bounds: exactly at unburned_max (0.10)
        # should still classify as UNBURNED, not LOW.
        result = classify_burn_severity(np.array([0.10]))
        assert result[0] == BurnSeverity.UNBURNED

    def test_custom_thresholds_change_classification(self) -> None:
        # With a much lower unburned_max, the same 0.15 value that would
        # normally be LOW should now be MODERATE_LOW or higher.
        strict_thresholds = SeverityThresholds(
            unburned_max=0.02, low_max=0.05, moderate_low_max=0.10, moderate_high_max=0.20
        )
        result = classify_burn_severity(np.array([0.15]), thresholds=strict_thresholds)
        assert result[0] == BurnSeverity.MODERATE_HIGH

    def test_full_gradient_array(self) -> None:
        dnbr = np.array([-0.1, 0.05, 0.2, 0.35, 0.55, 0.9, np.nan])
        result = classify_burn_severity(dnbr)
        expected = [
            BurnSeverity.UNBURNED,
            BurnSeverity.UNBURNED,
            BurnSeverity.LOW,
            BurnSeverity.MODERATE_LOW,
            BurnSeverity.MODERATE_HIGH,
            BurnSeverity.HIGH,
            BurnSeverity.NO_DATA,
        ]
        assert list(result) == expected


class TestSummarizeSeverityCounts:
    def test_counts_all_classes(self) -> None:
        dnbr = np.array([0.05, 0.05, 0.9, np.nan])
        labels = classify_burn_severity(dnbr)
        counts = summarize_severity_counts(labels)
        assert counts[BurnSeverity.UNBURNED.value] == 2
        assert counts[BurnSeverity.HIGH.value] == 1
        assert counts[BurnSeverity.NO_DATA.value] == 1
        assert counts[BurnSeverity.LOW.value] == 0

    def test_includes_zero_count_classes(self) -> None:
        dnbr = np.array([0.05])
        labels = classify_burn_severity(dnbr)
        counts = summarize_severity_counts(labels)
        assert set(counts.keys()) == {s.value for s in BurnSeverity}

    def test_works_on_2d_input(self) -> None:
        dnbr = np.array([[0.05, 0.9], [0.9, 0.05]])
        labels = classify_burn_severity(dnbr)
        counts = summarize_severity_counts(labels)
        assert counts[BurnSeverity.HIGH.value] == 2
        assert counts[BurnSeverity.UNBURNED.value] == 2


class TestDetectWildfire:
    def test_returns_evidence_level_detected(self) -> None:
        result = detect_wildfire(np.array([0.5]))
        assert result.evidence_level == EvidenceLevel.DETECTED

    def test_result_includes_thresholds_used(self) -> None:
        custom = SeverityThresholds(unburned_max=0.05, low_max=0.15, moderate_low_max=0.3, moderate_high_max=0.5)
        result = detect_wildfire(np.array([0.5]), thresholds=custom)
        assert result.thresholds is custom

    def test_burned_mask_and_severity_agree(self) -> None:
        # Anything classified above UNBURNED should be in the burned mask
        # when burn_threshold matches unburned_max (the default relationship).
        dnbr = np.array([0.05, 0.2, 0.5, 0.9, np.nan])
        result = detect_wildfire(dnbr, burn_threshold=0.10)
        for i, label in enumerate(result.severity_labels):
            if label in (BurnSeverity.UNBURNED, BurnSeverity.NO_DATA):
                assert result.burned_mask[i] == False  # noqa: E712
            else:
                assert result.burned_mask[i] == True  # noqa: E712

    def test_severity_counts_sum_to_total_pixels(self) -> None:
        dnbr = np.array([0.05, 0.2, 0.5, 0.9, np.nan, -0.1])
        result = detect_wildfire(dnbr)
        assert sum(result.severity_counts.values()) == dnbr.size

    def test_end_to_end_from_synthetic_bands(self) -> None:
        # Full realistic pipeline: bands -> NBR -> dNBR -> wildfire detection.
        from src.remote_sensing.nbr import compute_dnbr, compute_nbr

        nir_pre, swir_pre = np.array([0.5, 0.5]), np.array([0.15, 0.15])
        nir_post, swir_post = np.array([0.1, 0.5]), np.array([0.3, 0.15])

        nbr_pre = compute_nbr(nir_pre, swir_pre)
        nbr_post = compute_nbr(nir_post, swir_post)
        dnbr = compute_dnbr(nbr_pre, nbr_post)

        result = detect_wildfire(dnbr)
        assert result.burned_mask[0] == True  # noqa: E712
        assert result.burned_mask[1] == False  # noqa: E712
        assert result.severity_labels[0] == BurnSeverity.HIGH
        assert result.severity_labels[1] == BurnSeverity.UNBURNED
