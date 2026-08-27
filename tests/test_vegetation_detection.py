"""Tests for src/detection/vegetation.py"""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.vegetation import (
    VegetationChangeClass,
    VegetationChangeThresholds,
    analyze_vegetation_change,
    classify_vegetation_change,
    detect_vegetation_decline,
    summarize_change_counts,
)
from src.types import EvidenceLevel


class TestVegetationChangeThresholds:
    def test_default_thresholds_are_valid(self) -> None:
        t = VegetationChangeThresholds()
        assert t.improvement_max < t.stable_max < t.slight_decline_max < t.moderate_decline_max

    def test_non_increasing_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError):
            VegetationChangeThresholds(
                improvement_max=0.1, stable_max=0.05, slight_decline_max=0.2, moderate_decline_max=0.3
            )

    def test_custom_thresholds_accepted(self) -> None:
        t = VegetationChangeThresholds(
            improvement_max=-0.2, stable_max=0.0, slight_decline_max=0.1, moderate_decline_max=0.2
        )
        assert t.stable_max == 0.0


class TestDetectVegetationDecline:
    def test_above_threshold_is_declining(self) -> None:
        mask = detect_vegetation_decline(np.array([0.3]), decline_threshold=0.05)
        assert mask[0] == True  # noqa: E712

    def test_below_threshold_is_not_declining(self) -> None:
        mask = detect_vegetation_decline(np.array([0.01]), decline_threshold=0.05)
        assert mask[0] == False  # noqa: E712

    def test_exactly_at_threshold_is_declining(self) -> None:
        mask = detect_vegetation_decline(np.array([0.05]), decline_threshold=0.05)
        assert mask[0] == True  # noqa: E712

    def test_nan_pixel_is_not_declining(self) -> None:
        mask = detect_vegetation_decline(np.array([np.nan]), decline_threshold=0.05)
        assert mask[0] == False  # noqa: E712

    def test_negative_dndvi_is_not_declining(self) -> None:
        # Negative dNDVI = vegetation improvement, not decline.
        mask = detect_vegetation_decline(np.array([-0.3]), decline_threshold=0.05)
        assert mask[0] == False  # noqa: E712

    def test_nonfinite_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            detect_vegetation_decline(np.array([0.3]), decline_threshold=np.nan)

    def test_2d_array(self) -> None:
        dndvi = np.array([[0.3, 0.01], [np.nan, 0.5]])
        mask = detect_vegetation_decline(dndvi, decline_threshold=0.05)
        assert mask[0, 0] == True and mask[0, 1] == False  # noqa: E712
        assert mask[1, 0] == False and mask[1, 1] == True  # noqa: E712


class TestClassifyVegetationChange:
    def test_improvement_classification(self) -> None:
        result = classify_vegetation_change(np.array([-0.2]))
        assert result[0] == VegetationChangeClass.IMPROVEMENT

    def test_stable_classification(self) -> None:
        result = classify_vegetation_change(np.array([0.0]))
        assert result[0] == VegetationChangeClass.STABLE

    def test_slight_decline_classification(self) -> None:
        result = classify_vegetation_change(np.array([0.1]))
        assert result[0] == VegetationChangeClass.SLIGHT_DECLINE

    def test_moderate_decline_classification(self) -> None:
        result = classify_vegetation_change(np.array([0.25]))
        assert result[0] == VegetationChangeClass.MODERATE_DECLINE

    def test_severe_decline_classification(self) -> None:
        result = classify_vegetation_change(np.array([0.5]))
        assert result[0] == VegetationChangeClass.SEVERE_DECLINE

    def test_nan_gives_no_data(self) -> None:
        result = classify_vegetation_change(np.array([np.nan]))
        assert result[0] == VegetationChangeClass.NO_DATA

    def test_boundary_values_use_lower_class(self) -> None:
        result = classify_vegetation_change(np.array([0.05]))  # exactly stable_max
        assert result[0] == VegetationChangeClass.STABLE

    def test_custom_thresholds_change_classification(self) -> None:
        strict = VegetationChangeThresholds(
            improvement_max=-0.3, stable_max=-0.1, slight_decline_max=0.0, moderate_decline_max=0.1
        )
        result = classify_vegetation_change(np.array([0.05]), thresholds=strict)
        assert result[0] == VegetationChangeClass.MODERATE_DECLINE

    def test_full_gradient_array(self) -> None:
        dndvi = np.array([-0.2, 0.0, 0.1, 0.25, 0.5, np.nan])
        result = classify_vegetation_change(dndvi)
        expected = [
            VegetationChangeClass.IMPROVEMENT,
            VegetationChangeClass.STABLE,
            VegetationChangeClass.SLIGHT_DECLINE,
            VegetationChangeClass.MODERATE_DECLINE,
            VegetationChangeClass.SEVERE_DECLINE,
            VegetationChangeClass.NO_DATA,
        ]
        assert list(result) == expected

    def test_class_values_are_real_enum_not_truncated_string(self) -> None:
        # Guards against the np.full() str-Enum truncation bug found in
        # wildfire.py's development (see module docstring).
        result = classify_vegetation_change(np.array([0.0]))
        assert isinstance(result[0], VegetationChangeClass)
        assert result[0].value == "stable"


class TestSummarizeChangeCounts:
    def test_counts_all_classes(self) -> None:
        dndvi = np.array([-0.2, -0.2, 0.5, np.nan])
        labels = classify_vegetation_change(dndvi)
        counts = summarize_change_counts(labels)
        assert counts[VegetationChangeClass.IMPROVEMENT.value] == 2
        assert counts[VegetationChangeClass.SEVERE_DECLINE.value] == 1
        assert counts[VegetationChangeClass.NO_DATA.value] == 1
        assert counts[VegetationChangeClass.STABLE.value] == 0

    def test_includes_zero_count_classes(self) -> None:
        labels = classify_vegetation_change(np.array([0.0]))
        counts = summarize_change_counts(labels)
        assert set(counts.keys()) == {c.value for c in VegetationChangeClass}


class TestAnalyzeVegetationChange:
    def test_returns_evidence_level_detected(self) -> None:
        result = analyze_vegetation_change(np.array([0.2]))
        assert result.evidence_level == EvidenceLevel.DETECTED

    def test_result_includes_thresholds_used(self) -> None:
        custom = VegetationChangeThresholds(
            improvement_max=-0.2, stable_max=0.0, slight_decline_max=0.1, moderate_decline_max=0.2
        )
        result = analyze_vegetation_change(np.array([0.2]), thresholds=custom)
        assert result.thresholds is custom

    def test_decline_mask_and_classification_agree_by_default(self) -> None:
        dndvi = np.array([-0.2, 0.0, 0.1, 0.5, np.nan])
        result = analyze_vegetation_change(dndvi)
        for i, label in enumerate(result.change_labels):
            if label in (VegetationChangeClass.IMPROVEMENT, VegetationChangeClass.STABLE, VegetationChangeClass.NO_DATA):
                assert result.decline_mask[i] == False  # noqa: E712
            else:
                assert result.decline_mask[i] == True  # noqa: E712

    def test_change_counts_sum_to_total_pixels(self) -> None:
        dndvi = np.array([-0.2, 0.0, 0.1, 0.5, np.nan, 0.3])
        result = analyze_vegetation_change(dndvi)
        assert sum(result.change_counts.values()) == dndvi.size

    def test_mean_ndvi_change_ignores_nan(self) -> None:
        dndvi = np.array([0.2, 0.4, np.nan])
        result = analyze_vegetation_change(dndvi)
        assert result.mean_ndvi_change == pytest.approx(0.3)

    def test_mean_ndvi_change_is_nan_when_all_nan(self) -> None:
        dndvi = np.array([np.nan, np.nan])
        result = analyze_vegetation_change(dndvi)
        assert np.isnan(result.mean_ndvi_change)

    def test_end_to_end_from_synthetic_bands(self) -> None:
        from src.remote_sensing.ndvi import compute_dndvi, compute_ndvi

        nir_pre, red_pre = np.array([0.5, 0.5]), np.array([0.1, 0.1])
        nir_post, red_post = np.array([0.1, 0.5]), np.array([0.3, 0.1])

        ndvi_pre = compute_ndvi(nir_pre, red_pre)
        ndvi_post = compute_ndvi(nir_post, red_post)
        dndvi = compute_dndvi(ndvi_pre, ndvi_post)

        result = analyze_vegetation_change(dndvi)
        assert result.decline_mask[0] == True  # noqa: E712
        assert result.decline_mask[1] == False  # noqa: E712
        assert result.change_labels[0] == VegetationChangeClass.SEVERE_DECLINE
        assert result.change_labels[1] == VegetationChangeClass.STABLE
