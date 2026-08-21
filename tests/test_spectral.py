"""Tests for src/remote_sensing/spectral.py — the safe_divide / normalization
/ masking primitives that NDVI and NBR are built on top of."""

from __future__ import annotations

import numpy as np
import pytest

from src.remote_sensing.spectral import (
    compute_normalized_difference,
    mask_invalid_pixels,
    normalize_band,
    safe_divide,
)


class TestSafeDivide:
    def test_basic_division(self) -> None:
        result = safe_divide(np.array([4.0, 9.0]), np.array([2.0, 3.0]))
        np.testing.assert_allclose(result, [2.0, 3.0])

    def test_zero_over_zero_is_nan_not_zero(self) -> None:
        # This is the key scientific decision: no signal at all must not
        # silently read as index value 0.
        result = safe_divide(np.array([0.0]), np.array([0.0]))
        assert np.isnan(result[0])

    def test_nonzero_over_zero_is_nan_not_inf(self) -> None:
        result = safe_divide(np.array([5.0]), np.array([0.0]))
        assert np.isnan(result[0])
        assert not np.isinf(result[0])

    def test_negative_numerator_over_zero_is_nan(self) -> None:
        result = safe_divide(np.array([-5.0]), np.array([0.0]))
        assert np.isnan(result[0])

    def test_nan_input_propagates(self) -> None:
        result = safe_divide(np.array([np.nan, 4.0]), np.array([2.0, 2.0]))
        assert np.isnan(result[0])
        assert result[1] == 2.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            safe_divide(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_integer_input_does_not_truncate(self) -> None:
        # int 1 / int 3 would truncate to 0 under naive integer division;
        # safe_divide must cast to float64 first.
        result = safe_divide(np.array([1], dtype=np.int32), np.array([3], dtype=np.int32))
        assert result[0] == pytest.approx(1 / 3)

    def test_2d_array(self) -> None:
        num = np.array([[4.0, 0.0], [9.0, 6.0]])
        den = np.array([[2.0, 0.0], [3.0, 2.0]])
        result = safe_divide(num, den)
        assert result.shape == (2, 2)
        assert np.isnan(result[0, 1])
        assert result[1, 1] == 3.0


class TestComputeNormalizedDifference:
    def test_known_value(self) -> None:
        # (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6
        result = compute_normalized_difference(np.array([0.5]), np.array([0.1]))
        assert result[0] == pytest.approx(0.4 / 0.6)

    def test_identical_bands_gives_zero(self) -> None:
        result = compute_normalized_difference(np.array([0.3]), np.array([0.3]))
        assert result[0] == pytest.approx(0.0)

    def test_both_zero_is_nan(self) -> None:
        result = compute_normalized_difference(np.array([0.0]), np.array([0.0]))
        assert np.isnan(result[0])

    def test_result_bounded_for_nonnegative_reflectance(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.uniform(0, 1, size=1000)
        b = rng.uniform(0, 1, size=1000)
        # Exclude the degenerate a=b=0 case, which is legitimately NaN.
        mask = (a + b) > 0
        result = compute_normalized_difference(a[mask], b[mask])
        assert np.all(result >= -1.0 - 1e-9)
        assert np.all(result <= 1.0 + 1e-9)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_normalized_difference(np.array([1.0, 2.0]), np.array([1.0]))

    def test_nan_propagates(self) -> None:
        result = compute_normalized_difference(np.array([np.nan]), np.array([0.2]))
        assert np.isnan(result[0])


class TestNormalizeBand:
    def test_scales_by_default_factor(self) -> None:
        result = normalize_band(np.array([5000.0]))
        assert result[0] == pytest.approx(0.5)

    def test_clips_above_one(self) -> None:
        result = normalize_band(np.array([15000.0]))
        assert result[0] == 1.0

    def test_clips_below_zero(self) -> None:
        result = normalize_band(np.array([-500.0]))
        assert result[0] == 0.0

    def test_custom_scale_factor(self) -> None:
        result = normalize_band(np.array([50.0]), scale_factor=100.0)
        assert result[0] == pytest.approx(0.5)

    def test_nonpositive_scale_factor_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_band(np.array([1.0]), scale_factor=0.0)
        with pytest.raises(ValueError):
            normalize_band(np.array([1.0]), scale_factor=-10.0)

    def test_nan_passes_through(self) -> None:
        result = normalize_band(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_deterministic_regardless_of_array_context(self) -> None:
        # normalize_band must NOT be a min-max stretch: the same value
        # should map identically whether or not extreme values are
        # present elsewhere in the array.
        a = normalize_band(np.array([5000.0, 10000.0]))
        b = normalize_band(np.array([5000.0, 20000.0]))
        assert a[0] == pytest.approx(b[0])


class TestMaskInvalidPixels:
    def test_in_range_values_untouched(self) -> None:
        result = mask_invalid_pixels(np.array([0.0, 0.5, 1.0]), valid_range=(0.0, 1.0))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_out_of_range_values_masked(self) -> None:
        result = mask_invalid_pixels(np.array([-0.1, 0.5, 1.5]), valid_range=(0.0, 1.0))
        assert np.isnan(result[0])
        assert result[1] == 0.5
        assert np.isnan(result[2])

    def test_existing_nan_stays_nan(self) -> None:
        result = mask_invalid_pixels(np.array([np.nan, 0.5]), valid_range=(0.0, 1.0))
        assert np.isnan(result[0])

    def test_inverted_range_raises(self) -> None:
        with pytest.raises(ValueError):
            mask_invalid_pixels(np.array([0.5]), valid_range=(1.0, 0.0))

    def test_custom_range(self) -> None:
        result = mask_invalid_pixels(np.array([-50.0, 50.0, 150.0]), valid_range=(-100.0, 100.0))
        assert result[0] == -50.0
        assert result[1] == 50.0
        assert np.isnan(result[2])
