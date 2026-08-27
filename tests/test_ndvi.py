"""Tests for src/remote_sensing/ndvi.py"""

from __future__ import annotations

import numpy as np
import pytest

from src.remote_sensing.ndvi import compute_dndvi, compute_ndvi


class TestComputeNDVI:
    def test_known_healthy_vegetation_value(self) -> None:
        # Typical healthy vegetation: high NIR, low RED.
        # NDVI = (0.5 - 0.1) / (0.5 + 0.1) = 0.6667
        ndvi = compute_ndvi(nir=np.array([0.5]), red=np.array([0.1]))
        assert ndvi[0] == pytest.approx(2 / 3)

    def test_known_bare_soil_value(self) -> None:
        # Bare soil: NIR and RED reflectance are closer together, low NDVI.
        # NDVI = (0.3 - 0.25) / (0.3 + 0.25) = 0.0909
        ndvi = compute_ndvi(nir=np.array([0.3]), red=np.array([0.25]))
        assert ndvi[0] == pytest.approx(0.05 / 0.55)

    def test_water_gives_low_or_negative_ndvi(self) -> None:
        # Water: low NIR, slightly higher RED -> negative NDVI.
        ndvi = compute_ndvi(nir=np.array([0.05]), red=np.array([0.08]))
        assert ndvi[0] < 0

    def test_division_by_zero_returns_nan(self) -> None:
        ndvi = compute_ndvi(nir=np.array([0.0]), red=np.array([0.0]))
        assert np.isnan(ndvi[0])

    def test_nan_input_propagates(self) -> None:
        ndvi = compute_ndvi(nir=np.array([np.nan]), red=np.array([0.1]))
        assert np.isnan(ndvi[0])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_ndvi(nir=np.array([0.5, 0.6]), red=np.array([0.1]))

    def test_2d_image_array(self) -> None:
        nir = np.array([[0.5, 0.0], [0.3, 0.05]])
        red = np.array([[0.1, 0.0], [0.25, 0.08]])
        ndvi = compute_ndvi(nir, red)
        assert ndvi.shape == (2, 2)
        assert ndvi[0, 0] == pytest.approx(2 / 3)
        assert np.isnan(ndvi[0, 1])

    def test_result_within_valid_ndvi_bounds(self) -> None:
        rng = np.random.default_rng(7)
        nir = rng.uniform(0, 1, size=500)
        red = rng.uniform(0, 1, size=500)
        mask = (nir + red) > 0
        ndvi = compute_ndvi(nir[mask], red[mask])
        assert np.all(ndvi >= -1.0 - 1e-9)
        assert np.all(ndvi <= 1.0 + 1e-9)

    def test_accepts_plain_python_lists(self) -> None:
        # Convenience: callers shouldn't be forced to pre-wrap in np.array.
        ndvi = compute_ndvi(nir=[0.5], red=[0.1])
        assert ndvi[0] == pytest.approx(2 / 3)


class TestComputeDNDVI:
    def test_known_decline_value(self) -> None:
        # Baseline healthy (0.7), recent stressed (0.3) -> positive dNDVI = decline
        dndvi = compute_dndvi(ndvi_pre=np.array([0.7]), ndvi_post=np.array([0.3]))
        assert dndvi[0] == pytest.approx(0.4)

    def test_no_change_gives_zero(self) -> None:
        dndvi = compute_dndvi(ndvi_pre=np.array([0.5]), ndvi_post=np.array([0.5]))
        assert dndvi[0] == pytest.approx(0.0)

    def test_vegetation_increase_gives_negative_dndvi(self) -> None:
        # Recent greener than baseline (e.g. regrowth/recovery) -> negative dNDVI
        dndvi = compute_dndvi(ndvi_pre=np.array([0.2]), ndvi_post=np.array([0.6]))
        assert dndvi[0] == pytest.approx(-0.4)

    def test_nan_in_pre_propagates(self) -> None:
        dndvi = compute_dndvi(np.array([np.nan]), np.array([0.3]))
        assert np.isnan(dndvi[0])

    def test_nan_in_post_propagates(self) -> None:
        dndvi = compute_dndvi(np.array([0.3]), np.array([np.nan]))
        assert np.isnan(dndvi[0])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_dndvi(np.array([0.1, 0.2]), np.array([0.1]))

    def test_end_to_end_from_bands(self) -> None:
        nir_pre, red_pre = np.array([0.5, 0.5]), np.array([0.1, 0.1])
        nir_post, red_post = np.array([0.15, 0.5]), np.array([0.2, 0.1])

        ndvi_pre = compute_ndvi(nir_pre, red_pre)
        ndvi_post = compute_ndvi(nir_post, red_post)
        dndvi = compute_dndvi(ndvi_pre, ndvi_post)

        # Pixel 0 shows real decline; pixel 1 unchanged (~0).
        assert dndvi[0] > 0.3
        assert dndvi[1] == pytest.approx(0.0)
