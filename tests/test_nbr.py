"""Tests for src/remote_sensing/nbr.py"""

from __future__ import annotations

import numpy as np
import pytest

from src.remote_sensing.nbr import compute_dnbr, compute_nbr


class TestComputeNBR:
    def test_known_healthy_vegetation_value(self) -> None:
        # Healthy vegetation: high NIR, low SWIR -> high NBR.
        # NBR = (0.5 - 0.15) / (0.5 + 0.15) = 0.35 / 0.65
        nbr = compute_nbr(nir=np.array([0.5]), swir=np.array([0.15]))
        assert nbr[0] == pytest.approx(0.35 / 0.65)

    def test_known_burned_area_value(self) -> None:
        # Burned area: low NIR, high SWIR -> low/negative NBR.
        # NBR = (0.1 - 0.3) / (0.1 + 0.3) = -0.2 / 0.4 = -0.5
        nbr = compute_nbr(nir=np.array([0.1]), swir=np.array([0.3]))
        assert nbr[0] == pytest.approx(-0.5)

    def test_division_by_zero_returns_nan(self) -> None:
        nbr = compute_nbr(nir=np.array([0.0]), swir=np.array([0.0]))
        assert np.isnan(nbr[0])

    def test_nan_propagates(self) -> None:
        nbr = compute_nbr(nir=np.array([np.nan]), swir=np.array([0.2]))
        assert np.isnan(nbr[0])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_nbr(nir=np.array([0.5, 0.6]), swir=np.array([0.1]))


class TestComputeDNBR:
    def test_known_high_severity_change(self) -> None:
        # Pre-fire healthy NBR ~0.54, post-fire burned NBR ~-0.5
        nbr_pre = np.array([0.54])
        nbr_post = np.array([-0.5])
        dnbr = compute_dnbr(nbr_pre, nbr_post)
        assert dnbr[0] == pytest.approx(1.04)

    def test_no_change_gives_zero(self) -> None:
        nbr_pre = np.array([0.4])
        nbr_post = np.array([0.4])
        dnbr = compute_dnbr(nbr_pre, nbr_post)
        assert dnbr[0] == pytest.approx(0.0)

    def test_vegetation_regrowth_gives_negative_dnbr(self) -> None:
        # Post-fire NBR higher than pre-fire (e.g. comparing two already-
        # recovering scenes) should yield a negative dNBR, not be clipped.
        nbr_pre = np.array([0.1])
        nbr_post = np.array([0.4])
        dnbr = compute_dnbr(nbr_pre, nbr_post)
        assert dnbr[0] == pytest.approx(-0.3)

    def test_nan_in_pre_propagates(self) -> None:
        dnbr = compute_dnbr(np.array([np.nan]), np.array([0.2]))
        assert np.isnan(dnbr[0])

    def test_nan_in_post_propagates(self) -> None:
        dnbr = compute_dnbr(np.array([0.2]), np.array([np.nan]))
        assert np.isnan(dnbr[0])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_dnbr(np.array([0.1, 0.2]), np.array([0.1]))

    def test_end_to_end_from_bands(self) -> None:
        # Realistic pipeline: raw bands -> NBR -> dNBR, not just isolated
        # unit-level values.
        nir_pre = np.array([0.5, 0.5])
        swir_pre = np.array([0.15, 0.15])
        nir_post = np.array([0.1, 0.5])
        swir_post = np.array([0.3, 0.15])

        nbr_pre = compute_nbr(nir_pre, swir_pre)
        nbr_post = compute_nbr(nir_post, swir_post)
        dnbr = compute_dnbr(nbr_pre, nbr_post)

        # Pixel 0 burned (large positive dNBR); pixel 1 unchanged (~0).
        assert dnbr[0] > 0.5
        assert dnbr[1] == pytest.approx(0.0)
