"""Tests for src/geospatial/area.py"""

from __future__ import annotations

import numpy as np
import pytest

from src.geospatial.area import compute_area_stats, pixels_to_hectares


class TestPixelsToHectares:
    def test_known_conversion_10m_pixels(self) -> None:
        # 100 pixels at 10m resolution = 100 * 100 m^2 = 10,000 m^2 = 1 hectare
        result = pixels_to_hectares(pixel_count=100, pixel_resolution_m=10.0)
        assert result == pytest.approx(1.0)

    def test_known_conversion_30m_pixels(self) -> None:
        # 1 pixel at 30m resolution = 900 m^2 = 0.09 hectares
        result = pixels_to_hectares(pixel_count=1, pixel_resolution_m=30.0)
        assert result == pytest.approx(0.09)

    def test_zero_pixels_gives_zero_area(self) -> None:
        assert pixels_to_hectares(0, 10.0) == 0.0

    def test_negative_pixel_count_raises(self) -> None:
        with pytest.raises(ValueError):
            pixels_to_hectares(-5, 10.0)

    def test_nonpositive_resolution_raises(self) -> None:
        with pytest.raises(ValueError):
            pixels_to_hectares(100, 0.0)
        with pytest.raises(ValueError):
            pixels_to_hectares(100, -10.0)


class TestComputeAreaStats:
    def test_known_stats(self) -> None:
        # 4x4 grid at 10m resolution = 16 pixels total, 4 affected.
        mask = np.array(
            [
                [True, True, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, True, True],
            ]
        )
        stats = compute_area_stats(mask, pixel_resolution_m=10.0)
        assert stats.total_pixels == 16
        assert stats.affected_pixels == 4
        assert stats.affected_area_hectares == pytest.approx(0.04)  # 4*100 m^2 = 400 m^2
        assert stats.affected_area_km2 == pytest.approx(0.0004)
        assert stats.affected_percentage == pytest.approx(25.0)

    def test_all_affected(self) -> None:
        mask = np.ones((3, 3), dtype=bool)
        stats = compute_area_stats(mask, pixel_resolution_m=10.0)
        assert stats.affected_percentage == pytest.approx(100.0)

    def test_none_affected(self) -> None:
        mask = np.zeros((3, 3), dtype=bool)
        stats = compute_area_stats(mask, pixel_resolution_m=10.0)
        assert stats.affected_pixels == 0
        assert stats.affected_area_hectares == 0.0
        assert stats.affected_percentage == 0.0

    def test_nonpositive_resolution_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_area_stats(np.array([[True]]), pixel_resolution_m=0.0)

    def test_km2_and_hectares_consistent(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        stats = compute_area_stats(mask, pixel_resolution_m=100.0)
        assert stats.affected_area_km2 == pytest.approx(stats.affected_area_hectares / 100.0)
