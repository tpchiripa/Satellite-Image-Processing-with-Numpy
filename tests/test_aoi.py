"""Tests for src/geospatial/aoi.py"""

from __future__ import annotations

import pytest

from src.geospatial.aoi import AOI


class TestAOIValidation:
    def test_valid_aoi_constructs(self) -> None:
        aoi = AOI(label="test", west=-10.0, south=-5.0, east=10.0, north=5.0)
        assert aoi.label == "test"

    def test_west_must_be_less_than_east(self) -> None:
        with pytest.raises(ValueError):
            AOI(label="bad", west=10.0, south=-5.0, east=-10.0, north=5.0)

    def test_south_must_be_less_than_north(self) -> None:
        with pytest.raises(ValueError):
            AOI(label="bad", west=-10.0, south=5.0, east=10.0, north=-5.0)

    def test_longitude_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            AOI(label="bad", west=-200.0, south=-5.0, east=10.0, north=5.0)

    def test_latitude_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            AOI(label="bad", west=-10.0, south=-100.0, east=10.0, north=5.0)

    def test_aoi_is_immutable(self) -> None:
        aoi = AOI(label="test", west=-10.0, south=-5.0, east=10.0, north=5.0)
        with pytest.raises(Exception):
            aoi.west = 0.0  # type: ignore[misc]


class TestAOIConversions:
    def test_as_bbox_tuple(self) -> None:
        aoi = AOI(label="test", west=-10.0, south=-5.0, east=10.0, north=5.0)
        assert aoi.as_bbox_tuple() == (-10.0, -5.0, 10.0, 5.0)

    def test_as_firms_area_string(self) -> None:
        aoi = AOI(label="test", west=-10.0, south=-5.0, east=10.0, north=5.0)
        assert aoi.as_firms_area_string() == "-10.0,-5.0,10.0,5.0"


class TestAOIContainsPoint:
    def test_point_inside(self) -> None:
        aoi = AOI(label="test", west=-10.0, south=-5.0, east=10.0, north=5.0)
        assert aoi.contains_point(latitude=0.0, longitude=0.0) is True

    def test_point_outside(self) -> None:
        aoi = AOI(label="test", west=-10.0, south=-5.0, east=10.0, north=5.0)
        assert aoi.contains_point(latitude=50.0, longitude=50.0) is False

    def test_point_on_boundary_is_inside(self) -> None:
        aoi = AOI(label="test", west=-10.0, south=-5.0, east=10.0, north=5.0)
        assert aoi.contains_point(latitude=5.0, longitude=10.0) is True


class TestAOIFromPointRadius:
    def test_creates_square_around_point(self) -> None:
        aoi = AOI.from_point_radius(label="center", latitude=0.0, longitude=0.0, radius_deg=1.0)
        assert aoi.west == -1.0
        assert aoi.east == 1.0
        assert aoi.south == -1.0
        assert aoi.north == 1.0

    def test_nonpositive_radius_rejected(self) -> None:
        with pytest.raises(ValueError):
            AOI.from_point_radius(label="bad", latitude=0.0, longitude=0.0, radius_deg=0.0)
