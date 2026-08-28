"""
Tests for src/preprocessing/imagery.py

These use a REAL local GeoTIFF created with rasterio, in a genuine UTM
projection (EPSG:32735, UTM zone 35S — covers part of Southern Africa,
matching the AOIs used elsewhere in this project's notebooks). This
exercises the actual reprojection and windowed-read machinery against
real GDAL code, not a mock — the same code path that runs against a
real Sentinel-2 COG, just pointed at a small file on local disk instead
of S3.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.geospatial.aoi import AOI
from src.preprocessing.imagery import RasterReadError, read_band_window

UTM_35S = "EPSG:32735"


@pytest.fixture
def synthetic_geotiff(tmp_path):
    """A small real GeoTIFF: 100x100 pixels, 10m resolution, UTM 35S,
    origin at easting=500000, northing=7800000 (a real UTM coordinate,
    roughly in Zimbabwe/Zambia). Pixel value = row index, so reading a
    known window has a known, checkable result.
    """
    path = tmp_path / "synthetic_scene.tif"
    size = 100
    resolution = 10.0
    origin_x, origin_y = 500_000.0, 7_800_000.0

    transform = from_origin(origin_x, origin_y, resolution, resolution)
    data = np.tile(np.arange(size, dtype=np.uint16).reshape(size, 1), (1, size))

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype=np.uint16,
        crs=UTM_35S,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return path, transform, size, resolution


def _aoi_for_utm_window(transform, resolution: float, size: int, col0, col1, row0, row1) -> AOI:
    """Build a WGS84 AOI whose reprojected bounds land on a known pixel
    window of the synthetic UTM raster above."""
    import pyproj

    utm_to_wgs84 = pyproj.Transformer.from_crs(UTM_35S, "EPSG:4326", always_xy=True)

    easting_west = transform.c + col0 * resolution
    easting_east = transform.c + col1 * resolution
    northing_north = transform.f - row0 * resolution
    northing_south = transform.f - row1 * resolution

    lon_west, lat_south = utm_to_wgs84.transform(easting_west, northing_south)
    lon_east, lat_north = utm_to_wgs84.transform(easting_east, northing_north)

    return AOI(label="test window", west=lon_west, south=lat_south, east=lon_east, north=lat_north)


class TestReadBandWindow:
    def test_reads_correct_window_values(self, synthetic_geotiff) -> None:
        path, transform, size, resolution = synthetic_geotiff
        # Rows 10-30 -> pixel values 10..29 in every column, per our synthetic data.
        aoi = _aoi_for_utm_window(transform, resolution, size, col0=0, col1=size, row0=10, row1=30)

        result = read_band_window(str(path), aoi, unsigned=False)

        assert result.shape[0] == pytest.approx(20, abs=1)  # ~20 rows requested
        # Every value in the read window should fall within the known row range.
        assert result.min() >= 9  # allow +/-1 pixel edge tolerance from reprojection rounding
        assert result.max() <= 30

    def test_full_scene_read(self, synthetic_geotiff) -> None:
        path, transform, size, resolution = synthetic_geotiff
        aoi = _aoi_for_utm_window(transform, resolution, size, col0=0, col1=size, row0=0, row1=size)

        result = read_band_window(str(path), aoi, unsigned=False)

        assert result.shape[0] == pytest.approx(size, abs=1)
        assert result.shape[1] == pytest.approx(size, abs=1)

    def test_returns_native_dtype(self, synthetic_geotiff) -> None:
        path, transform, size, resolution = synthetic_geotiff
        aoi = _aoi_for_utm_window(transform, resolution, size, col0=0, col1=size, row0=0, row1=size)

        result = read_band_window(str(path), aoi, unsigned=False)
        assert result.dtype == np.uint16

    def test_nonoverlapping_aoi_raises(self, synthetic_geotiff) -> None:
        path, transform, size, resolution = synthetic_geotiff
        # Somewhere in Siberia — nowhere near the synthetic scene's UTM 35S location.
        far_away_aoi = AOI(label="far away", west=100.0, south=60.0, east=101.0, north=61.0)

        with pytest.raises(RasterReadError):
            read_band_window(str(path), far_away_aoi, unsigned=False)

    def test_nonexistent_file_raises_raster_read_error(self) -> None:
        aoi = AOI(label="test", west=10.0, south=-20.0, east=11.0, north=-19.0)
        with pytest.raises(RasterReadError):
            read_band_window("/nonexistent/path/does_not_exist.tif", aoi, unsigned=False)

    def test_partial_overlap_uses_fill_value(self, synthetic_geotiff) -> None:
        path, transform, size, resolution = synthetic_geotiff
        # A window straddling the raster's edge: half inside, half outside.
        aoi = _aoi_for_utm_window(
            transform, resolution, size, col0=size - 10, col1=size + 10, row0=0, row1=10
        )

        result = read_band_window(str(path), aoi, unsigned=False, fill_value=0.0)
        # Some pixels should be the fill value (0) since the window extends past the raster.
        assert 0 in result

    def test_custom_fill_value_applied(self, synthetic_geotiff) -> None:
        path, transform, size, resolution = synthetic_geotiff
        aoi = _aoi_for_utm_window(
            transform, resolution, size, col0=size - 5, col1=size + 15, row0=0, row1=10
        )

        result = read_band_window(str(path), aoi, unsigned=False, fill_value=9999)
        assert 9999 in result
