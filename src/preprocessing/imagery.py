"""
Raster band reading: turning a Cloud-Optimized GeoTIFF (COG) URL and an
AOI into a NumPy array.

Deliberately provider-agnostic: this module knows nothing about
Sentinel-2, STAC, or Earth Search — it just reads a windowed region from
any COG given its URL. src/ingestion/sentinel2.py is what knows how to
turn a STAC item into a COG URL for a given band; this module is what
turns that URL + an AOI into pixels. That separation means a future
Landsat or Sentinel-1 provider can reuse this exact function.

Returns raw digital numbers (DNs) as stored in the source file — NOT
reflectance. Sentinel-2 L2A COGs store DN scaled by 10000 (matching
src/remote_sensing/spectral.py's normalize_band() default scale_factor),
so the typical pipeline is:

    dn_array = read_band_window(asset_href, aoi)
    reflectance = normalize_band(dn_array)  # -> [0, 1]
"""

from __future__ import annotations

import numpy as np
import rasterio
import rasterio.windows
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds

from src.geospatial.aoi import AOI


class RasterReadError(Exception):
    """Raised when a COG can't be opened/read, or an AOI doesn't overlap it."""


def read_band_window(
    asset_href: str,
    aoi: AOI,
    unsigned: bool = True,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Read the pixels of a COG that fall within an AOI, reprojecting as needed.

    AOI coordinates are always WGS84 (EPSG:4326) lat/lon (see
    src/geospatial/aoi.py), but satellite COGs are typically stored in a
    local UTM projection. This function transforms the AOI into the
    raster's native CRS before computing the read window, so callers
    never have to think about projections.

    Args:
        asset_href: URL (http(s):// or s3://) of the COG to read.
        aoi: Area of Interest, in WGS84 lat/lon.
        unsigned: whether to access the source with unsigned (anonymous)
            S3 requests. True by default, matching the public
            requester-pays-free sentinel-cogs bucket Earth Search uses.
            Irrelevant for plain http(s):// URLs.
        fill_value: value used for any part of the requested window that
            falls outside the raster's actual extent (a boundless read).
            Defaults to 0, consistent with the nodata convention already
            used by normalize_band()/mask_invalid_pixels() elsewhere in
            GeoWatch.

    Returns:
        2D ndarray of raw digital numbers (native dtype of the source
        band — typically uint16 for Sentinel-2 L2A). Callers normalize
        to reflectance separately (see module docstring).

    Raises:
        RasterReadError: if the file can't be opened, or the AOI does
            not overlap the raster's extent at all.
    """
    env_options = {"AWS_NO_SIGN_REQUEST": "YES"} if unsigned else {}

    try:
        with rasterio.Env(**env_options):
            with rasterio.open(asset_href) as src:
                # Reproject the AOI's WGS84 bounds into the raster's own CRS.
                west, south, east, north = transform_bounds(
                    "EPSG:4326", src.crs, aoi.west, aoi.south, aoi.east, aoi.north
                )

                raster_west, raster_south, raster_east, raster_north = src.bounds
                overlaps = not (
                    east < raster_west
                    or west > raster_east
                    or north < raster_south
                    or south > raster_north
                )
                if not overlaps:
                    raise RasterReadError(
                        f"read_band_window: AOI {aoi.label!r} does not overlap "
                        f"the raster's extent at all (AOI in raster CRS: "
                        f"west={west:.2f}, south={south:.2f}, east={east:.2f}, "
                        f"north={north:.2f}; raster bounds: {src.bounds})."
                    )

                window = rasterio.windows.from_bounds(
                    west, south, east, north, transform=src.transform
                )
                data = src.read(
                    1,
                    window=window,
                    boundless=True,
                    fill_value=fill_value,
                )
        return data
    except RasterioIOError as exc:
        raise RasterReadError(f"read_band_window: could not open {asset_href!r}: {exc}") from exc
