"""
Affected-area calculations.

Converts a pixel-based mask (e.g. a burned-area mask from
src/detection/wildfire.py) into physical area units. Deliberately
simple for Milestone 2: assumes square pixels of a known ground
resolution (correct for common gridded products like Sentinel-2/Landsat
in their native projection) rather than doing per-pixel geodetic area
calculation. If GeoWatch later works with data in geographic
(lat/lon) coordinates where pixel area varies with latitude, this
module will need a proper geodetic area function — noted here so it
isn't forgotten, not implemented until it's actually needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HECTARES_PER_SQUARE_METER = 1.0 / 10_000.0
SQUARE_KM_PER_SQUARE_METER = 1.0 / 1_000_000.0


@dataclass
class AreaStats:
    """Area breakdown for a boolean mask over a known-resolution grid."""

    total_pixels: int
    affected_pixels: int
    pixel_resolution_m: float
    affected_area_hectares: float
    affected_area_km2: float
    affected_percentage: float


def pixels_to_hectares(pixel_count: int, pixel_resolution_m: float) -> float:
    """Convert a count of square pixels to hectares.

    Args:
        pixel_count: number of pixels.
        pixel_resolution_m: the ground length of one pixel edge, in meters
            (e.g. 10.0 for Sentinel-2 10m bands, 30.0 for Landsat).

    Returns:
        Area in hectares.

    Raises:
        ValueError: if pixel_count is negative or pixel_resolution_m is not positive.
    """
    if pixel_count < 0:
        raise ValueError(f"pixels_to_hectares: pixel_count must be >= 0, got {pixel_count}")
    if pixel_resolution_m <= 0:
        raise ValueError(
            f"pixels_to_hectares: pixel_resolution_m must be positive, got {pixel_resolution_m}"
        )

    pixel_area_m2 = pixel_resolution_m ** 2
    return pixel_count * pixel_area_m2 * HECTARES_PER_SQUARE_METER


def compute_area_stats(mask: np.ndarray, pixel_resolution_m: float) -> AreaStats:
    """Compute affected-area statistics for a boolean mask.

    Args:
        mask: boolean array-like, True where a pixel is "affected"
            (e.g. burned). NaN-derived False values (i.e. no-data pixels
            that were resolved to False upstream) are counted as
            unaffected, matching the convention used by
            detection.wildfire.detect_burned_area().
        pixel_resolution_m: ground length of one pixel edge, in meters.

    Returns:
        AreaStats with pixel counts, hectares, km^2, and percentage affected.

    Raises:
        ValueError: if pixel_resolution_m is not positive.
    """
    if pixel_resolution_m <= 0:
        raise ValueError(
            f"compute_area_stats: pixel_resolution_m must be positive, got {pixel_resolution_m}"
        )

    arr = np.asarray(mask, dtype=bool)
    total_pixels = int(arr.size)
    affected_pixels = int(np.count_nonzero(arr))

    affected_hectares = pixels_to_hectares(affected_pixels, pixel_resolution_m)
    affected_km2 = affected_hectares / 100.0  # 1 km^2 = 100 hectares
    affected_percentage = (
        (affected_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    )

    return AreaStats(
        total_pixels=total_pixels,
        affected_pixels=affected_pixels,
        pixel_resolution_m=pixel_resolution_m,
        affected_area_hectares=affected_hectares,
        affected_area_km2=affected_km2,
        affected_percentage=affected_percentage,
    )
