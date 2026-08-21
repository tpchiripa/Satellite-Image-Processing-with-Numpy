"""
NDVI (Normalized Difference Vegetation Index).

    NDVI = (NIR - RED) / (NIR + RED)

NDVI ranges from -1 to 1. Healthy, dense vegetation typically reads
0.6-0.9; sparse vegetation or stressed plants read lower; water, snow,
and bare soil often read near zero or negative. These are general
patterns, not universal thresholds — GeoWatch does not classify land
cover from NDVI value alone without additional context.
"""

from __future__ import annotations

import numpy as np

from src.remote_sensing.spectral import compute_normalized_difference


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute NDVI from NIR and RED reflectance bands.

    Args:
        nir: array-like of near-infrared reflectance values.
        red: array-like of red reflectance values, same shape as nir.

    Returns:
        float64 ndarray of NDVI values in [-1, 1]. NaN wherever nir and
        red are both zero (no data), wherever either input is NaN, or
        wherever both bands are simultaneously zero.

    Raises:
        ValueError: if nir and red do not have matching shapes.
    """
    return compute_normalized_difference(nir, red)
