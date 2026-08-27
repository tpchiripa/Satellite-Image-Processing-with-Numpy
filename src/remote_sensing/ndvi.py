"""
NDVI (Normalized Difference Vegetation Index) and dNDVI (delta NDVI).

    NDVI  = (NIR - RED) / (NIR + RED)
    dNDVI = NDVI_pre - NDVI_post

NDVI ranges from -1 to 1. Healthy, dense vegetation typically reads
0.6-0.9; sparse vegetation or stressed plants read lower; water, snow,
and bare soil often read near zero or negative. These are general
patterns, not universal thresholds — GeoWatch does not classify land
cover from NDVI value alone without additional context.

dNDVI follows the same sign convention as dNBR (see nbr.py): a positive
value means NDVI declined from the baseline scene to the recent scene
(vegetation loss); negative means NDVI increased (vegetation gain /
regrowth). Decline-severity classification built on top of dNDVI is
handled in src/detection/vegetation.py, with the same "thresholds are
methodology, not universal fact" caveat that applies to wildfire severity.
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


def compute_dndvi(ndvi_pre: np.ndarray, ndvi_post: np.ndarray) -> np.ndarray:
    """Compute dNDVI (change in NDVI) between a baseline and a more recent scene.

    Args:
        ndvi_pre: array-like of baseline NDVI values (e.g. from compute_ndvi()).
        ndvi_post: array-like of more recent NDVI values, same shape as ndvi_pre.

    Returns:
        float64 ndarray of dNDVI values (ndvi_pre - ndvi_post). Positive
        means vegetation declined; negative means it increased. NaN
        wherever either input is NaN, propagating any no-data pixels
        from the underlying NDVI calculations.

    Raises:
        ValueError: if ndvi_pre and ndvi_post do not have matching shapes.
    """
    pre = np.asarray(ndvi_pre, dtype=np.float64)
    post = np.asarray(ndvi_post, dtype=np.float64)

    if pre.shape != post.shape:
        raise ValueError(
            f"compute_dndvi: shape mismatch, ndvi_pre {pre.shape} vs ndvi_post {post.shape}"
        )

    return pre - post
