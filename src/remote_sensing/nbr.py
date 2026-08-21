"""
NBR (Normalized Burn Ratio) and dNBR (delta NBR).

    NBR  = (NIR - SWIR) / (NIR + SWIR)
    dNBR = NBR_pre - NBR_post

NBR is sensitive to the moisture/structure change that fire causes:
healthy vegetation has high NIR and low SWIR reflectance (high NBR);
burned areas have the opposite pattern (low or negative NBR). A larger
dNBR indicates a greater change between the pre- and post-fire scenes.

IMPORTANT: dNBR-to-severity classification is methodology-dependent.
Commonly cited threshold tables (e.g. USGS/USFS FIREMON) were developed
for specific sensors, ecosystems, and normalization conventions,and
are NOT universal ground truth. Any severity classification built on
top of compute_dnbr() must make its thresholds explicit and
configurable rather than presenting a fixed cutoff as scientific fact
(this will be handled in Milestone 2, src/detection/wildfire.py).
"""

from __future__ import annotations

import numpy as np

from src.remote_sensing.spectral import compute_normalized_difference


def compute_nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Compute NBR from NIR and SWIR reflectance bands.

    Args:
        nir: array-like of near-infrared reflectance values.
        swir: array-like of short-wave-infrared reflectance values,
            same shape as nir.

    Returns:
        float64 ndarray of NBR values in [-1, 1]. NaN wherever nir and
        swir are both zero, or wherever either input is NaN.

    Raises:
        ValueError: if nir and swir do not have matching shapes.
    """
    return compute_normalized_difference(nir, swir)


def compute_dnbr(nbr_pre: np.ndarray, nbr_post: np.ndarray) -> np.ndarray:
    """Compute dNBR (change in NBR) between a pre-fire and post-fire scene.

    Args:
        nbr_pre: array-like of pre-fire NBR values (e.g. from compute_nbr()).
        nbr_post: array-like of post-fire NBR values, same shape as nbr_pre.

    Returns:
        float64 ndarray of dNBR values (nbr_pre - nbr_post). NaN wherever
        either input is NaN, propagating any no-data pixels from the
        underlying NBR calculations.

    Raises:
        ValueError: if nbr_pre and nbr_post do not have matching shapes.
    """
    pre = np.asarray(nbr_pre, dtype=np.float64)
    post = np.asarray(nbr_post, dtype=np.float64)

    if pre.shape != post.shape:
        raise ValueError(
            f"compute_dnbr: shape mismatch, nbr_pre {pre.shape} vs nbr_post {post.shape}"
        )

    return pre - post
