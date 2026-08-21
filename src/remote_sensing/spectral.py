"""
Shared spectral-band utilities.

Everything in remote_sensing/ndvi.py and remote_sensing/nbr.py is built
on the same two primitives defined here:

    - safe_divide(): a division that never raises, never silently
      produces +/-inf, and treats "no signal at all" (0/0) as no-data
      rather than a computed value of zero.
    - compute_normalized_difference(): the (a - b) / (a + b) pattern
      shared by NDVI, NBR, and every other normalized-difference index
      GeoWatch will add later.

Design decisions, stated explicitly so they're easy to audit later:

    - All inputs are cast to float64 before any arithmetic, so integer
      (e.g. uint16 DN) inputs never silently truncate or overflow.
    - 0/0 -> NaN (no data), not 0. A pixel where both bands read zero
      carries no spectral information, so treating it as "zero index
      value" would be scientifically misleading.
    - x/0 where x != 0 -> NaN as well. This is a degenerate/invalid
      pixel, not a meaningful infinite index value.
    - NaN in any input propagates to NaN in the output. GeoWatch never
      invents a value for missing data.
"""

from __future__ import annotations

import numpy as np


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise numerator / denominator with deterministic no-data handling.

    Returns NaN (instead of raising, or returning +/-inf, or silently
    returning 0) wherever the denominator is zero.

    Args:
        numerator: array-like, will be cast to float64.
        denominator: array-like, same shape as numerator, cast to float64.

    Returns:
        float64 ndarray, same shape as the inputs.

    Raises:
        ValueError: if the input shapes do not match.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)

    if num.shape != den.shape:
        raise ValueError(
            f"safe_divide: shape mismatch, numerator {num.shape} vs denominator {den.shape}"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(
            num,
            den,
            out=np.full_like(num, np.nan, dtype=np.float64),
            where=(den != 0),
        )
    return result


def compute_normalized_difference(band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    """Compute (band_a - band_b) / (band_a + band_b), the shared pattern
    behind NDVI, NBR, and other normalized-difference spectral indices.

    Args:
        band_a: array-like, e.g. NIR.
        band_b: array-like, e.g. RED or SWIR. Same shape as band_a.

    Returns:
        float64 ndarray of index values, NaN where both bands sum to zero
        or where either input is NaN.

    Raises:
        ValueError: if the input shapes do not match.
    """
    a = np.asarray(band_a, dtype=np.float64)
    b = np.asarray(band_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"compute_normalized_difference: shape mismatch, {a.shape} vs {b.shape}"
        )

    return safe_divide(a - b, a + b)


def normalize_band(band: np.ndarray, scale_factor: float = 10000.0) -> np.ndarray:
    """Scale a raw digital-number band to approximate surface reflectance [0, 1].

    Many open optical products (e.g. Sentinel-2 L2A) distribute reflectance
    scaled by 10000 as integer DNs. This divides by that factor and clips
    to [0, 1] so downstream index calculations receive physically
    plausible reflectance values instead of raw DNs.

    This is NOT a min-max stretch — it does not depend on the specific
    image's value range, so the same input value always maps to the same
    output value regardless of what else is in the array. That
    determinism matters for reproducible index calculations.

    Args:
        band: array-like of raw digital numbers.
        scale_factor: divisor mapping the product's DN range to [0, 1].
            Defaults to 10000, the common Sentinel-2 L2A scaling factor.

    Returns:
        float64 ndarray, values clipped to [0, 1]. NaN inputs stay NaN.
    """
    if scale_factor <= 0:
        raise ValueError(f"normalize_band: scale_factor must be positive, got {scale_factor}")

    arr = np.asarray(band, dtype=np.float64)
    scaled = arr / scale_factor
    return np.clip(scaled, 0.0, 1.0)


def mask_invalid_pixels(
    band: np.ndarray, valid_range: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """Replace out-of-range values with NaN, leaving in-range values untouched.

    Args:
        band: array-like of values to validate.
        valid_range: inclusive (min, max) considered physically valid.
            Defaults to (0.0, 1.0), appropriate for reflectance.

    Returns:
        float64 ndarray, same shape as input. Values outside valid_range
        (and any pre-existing NaN) become NaN; in-range values pass through.

    Raises:
        ValueError: if valid_range[0] > valid_range[1].
    """
    low, high = valid_range
    if low > high:
        raise ValueError(f"mask_invalid_pixels: invalid range, min {low} > max {high}")

    arr = np.asarray(band, dtype=np.float64)
    out_of_range = (arr < low) | (arr > high)
    return np.where(out_of_range, np.nan, arr)
