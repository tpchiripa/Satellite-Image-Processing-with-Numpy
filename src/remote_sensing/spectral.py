"""
Shared spectral-band utilities: normalization, invalid-pixel masking,
and the reusable machinery NDVI/NBR (and future indices) build on.
Implemented in Milestone 1.
"""

from __future__ import annotations

import numpy as np


def normalize_band(band: np.ndarray) -> np.ndarray:
    """Normalize a single spectral band to [0, 1]. Implemented in Milestone 1."""
    raise NotImplementedError("normalize_band will be implemented in Milestone 1.")


def mask_invalid_pixels(band: np.ndarray, valid_range: tuple[float, float]) -> np.ndarray:
    """Mask out-of-range / invalid pixel values. Implemented in Milestone 1."""
    raise NotImplementedError("mask_invalid_pixels will be implemented in Milestone 1.")
