"""
NDVI (Normalized Difference Vegetation Index).

    NDVI = (NIR - RED) / (NIR + RED)

Implementation lands in Milestone 1, with deterministic, unit-tested
handling of division-by-zero, NaN, and infinity per the project spec.
Deliberately left unimplemented in Milestone 0 (repo restructure only).
"""

from __future__ import annotations

import numpy as np


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute NDVI from NIR and RED bands. Implemented in Milestone 1."""
    raise NotImplementedError("compute_ndvi will be implemented in Milestone 1.")
