"""
NBR (Normalized Burn Ratio) and dNBR (delta NBR, pre/post-fire).

    NBR  = (NIR - SWIR) / (NIR + SWIR)
    dNBR = NBR_pre - NBR_post

Implementation lands in Milestone 1 alongside NDVI. Burn-severity
classification from dNBR thresholds is documented as methodology-
dependent, not universal ground truth, per the project spec.
"""

from __future__ import annotations

import numpy as np


def compute_nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Compute NBR from NIR and SWIR bands. Implemented in Milestone 1."""
    raise NotImplementedError("compute_nbr will be implemented in Milestone 1.")


def compute_dnbr(nbr_pre: np.ndarray, nbr_post: np.ndarray) -> np.ndarray:
    """Compute dNBR from pre- and post-fire NBR arrays. Implemented in Milestone 1."""
    raise NotImplementedError("compute_dnbr will be implemented in Milestone 1.")
