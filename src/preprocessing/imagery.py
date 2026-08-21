"""
Image loading / band-stacking utilities. Builds on the existing NumPy
foundation from notebooks/01_image_processing.ipynb (Phase 1). Wraps
band loading behind a stable function signature so remote_sensing/
never has to know whether pixels came from a sample image, a Sentinel-2
scene, or a Landsat scene. Implemented starting Milestone 1.
"""

from __future__ import annotations
