"""
Wildfire detection: pre/post-fire comparison, burned-area masking, and
configurable burn-severity classification from dNBR (Phase 3/4).
Every classification returned here must carry an EvidenceLevel — see
src/types.py. Implemented starting Milestone 2.
"""

from __future__ import annotations
