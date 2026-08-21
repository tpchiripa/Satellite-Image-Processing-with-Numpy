"""
Generic land-disturbance / potential-mining-activity detection (Phase 9).
Per project principle, this module must NEVER emit a definitive
"illegal mining" label — only INFERRED-level phrasing such as
"potential mining-related land disturbance," generated via
EvidenceLevel.INFERRED.display_phrase(). Roadmap item — not in the
Milestone 0-5 MVP scope.
"""

from __future__ import annotations
