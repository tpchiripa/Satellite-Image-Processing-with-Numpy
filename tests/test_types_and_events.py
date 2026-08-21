"""
Milestone 0 tests.

These don't test any remote-sensing science yet (there isn't any until
Milestone 1) — they prove two things that matter architecturally:

1. The EvidenceLevel / GeoWatchEvent contract behaves as designed
   (confidence bounds enforced, evidence-level phrasing works).
2. The test harness (pytest, src/ imports) is wired correctly, so every
   later milestone can add tests with zero setup friction.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.monitoring.events import InMemoryEventStore
from src.types import (
    ConfidenceScore,
    Evidence,
    EvidenceLevel,
    EventStatus,
    EventType,
    GeoWatchEvent,
)


def _make_event(event_id: str = "evt-001") -> GeoWatchEvent:
    return GeoWatchEvent(
        event_id=event_id,
        event_type=EventType.WILDFIRE,
        latitude=-33.9249,
        longitude=18.4241,
        detected_at=datetime.now(timezone.utc),
        observation_time=datetime.now(timezone.utc),
        source="test-fixture",
        evidence_level=EvidenceLevel.DETECTED,
        confidence=ConfidenceScore(
            value=0.8, basis="synthetic test value", evidence_level=EvidenceLevel.DETECTED
        ),
        evidence=[
            Evidence(
                description="Synthetic dNBR value exceeded severity threshold",
                level=EvidenceLevel.DETECTED,
                source="test-fixture",
            )
        ],
    )


class TestEvidenceLevel:
    def test_display_phrase_is_defined_for_every_level(self) -> None:
        for level in EvidenceLevel:
            phrase = level.display_phrase()
            assert isinstance(phrase, str)
            assert phrase.strip() != ""

    def test_inferred_never_reads_as_confirmed_fact(self) -> None:
        # This is the guardrail behind Section 25/31 of the project spec:
        # INFERRED-level claims must never render with confirmed-sounding language.
        assert "potential" in EvidenceLevel.INFERRED.display_phrase().lower()
        assert "confirmed" not in EvidenceLevel.INFERRED.display_phrase().lower()


class TestConfidenceScore:
    def test_valid_score_accepted(self) -> None:
        score = ConfidenceScore(value=0.5, basis="test", evidence_level=EvidenceLevel.DETECTED)
        assert score.value == 0.5

    @pytest.mark.parametrize("bad_value", [-0.01, 1.01, -5, 5])
    def test_out_of_range_score_rejected(self, bad_value: float) -> None:
        with pytest.raises(ValueError):
            ConfidenceScore(value=bad_value, basis="test", evidence_level=EvidenceLevel.DETECTED)


class TestGeoWatchEvent:
    def test_summary_line_uses_evidence_qualifier(self) -> None:
        event = _make_event()
        summary = event.summary_line()
        assert "Detected:" in summary
        assert "wildfire" in summary

    def test_default_status_is_new(self) -> None:
        event = _make_event()
        assert event.status == EventStatus.NEW


class TestInMemoryEventStore:
    def test_add_and_get_event(self) -> None:
        store = InMemoryEventStore()
        event = _make_event()
        store.add_event(event)
        assert store.get_event("evt-001") is event

    def test_duplicate_event_id_rejected(self) -> None:
        store = InMemoryEventStore()
        store.add_event(_make_event("evt-001"))
        with pytest.raises(ValueError):
            store.add_event(_make_event("evt-001"))

    def test_list_and_count(self) -> None:
        store = InMemoryEventStore()
        store.add_event(_make_event("evt-001"))
        store.add_event(_make_event("evt-002"))
        assert store.count() == 2
        assert {e.event_id for e in store.list_events()} == {"evt-001", "evt-002"}
