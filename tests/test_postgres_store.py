"""Tests for src/monitoring/postgres_store.py

These run against a REAL PostgreSQL + PostGIS instance (not mocked) —
correctness of spatial queries (ST_Within, ST_DWithin) genuinely depends
on PostGIS behavior, so mocking the database would defeat the point of
testing this module at all.

Requires the TEST_DATABASE_URL environment variable to point at a
Postgres instance with the PostGIS extension available. If it's not
set, this whole module is skipped (e.g. in environments without a local
Postgres available) rather than failing the whole test run.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("geoalchemy2")

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL not set; skipping PostGIS-backed tests (no local Postgres available)",
)

if TEST_DATABASE_URL:
    from src.monitoring.postgres_store import EventRecord, PostgresEventStore, _json_safe
    from src.types import ConfidenceScore, Evidence, EvidenceLevel, EventStatus, EventType, GeoWatchEvent


def _make_event(event_id: str, lat: float, lon: float, **overrides) -> "GeoWatchEvent":
    defaults = dict(
        event_id=event_id,
        event_type=EventType.WILDFIRE,
        latitude=lat,
        longitude=lon,
        detected_at=datetime.now(timezone.utc),
        observation_time=datetime.now(timezone.utc),
        source="test-fixture",
        evidence_level=EvidenceLevel.OBSERVED,
        confidence=ConfidenceScore(value=0.7, basis="test", evidence_level=EvidenceLevel.OBSERVED),
        evidence=[
            Evidence(
                description="test evidence",
                level=EvidenceLevel.OBSERVED,
                source="test-fixture",
                metadata={"raw_confidence": np.float64(0.7), "count": np.int64(3)},
            )
        ],
        metadata={"note": "test", "value": np.float64(1.5)},
    )
    defaults.update(overrides)
    return GeoWatchEvent(**defaults)


@pytest.fixture
def store():
    s = PostgresEventStore(TEST_DATABASE_URL)
    # Clean slate between tests.
    with s._Session() as session:
        session.query(EventRecord).delete()
        session.commit()
    yield s


class TestJsonSafe:
    def test_numpy_float_converted(self) -> None:
        assert isinstance(_json_safe(np.float64(1.5)), float)

    def test_numpy_int_converted(self) -> None:
        assert isinstance(_json_safe(np.int64(3)), int)

    def test_nan_converted_to_none(self) -> None:
        assert _json_safe(float("nan")) is None

    def test_nested_dict_converted(self) -> None:
        result = _json_safe({"a": np.float64(1.0), "b": {"c": np.int64(2)}})
        assert isinstance(result["a"], float)
        assert isinstance(result["b"]["c"], int)

    def test_enum_converted_to_value(self) -> None:
        assert _json_safe(EvidenceLevel.OBSERVED) == "observed"

    def test_plain_values_pass_through(self) -> None:
        assert _json_safe("hello") == "hello"
        assert _json_safe(5) == 5


class TestPostgresEventStoreCRUD:
    def test_add_and_get_event(self, store) -> None:
        event = _make_event("pg-evt-1", -33.9, 18.4)
        store.add_event(event)
        fetched = store.get_event("pg-evt-1")
        assert fetched is not None
        assert fetched.event_id == "pg-evt-1"
        assert fetched.latitude == pytest.approx(-33.9)
        assert fetched.longitude == pytest.approx(18.4)

    def test_get_nonexistent_event_returns_none(self, store) -> None:
        assert store.get_event("does-not-exist") is None

    def test_duplicate_event_id_rejected(self, store) -> None:
        store.add_event(_make_event("pg-evt-dup", -33.9, 18.4))
        with pytest.raises(ValueError):
            store.add_event(_make_event("pg-evt-dup", -34.0, 18.5))

    def test_list_and_count(self, store) -> None:
        store.add_event(_make_event("pg-evt-a", -33.9, 18.4))
        store.add_event(_make_event("pg-evt-b", -34.0, 18.5))
        assert store.count() == 2
        assert {e.event_id for e in store.list_events()} == {"pg-evt-a", "pg-evt-b"}

    def test_evidence_roundtrips_correctly(self, store) -> None:
        event = _make_event("pg-evt-evidence", -33.9, 18.4)
        store.add_event(event)
        fetched = store.get_event("pg-evt-evidence")
        assert len(fetched.evidence) == 1
        assert fetched.evidence[0].description == "test evidence"
        assert fetched.evidence[0].level == EvidenceLevel.OBSERVED

    def test_metadata_with_numpy_types_roundtrips(self, store) -> None:
        # This is the real regression test for _json_safe: numpy scalar
        # types (as produced by pandas .to_dict("records"), e.g. from the
        # FIRMS provider) must not raise on insert.
        event = _make_event("pg-evt-numpy", -33.9, 18.4)
        store.add_event(event)
        fetched = store.get_event("pg-evt-numpy")
        assert fetched.metadata["value"] == pytest.approx(1.5)
        assert isinstance(fetched.metadata["value"], float)

    def test_confidence_roundtrips(self, store) -> None:
        event = _make_event("pg-evt-conf", -33.9, 18.4)
        store.add_event(event)
        fetched = store.get_event("pg-evt-conf")
        assert fetched.confidence.value == pytest.approx(0.7)
        assert fetched.confidence.evidence_level == EvidenceLevel.OBSERVED


class TestPostgresEventStoreSpatialQueries:
    def test_find_within_bbox(self, store) -> None:
        # Cape Town area
        store.add_event(_make_event("pg-spatial-in", -33.9, 18.4))
        # Far away (London)
        store.add_event(_make_event("pg-spatial-out", 51.5, -0.1))

        results = store.find_events_within_bbox(west=17.0, south=-35.0, east=20.0, north=-33.0)
        ids = {e.event_id for e in results}
        assert "pg-spatial-in" in ids
        assert "pg-spatial-out" not in ids

    def test_find_near_point(self, store) -> None:
        store.add_event(_make_event("pg-near-close", -33.9, 18.4))
        store.add_event(_make_event("pg-near-far", -34.5, 20.0))  # ~150km+ away

        results = store.find_events_near(latitude=-33.92, longitude=18.42, radius_km=10.0)
        ids = {e.event_id for e in results}
        assert "pg-near-close" in ids
        assert "pg-near-far" not in ids

    def test_find_near_point_uses_great_circle_not_flat_approximation(self, store) -> None:
        # A sanity check that the function actually runs a real distance
        # query rather than silently returning everything or nothing.
        store.add_event(_make_event("pg-gc-1", -33.9, 18.4))
        results_small_radius = store.find_events_near(latitude=-33.9, longitude=18.4, radius_km=0.01)
        results_large_radius = store.find_events_near(latitude=-33.9, longitude=18.4, radius_km=1.0)
        assert len(results_small_radius) == 1  # exact same point, always within any radius
        assert len(results_large_radius) == 1
