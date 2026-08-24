"""
PostgreSQL + PostGIS-backed event store (Phase 10/13).

Replaces InMemoryEventStore (src/monitoring/events.py) for real usage,
behind the same add_event() / get_event() / list_events() / count()
interface, so nothing that already depends on that interface needs to
change. Additionally exposes find_events_within_bbox() and
find_events_near(), which InMemoryEventStore deliberately does not
implement — these use real PostGIS spatial functions (ST_MakeEnvelope,
ST_DWithin) rather than reimplementing geometry math in Python, which is
the actual reason this system uses PostGIS instead of a plain key-value
store.

Requires: sqlalchemy, geoalchemy2, psycopg2-binary (see requirements.txt).
Requires a running PostgreSQL instance with the PostGIS extension
enabled (see docker-compose.yml's `db` service).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np
from geoalchemy2 import Geography, Geometry
from sqlalchemy import Column, DateTime, Float, String, create_engine, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.monitoring.events import EventStore
from src.types import (
    ConfidenceScore,
    Evidence,
    EvidenceLevel,
    EventStatus,
    EventType,
    GeoWatchEvent,
)


def _json_safe(value: Any) -> Any:
    """Recursively convert a value into something the JSON/JSONB column can store.

    Needed because event metadata frequently contains numpy scalar types
    (e.g. from a pandas DataFrame produced by the FIRMS provider) and
    datetime objects, neither of which the stdlib json encoder handles
    by default.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


class Base(DeclarativeBase):
    pass


class EventRecord(Base):
    """SQLAlchemy model backing the geowatch_events table."""

    __tablename__ = "geowatch_events"

    event_id = Column(String, primary_key=True)
    event_type = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location = Column(Geometry(geometry_type="POINT", srid=4326), nullable=False)
    detected_at = Column(DateTime(timezone=True), nullable=False)
    observation_time = Column(DateTime(timezone=True), nullable=True)
    source = Column(String, nullable=False)
    evidence_level = Column(String, nullable=False)
    confidence_value = Column(Float, nullable=False)
    confidence_basis = Column(String, nullable=False)
    confidence_evidence_level = Column(String, nullable=False)
    status = Column(String, nullable=False)
    severity = Column(String, nullable=True)
    geometry_json = Column(JSONB, nullable=True)
    evidence_json = Column(JSONB, nullable=False)
    metadata_json = Column(JSONB, nullable=False)


class PostgresEventStore(EventStore):
    """PostGIS-backed implementation of the GeoWatch event store interface."""

    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    def add_event(self, event: GeoWatchEvent) -> None:
        with self._Session() as session:
            if session.get(EventRecord, event.event_id) is not None:
                raise ValueError(f"Event with id {event.event_id!r} already exists.")
            session.add(self._to_record(event))
            session.commit()

    def get_event(self, event_id: str) -> Optional[GeoWatchEvent]:
        with self._Session() as session:
            record = session.get(EventRecord, event_id)
            return self._to_domain(record) if record is not None else None

    def list_events(self) -> list[GeoWatchEvent]:
        with self._Session() as session:
            return [self._to_domain(r) for r in session.query(EventRecord).all()]

    def count(self) -> int:
        with self._Session() as session:
            return session.query(EventRecord).count()

    def find_events_within_bbox(
        self, west: float, south: float, east: float, north: float
    ) -> list[GeoWatchEvent]:
        """Find events whose location falls within a bounding box, using
        PostGIS's ST_MakeEnvelope + ST_Within rather than filtering in Python.
        """
        with self._Session() as session:
            envelope = func.ST_MakeEnvelope(west, south, east, north, 4326)
            records = (
                session.query(EventRecord)
                .filter(func.ST_Within(EventRecord.location, envelope))
                .all()
            )
            return [self._to_domain(r) for r in records]

    def find_events_near(
        self, latitude: float, longitude: float, radius_km: float
    ) -> list[GeoWatchEvent]:
        """Find events within radius_km of a point, using PostGIS's
        geography-cast ST_DWithin for accurate great-circle distance
        (not a flat-plane approximation).
        """
        with self._Session() as session:
            point_wkt = f"SRID=4326;POINT({longitude} {latitude})"
            records = (
                session.query(EventRecord)
                .filter(
                    func.ST_DWithin(
                        func.cast(EventRecord.location, Geography),
                        func.cast(point_wkt, Geography),
                        radius_km * 1000.0,
                    )
                )
                .all()
            )
            return [self._to_domain(r) for r in records]

    @staticmethod
    def _to_record(event: GeoWatchEvent) -> EventRecord:
        point_wkt = f"POINT({event.longitude} {event.latitude})"
        return EventRecord(
            event_id=event.event_id,
            event_type=event.event_type.value,
            latitude=event.latitude,
            longitude=event.longitude,
            location=point_wkt,
            detected_at=event.detected_at,
            observation_time=event.observation_time,
            source=event.source,
            evidence_level=event.evidence_level.value,
            confidence_value=event.confidence.value,
            confidence_basis=event.confidence.basis,
            confidence_evidence_level=event.confidence.evidence_level.value,
            status=event.status.value,
            severity=event.severity,
            geometry_json=_json_safe(event.geometry) if event.geometry else None,
            evidence_json=[
                {
                    "description": ev.description,
                    "level": ev.level.value,
                    "source": ev.source,
                    "observed_at": ev.observed_at.isoformat() if ev.observed_at else None,
                    "metadata": _json_safe(ev.metadata),
                }
                for ev in event.evidence
            ],
            metadata_json=_json_safe(event.metadata),
        )

    @staticmethod
    def _to_domain(record: EventRecord) -> GeoWatchEvent:
        evidence = [
            Evidence(
                description=ev["description"],
                level=EvidenceLevel(ev["level"]),
                source=ev["source"],
                observed_at=(
                    datetime.fromisoformat(ev["observed_at"]) if ev.get("observed_at") else None
                ),
                metadata=ev.get("metadata") or {},
            )
            for ev in record.evidence_json
        ]
        return GeoWatchEvent(
            event_id=record.event_id,
            event_type=EventType(record.event_type),
            latitude=record.latitude,
            longitude=record.longitude,
            detected_at=record.detected_at,
            observation_time=record.observation_time,
            source=record.source,
            evidence_level=EvidenceLevel(record.evidence_level),
            confidence=ConfidenceScore(
                value=record.confidence_value,
                basis=record.confidence_basis,
                evidence_level=EvidenceLevel(record.confidence_evidence_level),
            ),
            status=EventStatus(record.status),
            severity=record.severity,
            geometry=record.geometry_json,
            evidence=evidence,
            metadata=record.metadata_json or {},
        )
