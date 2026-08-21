"""
Event store (Phase 10 / Phase 13).

Milestone 0 ships a minimal in-memory store purely so the GeoWatchEvent
model (src/types.py) has something real backing it and can be unit
tested from day one. Milestone 3 onward will swap this for a
PostgreSQL + PostGIS-backed store behind the SAME interface below, so
nothing calling add_event()/get_event()/list_events() needs to change.
"""

from __future__ import annotations

from typing import Optional

from src.types import GeoWatchEvent


class InMemoryEventStore:
    """Temporary event store. Not for production use — no persistence.

    Replaced by a PostGIS-backed store in Milestone 3+; this class exists
    so the rest of the system can be built and tested against a stable
    interface before the database layer lands.
    """

    def __init__(self) -> None:
        self._events: dict[str, GeoWatchEvent] = {}

    def add_event(self, event: GeoWatchEvent) -> None:
        if event.event_id in self._events:
            raise ValueError(f"Event with id {event.event_id!r} already exists.")
        self._events[event.event_id] = event

    def get_event(self, event_id: str) -> Optional[GeoWatchEvent]:
        return self._events.get(event_id)

    def list_events(self) -> list[GeoWatchEvent]:
        return list(self._events.values())

    def count(self) -> int:
        return len(self._events)
