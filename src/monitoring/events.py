"""
Event store (Phase 10 / Phase 13).

EventStore defines the interface every event store implements.
InMemoryEventStore is a non-persistent placeholder used in early tests
and anywhere a lightweight, dependency-free store is convenient.
PostgresEventStore (src/monitoring/postgres_store.py) is the real,
persistent PostGIS-backed implementation used from Milestone 3 onward —
kept in its own module since it pulls in SQLAlchemy/GeoAlchemy2/numpy
dependencies that InMemoryEventStore's callers shouldn't be forced to
install.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.types import GeoWatchEvent


class EventStore(ABC):
    """Interface every GeoWatch event store implements."""

    @abstractmethod
    def add_event(self, event: GeoWatchEvent) -> None:
        """Persist a new event. Raises ValueError if event_id already exists."""
        raise NotImplementedError

    @abstractmethod
    def get_event(self, event_id: str) -> Optional[GeoWatchEvent]:
        """Retrieve an event by id, or None if it doesn't exist."""
        raise NotImplementedError

    @abstractmethod
    def list_events(self) -> list[GeoWatchEvent]:
        """Return all stored events."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored events."""
        raise NotImplementedError


class InMemoryEventStore(EventStore):
    """Temporary event store. Not for production use — no persistence.

    Used for early tests and anywhere a lightweight, dependency-free
    store is convenient. PostgresEventStore is the persistent choice for
    real usage from Milestone 3 onward.
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
