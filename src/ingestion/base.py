"""
SatelliteDataProvider abstraction (Phase 5 of the GeoWatch roadmap).

GeoWatch must never be hard-coded around one data source. Every concrete
provider (NASA FIRMS, Sentinel-2, Landsat, Sentinel-1 later) implements
this interface. Milestone 3 will implement FIRMSProvider first, since
it is the simplest (no imagery download, just tabular fire detections).

Not implemented yet — this file only defines the contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional


class SatelliteDataProvider(ABC):
    """Common interface every Earth-observation data source must implement."""

    name: str

    @abstractmethod
    def search_observations(
        self,
        bbox: tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: Optional[float] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return available observations matching the given filters."""
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, observation_id: str) -> dict[str, Any]:
        """Return metadata for a single observation."""
        raise NotImplementedError

    @abstractmethod
    def download_asset(self, observation_id: str, asset_key: str, destination: str) -> str:
        """Download a specific asset (e.g. a spectral band) to disk. Returns the local path."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self, observation_id: str) -> dict[str, Any]:
        """Return provider-specific metadata for an observation."""
        raise NotImplementedError
