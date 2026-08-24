"""
Area of Interest (AOI) — a bounding box GeoWatch monitors.

Deliberately generic: an AOI is just a bounding box and a human-readable
label. No region is hardcoded anywhere in GeoWatch — every provider
query and dashboard view takes an AOI as a parameter. Polygon-shaped
AOIs (as opposed to rectangular bounding boxes) are a roadmap item —
see Phase 6 in the project design notes — not needed for Milestone 3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AOI:
    """A rectangular Area of Interest.

    Coordinates follow the common (west, south, east, north) convention
    used by NASA FIRMS and most bounding-box APIs — i.e.
    (min_lon, min_lat, max_lon, max_lat).
    """

    label: str
    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not (-180.0 <= self.west <= 180.0) or not (-180.0 <= self.east <= 180.0):
            raise ValueError(f"AOI longitude out of range: west={self.west}, east={self.east}")
        if not (-90.0 <= self.south <= 90.0) or not (-90.0 <= self.north <= 90.0):
            raise ValueError(f"AOI latitude out of range: south={self.south}, north={self.north}")
        if self.west >= self.east:
            raise ValueError(f"AOI west ({self.west}) must be < east ({self.east})")
        if self.south >= self.north:
            raise ValueError(f"AOI south ({self.south}) must be < north ({self.north})")

    def as_bbox_tuple(self) -> tuple[float, float, float, float]:
        """Return (west, south, east, north), the format FIRMS and most APIs expect."""
        return (self.west, self.south, self.east, self.north)

    def as_firms_area_string(self) -> str:
        """Return the comma-separated 'west,south,east,north' string FIRMS's
        area API expects in its URL path."""
        return f"{self.west},{self.south},{self.east},{self.north}"

    def contains_point(self, latitude: float, longitude: float) -> bool:
        """Whether a point falls inside this AOI (inclusive bounds)."""
        return self.west <= longitude <= self.east and self.south <= latitude <= self.north

    @classmethod
    def from_point_radius(cls, label: str, latitude: float, longitude: float, radius_deg: float) -> "AOI":
        """Convenience constructor: a square AOI centered on a point.

        Note this uses a simple degree-based radius, not a true geodesic
        buffer — fine for Milestone 3's purposes (scoping a FIRMS query),
        not precise near the poles or for large radii.
        """
        if radius_deg <= 0:
            raise ValueError(f"radius_deg must be positive, got {radius_deg}")
        return cls(
            label=label,
            west=longitude - radius_deg,
            south=latitude - radius_deg,
            east=longitude + radius_deg,
            north=latitude + radius_deg,
        )
