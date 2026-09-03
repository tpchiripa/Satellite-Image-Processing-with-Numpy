"""
Bridge from WildfireDetectionResult (a raster, see src/detection/wildfire.py)
to geolocated GeoWatchEvent objects, ready for PostgresEventStore and the
map — the same event model NASA FIRMS detections already use.

This is genuinely necessary, not cosmetic: a burned-area mask is a grid
of thousands of pixels, and turning every burned pixel into its own map
marker would be both useless (an unreadable wall of dots) and
misleading (implying thousands of independent detections instead of a
handful of actual burned regions). Connected-component clustering
groups contiguous burned pixels into distinct regions — one event per
real burned area, each carrying an aggregate location, size, and
dominant severity.

Every event produced here carries EvidenceLevel.DETECTED (a
deterministic dNBR-based calculation applied to satellite imagery),
matching WildfireDetectionResult's own evidence level — never upgraded
to something that sounds more certain than the underlying math.

On "confidence": this module's confidence score is NOT the same kind of
thing as NASA FIRMS's sensor-reported confidence, and callers should
not treat the two as directly comparable. Here, confidence is a
deterministic function of how far a region's mean dNBR sits past the
burn-detection threshold, relative to the "high severity" boundary — a
region barely past threshold scores low, one well into high-severity
territory scores high. It reflects DETECTION STRENGTH, not statistical
validation, and it is a distinct dimension from severity (a small
region deep in high-severity territory can have high confidence and
still be geographically small) — see the module docstring in
src/detection/wildfire.py and the Responsible Use section of the
project README for why these dimensions are kept separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import ndimage

from src.detection.wildfire import BurnSeverity, WildfireDetectionResult
from src.geospatial.aoi import AOI
from src.types import ConfidenceScore, Evidence, EventStatus, EventType, EvidenceLevel, GeoWatchEvent

# Confidence scaling bounds -- deliberately conservative (never 0 or 1.0),
# since this is a heuristic, not a statistically validated probability.
_MIN_REGION_CONFIDENCE = 0.3
_MAX_REGION_CONFIDENCE = 0.95


@dataclass
class BurnedRegion:
    """One connected burned-pixel region, before conversion to a GeoWatchEvent.
    Exposed separately from the event conversion so the clustering step can
    be tested and inspected independently of GeoWatchEvent's shape."""

    label: int
    pixel_count: int
    centroid_row: float
    centroid_col: float
    latitude: float
    longitude: float
    mean_dnbr: float
    dominant_severity: BurnSeverity
    severity_counts: dict[str, int]
    area_hectares: float


def _pixel_to_lonlat(row: float, col: float, aoi: AOI, shape: tuple[int, int]) -> tuple[float, float]:
    """Linear pixel-to-lat/lon conversion assuming the raster's rows/cols map
    directly onto the AOI's bounding box (row 0 = north edge, col 0 = west
    edge) -- valid for the flat-rectangle AOI model used throughout GeoWatch
    (see src/geospatial/aoi.py), and accurate at the AOI scales this project
    operates at. Not a substitute for a real affine transform if a future
    caller needs sub-pixel or large-scale geodetic precision."""
    height, width = shape
    lon = aoi.west + (col + 0.5) / width * (aoi.east - aoi.west)
    lat = aoi.north - (row + 0.5) / height * (aoi.north - aoi.south)
    return lon, lat


def find_burned_regions(
    result: WildfireDetectionResult,
    dnbr: np.ndarray,
    aoi: AOI,
    pixel_resolution_m: float,
    min_region_pixels: int = 4,
) -> list[BurnedRegion]:
    """Cluster a burned-area mask into connected regions.

    Args:
        result: output of detect_wildfire() (src/detection/wildfire.py).
        dnbr: the same dNBR array that produced `result` — needed to
            compute each region's mean dNBR for the confidence heuristic.
            Must be the same shape as result.burned_mask.
        aoi: the Area of Interest the raster covers (used to geolocate
            each region's centroid).
        pixel_resolution_m: ground resolution of one pixel, in meters
            (e.g. 10.0 for Sentinel-2's 10m bands) — needed to convert
            pixel counts into real hectares.
        min_region_pixels: minimum contiguous pixel count for a region
            to be reported. Smaller regions are treated as likely noise
            rather than a genuine detection. Defaults to 4.

    Returns:
        List of BurnedRegion, one per connected burned-pixel cluster
        meeting the minimum size, largest first.

    Raises:
        ValueError: if dnbr's shape doesn't match result.burned_mask's,
            or pixel_resolution_m/min_region_pixels are not positive.
    """
    if dnbr.shape != result.burned_mask.shape:
        raise ValueError(
            f"find_burned_regions: dnbr shape {dnbr.shape} does not match "
            f"result.burned_mask shape {result.burned_mask.shape}"
        )
    if pixel_resolution_m <= 0:
        raise ValueError(
            f"find_burned_regions: pixel_resolution_m must be positive, got {pixel_resolution_m}"
        )
    if min_region_pixels <= 0:
        raise ValueError(
            f"find_burned_regions: min_region_pixels must be positive, got {min_region_pixels}"
        )

    # 8-connectivity: pixels touching diagonally count as the same region,
    # since burned-area edges are rarely perfectly axis-aligned.
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = ndimage.label(result.burned_mask, structure=structure)

    regions: list[BurnedRegion] = []
    for label_id in range(1, num_features + 1):
        region_mask = labeled == label_id
        pixel_count = int(np.count_nonzero(region_mask))
        if pixel_count < min_region_pixels:
            continue

        rows, cols = np.nonzero(region_mask)
        centroid_row = float(rows.mean())
        centroid_col = float(cols.mean())
        lon, lat = _pixel_to_lonlat(centroid_row, centroid_col, aoi, result.burned_mask.shape)

        region_dnbr = dnbr[region_mask]
        valid_dnbr = region_dnbr[~np.isnan(region_dnbr)]
        mean_dnbr = float(valid_dnbr.mean()) if valid_dnbr.size > 0 else float("nan")

        region_severity_labels = result.severity_labels[region_mask]
        severity_counts: dict[str, int] = {}
        for label in region_severity_labels:
            severity_counts[label.value] = severity_counts.get(label.value, 0) + 1
        non_no_data = {k: v for k, v in severity_counts.items() if k != BurnSeverity.NO_DATA.value}
        dominant_value = max(non_no_data, key=non_no_data.get) if non_no_data else BurnSeverity.NO_DATA.value
        dominant_severity = BurnSeverity(dominant_value)

        area_hectares = pixel_count * (pixel_resolution_m**2) / 10_000.0

        regions.append(
            BurnedRegion(
                label=label_id,
                pixel_count=pixel_count,
                centroid_row=centroid_row,
                centroid_col=centroid_col,
                latitude=lat,
                longitude=lon,
                mean_dnbr=mean_dnbr,
                dominant_severity=dominant_severity,
                severity_counts=severity_counts,
                area_hectares=area_hectares,
            )
        )

    regions.sort(key=lambda r: r.pixel_count, reverse=True)
    return regions


def _region_confidence(region: BurnedRegion, result: WildfireDetectionResult, burn_threshold: float) -> ConfidenceScore:
    """Deterministic confidence heuristic — see module docstring for what
    this does and does not mean."""
    t = result.thresholds
    reference_span = t.moderate_high_max - burn_threshold
    if reference_span <= 0 or np.isnan(region.mean_dnbr):
        value = _MIN_REGION_CONFIDENCE
    else:
        fraction = (region.mean_dnbr - burn_threshold) / reference_span
        value = _MIN_REGION_CONFIDENCE + fraction * (_MAX_REGION_CONFIDENCE - _MIN_REGION_CONFIDENCE)
        value = max(_MIN_REGION_CONFIDENCE, min(_MAX_REGION_CONFIDENCE, value))

    return ConfidenceScore(
        value=round(value, 3),
        basis=(
            f"Region mean dNBR ({region.mean_dnbr:.3f}) relative to burn threshold "
            f"({burn_threshold}) and high-severity boundary ({t.moderate_high_max}) — "
            f"a detection-strength heuristic, not a statistical confidence measure."
        ),
        evidence_level=EvidenceLevel.DETECTED,
    )


def wildfire_result_to_events(
    result: WildfireDetectionResult,
    dnbr: np.ndarray,
    aoi: AOI,
    observation_time: datetime,
    source: str,
    pixel_resolution_m: float,
    burn_threshold: float = 0.10,
    min_region_pixels: int = 4,
) -> list[GeoWatchEvent]:
    """Convert a wildfire detection result into geolocated GeoWatchEvent objects.

    One event per connected burned-pixel region (see find_burned_regions()),
    not one event per pixel — see module docstring for why.

    Args:
        result: output of detect_wildfire().
        dnbr: the dNBR array that produced `result`.
        aoi: the Area of Interest the raster covers.
        observation_time: acquisition time of the imagery this detection
            is based on.
        source: human-readable description of the imagery source, e.g.
            "Sentinel-2 (scene S2B_35KMS_20250615_0_L2A)".
        pixel_resolution_m: ground resolution of one pixel, in meters.
        burn_threshold: the burn_threshold originally passed to
            detect_wildfire() when producing `result` — needed here for
            the confidence heuristic. Defaults to 0.10, matching
            detect_wildfire()'s own default.
        min_region_pixels: passed to find_burned_regions().

    Returns:
        List of GeoWatchEvent, one per burned region, each with
        EvidenceLevel.DETECTED, a deterministic event_id (so re-running
        detection on the same result produces the same IDs rather than
        duplicate events), and evidence describing the region's size,
        dominant severity, and detection method.
    """
    regions = find_burned_regions(result, dnbr, aoi, pixel_resolution_m, min_region_pixels)

    events: list[GeoWatchEvent] = []
    for region in regions:
        event_id = (
            f"wildfire-raster:{source}:{observation_time.isoformat()}:"
            f"{round(region.latitude, 5)}:{round(region.longitude, 5)}"
        )
        confidence = _region_confidence(region, result, burn_threshold)

        evidence = Evidence(
            description=(
                f"Connected burned-area region of {region.pixel_count} pixels "
                f"({region.area_hectares:.2f} ha), dominant severity "
                f"{region.dominant_severity.value}, detected via dNBR threshold "
                f"analysis."
            ),
            level=EvidenceLevel.DETECTED,
            source=source,
            observed_at=observation_time,
            metadata={
                "pixel_count": region.pixel_count,
                "area_hectares": round(region.area_hectares, 2),
                "mean_dnbr": round(region.mean_dnbr, 4) if not np.isnan(region.mean_dnbr) else None,
                "severity_counts": region.severity_counts,
                "burn_threshold": burn_threshold,
                "pixel_resolution_m": pixel_resolution_m,
            },
        )

        events.append(
            GeoWatchEvent(
                event_id=event_id,
                event_type=EventType.WILDFIRE,
                latitude=region.latitude,
                longitude=region.longitude,
                detected_at=observation_time,
                observation_time=observation_time,
                source=source,
                evidence_level=EvidenceLevel.DETECTED,
                confidence=confidence,
                status=EventStatus.NEW,
                severity=region.dominant_severity.value,
                evidence=[evidence],
                metadata={
                    "pixel_count": region.pixel_count,
                    "area_hectares": round(region.area_hectares, 2),
                    "aoi_label": aoi.label,
                },
            )
        )

    return events
