"""
NASA FIRMS (Fire Information for Resource Management System) provider.

Implements the Area Fire Detections API:
    https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{day_range}

Requires a free MAP_KEY (see https://firms.modaps.eosdis.nasa.gov/api/area/
-> "Get MAP_KEY"). Never hardcode a MAP_KEY in source or commit one to
git — this module reads it from the FIRMS_MAP_KEY environment variable
by default.

Rate limit: 5,000 transactions per 10-minute window per MAP_KEY. A
multi-day query counts as multiple transactions. See
check_transaction_status() to inspect current usage before a large query.

This module does NOT implement SatelliteDataProvider's download_asset()
or get_observation() — FIRMS is a tabular hotspot-detection API, not an
imagery archive, and has no per-observation ID lookup. Both raise
NotImplementedError with an explanation, which is the honest behavior
for a provider that genuinely does not support those operations, rather
than faking a response.
"""

from __future__ import annotations

import csv
import io
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import requests

from src.geospatial.aoi import AOI
from src.ingestion.base import SatelliteDataProvider
from src.types import ConfidenceScore, Evidence, EventType, EvidenceLevel, GeoWatchEvent

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov"

# Near-real-time hotspot detection sources. LANDSAT_NRT is US/Canada only.
# See https://firms.modaps.eosdis.nasa.gov/api/data_availability for the
# full, current list — this is the practical subset for near-real-time
# global wildfire monitoring.
VALID_SOURCES = frozenset(
    {
        "MODIS_NRT",
        "VIIRS_NOAA20_NRT",
        "VIIRS_NOAA21_NRT",
        "VIIRS_SNPP_NRT",
        "LANDSAT_NRT",
    }
)

DEFAULT_SOURCE = "VIIRS_NOAA20_NRT"
MAX_DAY_RANGE = 10  # FIRMS API hard limit


class FIRMSError(Exception):
    """Raised for FIRMS API failures GeoWatch cannot recover from automatically."""


class FIRMSProvider(SatelliteDataProvider):
    """SatelliteDataProvider implementation for NASA FIRMS active-fire detections."""

    name = "NASA_FIRMS"

    def __init__(self, map_key: Optional[str] = None, session: Optional[requests.Session] = None):
        """
        Args:
            map_key: FIRMS MAP_KEY. If not provided, read from the
                FIRMS_MAP_KEY environment variable.
            session: optional requests.Session, mainly for test injection.

        Raises:
            FIRMSError: if no map_key is provided and FIRMS_MAP_KEY is unset.
        """
        resolved_key = map_key or os.environ.get("FIRMS_MAP_KEY")
        if not resolved_key:
            raise FIRMSError(
                "No FIRMS MAP_KEY provided. Set the FIRMS_MAP_KEY environment "
                "variable, or pass map_key= explicitly. Get a free key at "
                "https://firms.modaps.eosdis.nasa.gov/api/area/"
            )
        self._map_key = resolved_key
        self._session = session or requests.Session()

    def check_transaction_status(self) -> dict[str, Any]:
        """Query current MAP_KEY transaction usage against the 10-minute rate limit.

        Returns:
            Dict with keys 'transaction_limit', 'current_transactions',
            'transaction_interval' as reported by the FIRMS API.

        Raises:
            FIRMSError: on network failure or an unparseable response
                (most commonly caused by an invalid MAP_KEY).
        """
        url = f"{FIRMS_BASE_URL}/mapserver/mapkey_status/"
        try:
            response = self._session.get(url, params={"MAP_KEY": self._map_key}, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise FIRMSError(f"FIRMS transaction status check failed: {exc}") from exc
        except ValueError as exc:  # response.json() failed to parse
            raise FIRMSError(
                f"FIRMS transaction status returned an unparseable response "
                f"(possible invalid MAP_KEY): {exc}"
            ) from exc

    def search_observations(
        self,
        bbox: tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: Optional[float] = None,
        sensor: str = DEFAULT_SOURCE,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Fetch active-fire detections for a bounding box and date range.

        Args:
            bbox: (west, south, east, north) in decimal degrees.
            start_date: earliest date of interest (day-level granularity;
                the FIRMS area API is day-resolution, not sub-day).
            end_date: latest date of interest. Must be >= start_date.
                GeoWatch does not claim this returns exhaustive historical
                coverage beyond MAX_DAY_RANGE days back from end_date —
                see the day_range note below.
            max_cloud_cover: accepted for interface consistency with other
                providers, but not applicable to FIRMS (a thermal-anomaly
                detection product, not an optical scene catalogue).
                Ignored if provided.
            sensor: one of VALID_SOURCES. Defaults to VIIRS_NOAA20_NRT.
            **kwargs: unused, accepted for interface consistency.

        Returns:
            List of raw detection dicts, one per hotspot pixel, as
            returned by the FIRMS CSV response (fields vary slightly by
            sensor — VIIRS includes bright_ti4/bright_ti5, MODIS includes
            brightness/bright_t31). Use rows_to_events() to convert these
            into GeoWatchEvent objects.

        Raises:
            ValueError: for invalid sensor, bbox, or date range.
            FIRMSError: on network failure, rate-limit rejection, or an
                unparseable/error response from FIRMS.
        """
        if sensor not in VALID_SOURCES:
            raise ValueError(
                f"search_observations: unknown sensor {sensor!r}. "
                f"Valid options: {sorted(VALID_SOURCES)}"
            )
        if end_date < start_date:
            raise ValueError(
                f"search_observations: end_date ({end_date}) is before start_date ({start_date})"
            )

        west, south, east, north = bbox
        area = AOI(label="query", west=west, south=south, east=east, north=north)  # validates bounds

        day_range = (end_date.date() - start_date.date()).days + 1
        if day_range > MAX_DAY_RANGE:
            raise ValueError(
                f"search_observations: requested range spans {day_range} days, "
                f"FIRMS area API supports at most {MAX_DAY_RANGE}. Split into "
                f"multiple queries."
            )

        # FIRMS's [DATE] parameter is the END of the day_range window
        # (it returns day_range days ending on, and including, [DATE]).
        query_date = end_date.date().isoformat()

        url = (
            f"{FIRMS_BASE_URL}/api/area/csv/{self._map_key}/{sensor}/"
            f"{area.as_firms_area_string()}/{day_range}/{query_date}"
        )

        try:
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FIRMSError(f"FIRMS area query failed: {exc}") from exc

        text = response.text.strip()

        # FIRMS returns HTTP 200 even for some error conditions (e.g. an
        # invalid MAP_KEY), just with a short plain-text error body instead
        # of CSV. Detect that rather than trying to parse it as data.
        if not text or "," not in text.splitlines()[0]:
            raise FIRMSError(
                f"FIRMS area query returned an unexpected (non-CSV) response, "
                f"likely an invalid MAP_KEY or rate limit rejection: {text[:200]!r}"
            )

        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        return rows

    def rows_to_events(self, rows: list[dict[str, Any]], sensor: str = DEFAULT_SOURCE) -> list[GeoWatchEvent]:
        """Convert raw FIRMS CSV rows into GeoWatchEvent objects.

        Each row becomes one WILDFIRE event with EvidenceLevel.OBSERVED —
        this is a raw satellite-instrument detection (NASA's own active-fire
        algorithm already applied), not a GeoWatch-computed result, which is
        why it is OBSERVED rather than DETECTED. See src/types.py.

        Args:
            rows: output of search_observations().
            sensor: the sensor these rows came from, used to interpret the
                confidence field correctly (VIIRS uses l/n/h categorical
                confidence; MODIS uses a 0-100 numeric percentage).

        Returns:
            List of GeoWatchEvent, one per input row. Rows that fail to
            parse (missing/malformed required fields) are skipped rather
            than raising, since a single malformed row should not discard
            an entire batch of otherwise-valid detections; skipped rows
            are not silently lost — count of skipped rows is not currently
            tracked and would be a reasonable future improvement if this
            becomes a problem in practice.
        """
        events: list[GeoWatchEvent] = []
        for row in rows:
            try:
                events.append(self._row_to_event(row, sensor))
            except (KeyError, ValueError):
                continue
        return events

    def _row_to_event(self, row: dict[str, Any], sensor: str) -> GeoWatchEvent:
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        acq_date = row["acq_date"]
        acq_time = str(row["acq_time"]).zfill(4)  # FIRMS gives HHMM, sometimes without leading zeros
        satellite = row.get("satellite", "unknown")
        instrument = row.get("instrument", "unknown")

        observation_time = datetime.strptime(f"{acq_date} {acq_time}", "%Y-%m-%d %H%M").replace(
            tzinfo=timezone.utc
        )

        confidence = self._parse_confidence(row.get("confidence", ""), sensor)

        frp = row.get("frp")  # Fire Radiative Power (MW), where available
        metadata: dict[str, Any] = {
            "satellite": satellite,
            "instrument": instrument,
            "daynight": row.get("daynight"),
        }
        if frp not in (None, ""):
            try:
                metadata["fire_radiative_power_mw"] = float(frp)
            except ValueError:
                pass

        event_id = f"firms:{sensor}:{acq_date}:{acq_time}:{latitude:.5f}:{longitude:.5f}"

        return GeoWatchEvent(
            event_id=event_id,
            event_type=EventType.WILDFIRE,
            latitude=latitude,
            longitude=longitude,
            detected_at=datetime.now(timezone.utc),
            observation_time=observation_time,
            source=f"NASA FIRMS ({sensor})",
            evidence_level=EvidenceLevel.OBSERVED,
            confidence=confidence,
            evidence=[
                Evidence(
                    description=f"Active-fire hotspot detected by {satellite} {instrument}",
                    level=EvidenceLevel.OBSERVED,
                    source=f"NASA FIRMS ({sensor})",
                    observed_at=observation_time,
                    metadata=metadata,
                )
            ],
            metadata=metadata,
        )

    @staticmethod
    def _parse_confidence(raw: str, sensor: str) -> ConfidenceScore:
        """Map FIRMS's confidence field to a GeoWatch ConfidenceScore.

        VIIRS sensors report categorical confidence ('l'ow / 'n'ominal /
        'h'igh); MODIS reports a 0-100 numeric percentage. The categorical
        -> numeric mapping used here (low=0.3, nominal=0.6, high=0.9) is
        GeoWatch's own interpretation for display/filtering purposes, not
        an official NASA conversion — documented here so it's auditable.
        """
        raw = str(raw).strip().lower()

        if sensor.startswith("VIIRS"):
            mapping = {"l": 0.3, "low": 0.3, "n": 0.6, "nominal": 0.6, "h": 0.9, "high": 0.9}
            value = mapping.get(raw, 0.5)  # unknown categorical -> neutral default
            basis = f"FIRMS VIIRS categorical confidence: {raw!r}"
        else:
            try:
                value = max(0.0, min(1.0, float(raw) / 100.0))
                basis = f"FIRMS numeric confidence: {raw}%"
            except ValueError:
                value = 0.5
                basis = f"FIRMS confidence unparseable ({raw!r}), defaulted"

        return ConfidenceScore(value=value, basis=basis, evidence_level=EvidenceLevel.OBSERVED)

    # --- SatelliteDataProvider methods FIRMS genuinely does not support ---

    def get_observation(self, observation_id: str) -> dict[str, Any]:
        raise NotImplementedError(
            "FIRMS does not support single-observation lookup by ID. "
            "Use search_observations() to query by area and date range."
        )

    def download_asset(self, observation_id: str, asset_key: str, destination: str) -> str:
        raise NotImplementedError(
            "FIRMS is a tabular active-fire detection API, not an imagery "
            "archive — there is no downloadable asset per detection."
        )

    def get_metadata(self, observation_id: str) -> dict[str, Any]:
        raise NotImplementedError(
            "FIRMS does not support per-observation metadata lookup by ID. "
            "Metadata for each detection is included directly in "
            "search_observations() results."
        )
