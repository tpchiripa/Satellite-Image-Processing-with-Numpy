"""
Tests for src/ingestion/firms.py

The FIRMS API is not reachable from this test environment's network, so
these tests mock requests.Session.get() with realistic response shapes
taken directly from NASA's own FIRMS API tutorial documentation
(https://firms.modaps.eosdis.nasa.gov/content/academy/data_api/firms_api_use.html).
A live end-to-end check against the real API (with a real MAP_KEY) is a
separate, manual verification step — not something a unit test suite
should depend on to pass.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.ingestion.firms import FIRMSError, FIRMSProvider
from src.types import EventType, EvidenceLevel

# Realistic VIIRS CSV sample, shape taken from NASA's own FIRMS API tutorial.
VIIRS_CSV_SAMPLE = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight\n"
    "-16.28359,29.40531,295.78,0.50,0.66,2025-06-06,0100,N20,VIIRS,n,2.0NRT,284.11,1.17,N\n"
    "-14.98900,28.36286,341.04,0.41,0.60,2025-06-06,0100,N20,VIIRS,l,2.0NRT,279.77,4.59,N\n"
    "28.39525,-16.54512,348.49,0.49,0.40,2025-06-06,1427,N20,VIIRS,h,2.0NRT,307.14,10.68,D\n"
)

MODIS_CSV_SAMPLE = (
    "country_id,latitude,longitude,brightness,scan,track,acq_date,acq_time,"
    "satellite,instrument,confidence,version,bright_t31,frp,daynight\n"
    "PER,-6.99466,-76.58813,311.65,1.03,1.02,2025-06-03,0236,Terra,MODIS,83,"
    "6.1NRT,292.65,10.58,N\n"
)


def _mock_session(text: str, status_code: int = 200) -> MagicMock:
    session = MagicMock()
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests

        response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    session.get.return_value = response
    return session


class TestFIRMSProviderInit:
    def test_requires_map_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
        with pytest.raises(FIRMSError):
            FIRMSProvider()

    def test_accepts_explicit_map_key(self) -> None:
        provider = FIRMSProvider(map_key="test-key-123")
        assert provider._map_key == "test-key-123"

    def test_reads_map_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRMS_MAP_KEY", "env-key-456")
        provider = FIRMSProvider()
        assert provider._map_key == "env-key-456"


class TestSearchObservations:
    def test_valid_query_returns_parsed_rows(self) -> None:
        session = _mock_session(VIIRS_CSV_SAMPLE)
        provider = FIRMSProvider(map_key="test-key", session=session)

        rows = provider.search_observations(
            bbox=(-20.0, -20.0, 40.0, 40.0),
            start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
        )
        assert len(rows) == 3
        assert rows[0]["latitude"] == "-16.28359"

    def test_unknown_sensor_rejected(self) -> None:
        session = _mock_session(VIIRS_CSV_SAMPLE)
        provider = FIRMSProvider(map_key="test-key", session=session)
        with pytest.raises(ValueError):
            provider.search_observations(
                bbox=(-20.0, -20.0, 40.0, 40.0),
                start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
                end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
                sensor="NOT_A_REAL_SENSOR",
            )

    def test_end_before_start_rejected(self) -> None:
        session = _mock_session(VIIRS_CSV_SAMPLE)
        provider = FIRMSProvider(map_key="test-key", session=session)
        with pytest.raises(ValueError):
            provider.search_observations(
                bbox=(-20.0, -20.0, 40.0, 40.0),
                start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
                end_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            )

    def test_range_exceeding_max_day_range_rejected(self) -> None:
        session = _mock_session(VIIRS_CSV_SAMPLE)
        provider = FIRMSProvider(map_key="test-key", session=session)
        with pytest.raises(ValueError):
            provider.search_observations(
                bbox=(-20.0, -20.0, 40.0, 40.0),
                start_date=datetime(2025, 5, 1, tzinfo=timezone.utc),
                end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),  # 37 days
            )

    def test_invalid_bbox_rejected(self) -> None:
        session = _mock_session(VIIRS_CSV_SAMPLE)
        provider = FIRMSProvider(map_key="test-key", session=session)
        with pytest.raises(ValueError):
            provider.search_observations(
                bbox=(40.0, -20.0, -20.0, 40.0),  # west > east
                start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
                end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
            )

    def test_network_failure_raises_firms_error(self) -> None:
        import requests

        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("network unreachable")
        provider = FIRMSProvider(map_key="test-key", session=session)
        with pytest.raises(FIRMSError):
            provider.search_observations(
                bbox=(-20.0, -20.0, 40.0, 40.0),
                start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
                end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
            )

    def test_invalid_map_key_error_response_raises_firms_error(self) -> None:
        # FIRMS returns HTTP 200 with a plain-text error body for a bad key,
        # not valid CSV — this must be detected, not silently parsed as
        # zero rows.
        session = _mock_session("Invalid MAP_KEY.")
        provider = FIRMSProvider(map_key="bad-key", session=session)
        with pytest.raises(FIRMSError):
            provider.search_observations(
                bbox=(-20.0, -20.0, 40.0, 40.0),
                start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
                end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
            )

    def test_http_error_status_raises_firms_error(self) -> None:
        session = _mock_session("", status_code=503)
        provider = FIRMSProvider(map_key="test-key", session=session)
        with pytest.raises(FIRMSError):
            provider.search_observations(
                bbox=(-20.0, -20.0, 40.0, 40.0),
                start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
                end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
            )


class TestRowsToEvents:
    def test_converts_rows_to_wildfire_events(self) -> None:
        session = _mock_session(VIIRS_CSV_SAMPLE)
        provider = FIRMSProvider(map_key="test-key", session=session)
        rows = provider.search_observations(
            bbox=(-20.0, -20.0, 40.0, 40.0),
            start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
        )
        events = provider.rows_to_events(rows, sensor="VIIRS_NOAA20_NRT")

        assert len(events) == 3
        assert all(e.event_type == EventType.WILDFIRE for e in events)
        assert all(e.evidence_level == EvidenceLevel.OBSERVED for e in events)

    def test_event_coordinates_match_source_row(self) -> None:
        session = _mock_session(VIIRS_CSV_SAMPLE)
        provider = FIRMSProvider(map_key="test-key", session=session)
        rows = provider.search_observations(
            bbox=(-20.0, -20.0, 40.0, 40.0),
            start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
        )
        events = provider.rows_to_events(rows)
        assert events[0].latitude == pytest.approx(-16.28359)
        assert events[0].longitude == pytest.approx(29.40531)

    def test_viirs_categorical_confidence_mapped(self) -> None:
        session = _mock_session(VIIRS_CSV_SAMPLE)
        provider = FIRMSProvider(map_key="test-key", session=session)
        rows = provider.search_observations(
            bbox=(-20.0, -20.0, 40.0, 40.0),
            start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
        )
        events = provider.rows_to_events(rows, sensor="VIIRS_NOAA20_NRT")
        # Row 0 has confidence 'n' (nominal) -> 0.6, row 2 has 'h' (high) -> 0.9
        assert events[0].confidence.value == pytest.approx(0.6)
        assert events[2].confidence.value == pytest.approx(0.9)

    def test_modis_numeric_confidence_mapped(self) -> None:
        session = _mock_session(MODIS_CSV_SAMPLE)
        provider = FIRMSProvider(map_key="test-key", session=session)
        rows = provider.search_observations(
            bbox=(-90.0, -20.0, -60.0, 10.0),
            start_date=datetime(2025, 6, 3, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 3, tzinfo=timezone.utc),
            sensor="MODIS_NRT",
        )
        events = provider.rows_to_events(rows, sensor="MODIS_NRT")
        # confidence=83 -> 0.83
        assert events[0].confidence.value == pytest.approx(0.83)

    def test_frp_captured_in_metadata_when_present(self) -> None:
        session = _mock_session(VIIRS_CSV_SAMPLE)
        provider = FIRMSProvider(map_key="test-key", session=session)
        rows = provider.search_observations(
            bbox=(-20.0, -20.0, 40.0, 40.0),
            start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
        )
        events = provider.rows_to_events(rows)
        assert events[0].metadata["fire_radiative_power_mw"] == pytest.approx(1.17)

    def test_malformed_row_is_skipped_not_raised(self) -> None:
        provider = FIRMSProvider(map_key="test-key", session=_mock_session(""))
        bad_rows = [{"latitude": "not-a-number", "longitude": "0", "acq_date": "2025-06-06", "acq_time": "0100"}]
        events = provider.rows_to_events(bad_rows)
        assert events == []

    def test_event_ids_are_unique_across_rows(self) -> None:
        session = _mock_session(VIIRS_CSV_SAMPLE)
        provider = FIRMSProvider(map_key="test-key", session=session)
        rows = provider.search_observations(
            bbox=(-20.0, -20.0, 40.0, 40.0),
            start_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 6, tzinfo=timezone.utc),
        )
        events = provider.rows_to_events(rows)
        assert len({e.event_id for e in events}) == len(events)


class TestUnsupportedMethods:
    def test_get_observation_not_implemented(self) -> None:
        provider = FIRMSProvider(map_key="test-key")
        with pytest.raises(NotImplementedError):
            provider.get_observation("anything")

    def test_download_asset_not_implemented(self) -> None:
        provider = FIRMSProvider(map_key="test-key")
        with pytest.raises(NotImplementedError):
            provider.download_asset("anything", "band", "/tmp/out.tif")

    def test_get_metadata_not_implemented(self) -> None:
        provider = FIRMSProvider(map_key="test-key")
        with pytest.raises(NotImplementedError):
            provider.get_metadata("anything")


class TestCheckTransactionStatus:
    def test_parses_status_response(self) -> None:
        session = MagicMock()
        response = MagicMock()
        response.json.return_value = {
            "transaction_limit": 5000,
            "current_transactions": 46,
            "transaction_interval": "10 minutes",
        }
        response.raise_for_status = MagicMock()
        session.get.return_value = response

        provider = FIRMSProvider(map_key="test-key", session=session)
        status = provider.check_transaction_status()
        assert status["transaction_limit"] == 5000

    def test_network_failure_raises_firms_error(self) -> None:
        import requests

        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("unreachable")
        provider = FIRMSProvider(map_key="test-key", session=session)
        with pytest.raises(FIRMSError):
            provider.check_transaction_status()
