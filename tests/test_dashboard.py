"""
Tests for app/dashboard.py, using Streamlit's AppTest framework.

AppTest actually executes the dashboard script and its Streamlit calls
end-to-end (script run, widget interaction, session state) rather than
just checking it imports cleanly — this is the closest thing to a real
browser test without one.

Database-backed tests are skipped if no reachable PostGIS instance is
configured (same convention as tests/test_postgres_store.py). The
in-memory fallback path tests always run regardless.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from streamlit.testing.v1 import AppTest

DEFAULT_TEST_DB_URL = "postgresql://geowatch:geowatch@localhost:5439/geowatch"
TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DB_URL)
DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "app" / "dashboard.py"


def _database_reachable() -> bool:
    try:
        engine = create_engine(TEST_DB_URL)
        with engine.connect():
            return True
    except OperationalError:
        return False


requires_db = pytest.mark.skipif(
    not _database_reachable(),
    reason=f"No PostgreSQL/PostGIS reachable at {TEST_DB_URL} — start it with `docker compose up -d db`",
)

SAMPLE_VIIRS_ROW = {
    "latitude": "-16.28359",
    "longitude": "29.40531",
    "acq_date": "2025-06-06",
    "acq_time": "0100",
    "satellite": "N20",
    "instrument": "VIIRS",
    "confidence": "h",
    "frp": "5.0",
    "daynight": "D",
}


@pytest.fixture(autouse=True)
def _clear_caches_and_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure each test gets a fresh get_event_store()/get_firms_provider()
    resolution rather than a cached instance leaking in from another test,
    and a clean environment so tests are independent of each other."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    yield


class TestDashboardWithoutDatabase:
    """The fallback path: no DATABASE_URL at all, no FIRMS_MAP_KEY."""

    def test_runs_without_exceptions(self) -> None:
        at = AppTest.from_file(str(DASHBOARD_PATH))
        at.run(timeout=30)
        assert not at.exception

    def test_falls_back_to_in_memory_store(self) -> None:
        at = AppTest.from_file(str(DASHBOARD_PATH))
        at.run(timeout=30)
        backend_metric = next(m for m in at.metric if m.label == "Storage backend")
        assert "In-memory" in backend_metric.value

    def test_has_four_tabs(self) -> None:
        at = AppTest.from_file(str(DASHBOARD_PATH))
        at.run(timeout=30)
        assert len(at.tabs) == 4

    def test_empty_state_messaging_shown(self) -> None:
        at = AppTest.from_file(str(DASHBOARD_PATH))
        at.run(timeout=30)
        all_messages = [w.value for w in at.warning] + [i.value for i in at.info]
        assert any("No events stored yet" in msg for msg in all_messages)

    def test_no_firms_key_shows_instructions_not_a_crash(self) -> None:
        at = AppTest.from_file(str(DASHBOARD_PATH))
        at.run(timeout=30)
        assert not at.exception
        all_messages = [i.value for i in at.info]
        assert any("FIRMS_MAP_KEY" in msg for msg in all_messages)


@requires_db
class TestDashboardWithDatabase:
    """Real PostGIS-backed behavior."""

    @pytest.fixture(autouse=True)
    def _clean_table(self):
        from src.monitoring.postgres_store import Base

        engine = create_engine(TEST_DB_URL)
        Base.metadata.create_all(engine)
        yield
        with engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM geowatch_events;")

    def test_connects_and_reports_postgres_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)
        at = AppTest.from_file(str(DASHBOARD_PATH))
        at.run(timeout=30)
        assert not at.exception
        backend_metric = next(m for m in at.metric if m.label == "Storage backend")
        assert "PostgreSQL" in backend_metric.value

    def test_fetch_button_stores_real_event_in_postgis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)
        monkeypatch.setenv("FIRMS_MAP_KEY", "fake-test-key")

        with patch(
            "src.ingestion.firms.FIRMSProvider.search_observations",
            return_value=[SAMPLE_VIIRS_ROW],
        ):
            at = AppTest.from_file(str(DASHBOARD_PATH))
            at.run(timeout=30)

            fetch_buttons = [b for b in at.button if "Fetch latest fire data" in b.label]
            assert len(fetch_buttons) == 1
            fetch_buttons[0].click().run(timeout=30)

        assert not at.exception
        total_metric = next(m for m in at.metric if m.label == "Total events stored")
        assert total_metric.value == "1"

    def test_double_fetch_does_not_duplicate_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Confirms the dashboard relies on PostgresEventStore's duplicate-id
        # rejection rather than re-adding the same detection on every click.
        monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)
        monkeypatch.setenv("FIRMS_MAP_KEY", "fake-test-key")

        with patch(
            "src.ingestion.firms.FIRMSProvider.search_observations",
            return_value=[SAMPLE_VIIRS_ROW],
        ):
            at = AppTest.from_file(str(DASHBOARD_PATH))
            at.run(timeout=30)
            fetch_button = next(b for b in at.button if "Fetch latest fire data" in b.label)
            fetch_button.click().run(timeout=30)

            fetch_button = next(b for b in at.button if "Fetch latest fire data" in b.label)
            fetch_button.click().run(timeout=30)

        assert not at.exception
        total_metric = next(m for m in at.metric if m.label == "Total events stored")
        assert total_metric.value == "1"

    def test_time_series_chart_appears_with_multi_day_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)
        monkeypatch.setenv("FIRMS_MAP_KEY", "fake-test-key")

        multi_day_rows = [
            {**SAMPLE_VIIRS_ROW, "latitude": "-16.10", "acq_date": "2025-06-01"},
            {**SAMPLE_VIIRS_ROW, "latitude": "-16.20", "acq_date": "2025-06-03"},
            {**SAMPLE_VIIRS_ROW, "latitude": "-16.30", "acq_date": "2025-06-05"},
        ]

        with patch(
            "src.ingestion.firms.FIRMSProvider.search_observations",
            return_value=multi_day_rows,
        ):
            at = AppTest.from_file(str(DASHBOARD_PATH))
            at.run(timeout=30)
            fetch_button = next(b for b in at.button if "Fetch latest fire data" in b.label)
            fetch_button.click().run(timeout=30)

        assert not at.exception
        trend_captions = [c.value for c in at.caption if "Trend:" in c.value]
        assert len(trend_captions) == 1
        assert "day(s) of data" in trend_captions[0]

    def test_single_day_data_shows_no_chart_message_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)
        monkeypatch.setenv("FIRMS_MAP_KEY", "fake-test-key")

        with patch(
            "src.ingestion.firms.FIRMSProvider.search_observations",
            return_value=[SAMPLE_VIIRS_ROW],
        ):
            at = AppTest.from_file(str(DASHBOARD_PATH))
            at.run(timeout=30)
            fetch_button = next(b for b in at.button if "Fetch latest fire data" in b.label)
            fetch_button.click().run(timeout=30)

        assert not at.exception
        captions = [c.value for c in at.caption]
        assert any("nothing to chart yet" in c for c in captions)

    def test_sentinel2_wildfire_detection_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)

        import numpy as np

        PRE_SCENE = {
            "id": "S2_TEST_PRE",
            "properties": {"datetime": "2025-01-01T10:00:00Z", "eo:cloud_cover": 5.0},
            "assets": {},
        }
        POST_SCENE = {
            "id": "S2_TEST_POST",
            "properties": {"datetime": "2025-03-01T10:00:00Z", "eo:cloud_cover": 8.0},
            "assets": {},
        }

        healthy_bands = {"nir": np.full((10, 10), 3000, dtype=np.uint16), "swir16": np.full((10, 10), 1500, dtype=np.uint16)}
        burned_bands = {"nir": np.full((10, 10), 1000, dtype=np.uint16), "swir16": np.full((10, 10), 3500, dtype=np.uint16)}

        with (
            patch(
                "src.ingestion.sentinel2.Sentinel2Provider.search_observations",
                side_effect=[[PRE_SCENE], [POST_SCENE]],
            ),
            patch(
                "src.ingestion.sentinel2.Sentinel2Provider.read_scene_bands",
                side_effect=[healthy_bands, burned_bands],
            ),
        ):
            at = AppTest.from_file(str(DASHBOARD_PATH))
            at.run(timeout=30)
            run_button = next(b for b in at.button if "Run Sentinel-2 wildfire detection" in b.label)
            run_button.click().run(timeout=30)

        assert not at.exception
        # The success message is transient -- st.rerun() at the end of a
        # successful run wipes it, same reason test_fetch_button_stores_
        # real_event_in_postgis checks persisted state rather than the
        # NASA FIRMS fetch's own success message. Check what actually
        # persists: the stored-detections caption on this tab.
        captions = [c.value for c in at.caption]
        assert any("stored Sentinel-2-derived detection" in c for c in captions)

    def test_sentinel2_no_scenes_found_shows_warning_not_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)

        with patch("src.ingestion.sentinel2.Sentinel2Provider.search_observations", return_value=[]):
            at = AppTest.from_file(str(DASHBOARD_PATH))
            at.run(timeout=30)
            run_button = next(b for b in at.button if "Run Sentinel-2 wildfire detection" in b.label)
            run_button.click().run(timeout=30)

        assert not at.exception
        warnings = [w.value for w in at.warning]
        assert any("No pre-fire scene found" in w for w in warnings)

    def test_sentinel2_error_shown_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)

        from src.ingestion.sentinel2 import Sentinel2Error

        with patch(
            "src.ingestion.sentinel2.Sentinel2Provider.search_observations",
            side_effect=Sentinel2Error("simulated network failure"),
        ):
            at = AppTest.from_file(str(DASHBOARD_PATH))
            at.run(timeout=30)
            run_button = next(b for b in at.button if "Run Sentinel-2 wildfire detection" in b.label)
            run_button.click().run(timeout=30)

        assert not at.exception
        errors = [e.value for e in at.error]
        assert any("Sentinel-2 wildfire detection failed" in e for e in errors)

    def test_no_sentinel2_events_shows_empty_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", TEST_DB_URL)

        at = AppTest.from_file(str(DASHBOARD_PATH))
        at.run(timeout=30)

        assert not at.exception
        infos = [i.value for i in at.info]
        assert any("No Sentinel-2-derived wildfire detections stored yet" in i for i in infos)
