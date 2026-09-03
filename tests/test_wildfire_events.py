"""Tests for src/detection/wildfire_events.py"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.detection.wildfire import SeverityThresholds, detect_wildfire
from src.detection.wildfire_events import find_burned_regions, wildfire_result_to_events
from src.geospatial.aoi import AOI
from src.types import EventType, EvidenceLevel

OBS_TIME = datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc)


def _square_aoi() -> AOI:
    # A simple 1-degree square, easy to reason about pixel->lonlat mapping.
    return AOI(label="test AOI", west=20.0, south=-20.0, east=21.0, north=-19.0)


class TestFindBurnedRegions:
    def test_no_burned_pixels_gives_no_regions(self) -> None:
        dnbr = np.zeros((20, 20))
        result = detect_wildfire(dnbr)
        regions = find_burned_regions(result, dnbr, _square_aoi(), pixel_resolution_m=10.0)
        assert regions == []

    def test_single_contiguous_region_detected(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5:10, 5:10] = 0.6  # a 5x5 burned block, well past threshold
        result = detect_wildfire(dnbr)
        regions = find_burned_regions(result, dnbr, _square_aoi(), pixel_resolution_m=10.0)
        assert len(regions) == 1
        assert regions[0].pixel_count == 25

    def test_two_separate_regions_detected_separately(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[0:3, 0:3] = 0.6      # region A, far corner
        dnbr[15:18, 15:18] = 0.6  # region B, opposite corner
        result = detect_wildfire(dnbr)
        regions = find_burned_regions(result, dnbr, _square_aoi(), pixel_resolution_m=10.0)
        assert len(regions) == 2

    def test_diagonally_touching_pixels_are_one_region(self) -> None:
        dnbr = np.zeros((10, 10))
        dnbr[2, 2] = 0.6
        dnbr[3, 3] = 0.6  # diagonal neighbor only
        result = detect_wildfire(dnbr)
        regions = find_burned_regions(result, dnbr, _square_aoi(), pixel_resolution_m=10.0, min_region_pixels=1)
        assert len(regions) == 1
        assert regions[0].pixel_count == 2

    def test_regions_below_min_pixels_excluded(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5, 5] = 0.6  # single pixel
        dnbr[10:15, 10:15] = 0.6  # 25-pixel region
        result = detect_wildfire(dnbr)
        regions = find_burned_regions(result, dnbr, _square_aoi(), pixel_resolution_m=10.0, min_region_pixels=4)
        assert len(regions) == 1
        assert regions[0].pixel_count == 25

    def test_regions_sorted_largest_first(self) -> None:
        dnbr = np.zeros((30, 30))
        dnbr[0:2, 0:2] = 0.6    # small: 4 px
        dnbr[10:18, 10:18] = 0.6  # large: 64 px
        result = detect_wildfire(dnbr)
        regions = find_burned_regions(result, dnbr, _square_aoi(), pixel_resolution_m=10.0)
        assert regions[0].pixel_count >= regions[1].pixel_count

    def test_area_hectares_computed_correctly(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5:10, 5:10] = 0.6  # 25 pixels
        result = detect_wildfire(dnbr)
        regions = find_burned_regions(result, dnbr, _square_aoi(), pixel_resolution_m=10.0)
        # 25 pixels * 100 m^2 / 10000 = 0.25 hectares
        assert regions[0].area_hectares == pytest.approx(0.25)

    def test_centroid_within_aoi_bounds(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5:10, 5:10] = 0.6
        result = detect_wildfire(dnbr)
        aoi = _square_aoi()
        regions = find_burned_regions(result, dnbr, aoi, pixel_resolution_m=10.0)
        assert aoi.west <= regions[0].longitude <= aoi.east
        assert aoi.south <= regions[0].latitude <= aoi.north

    def test_dominant_severity_reflects_majority(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5:10, 5:10] = 0.9  # high severity block
        result = detect_wildfire(dnbr)
        regions = find_burned_regions(result, dnbr, _square_aoi(), pixel_resolution_m=10.0)
        assert regions[0].dominant_severity.value == "high"

    def test_mismatched_dnbr_shape_raises(self) -> None:
        dnbr = np.zeros((20, 20))
        result = detect_wildfire(dnbr)
        wrong_shape_dnbr = np.zeros((10, 10))
        with pytest.raises(ValueError):
            find_burned_regions(result, wrong_shape_dnbr, _square_aoi(), pixel_resolution_m=10.0)

    def test_nonpositive_resolution_raises(self) -> None:
        dnbr = np.zeros((10, 10))
        result = detect_wildfire(dnbr)
        with pytest.raises(ValueError):
            find_burned_regions(result, dnbr, _square_aoi(), pixel_resolution_m=0.0)

    def test_nonpositive_min_region_pixels_raises(self) -> None:
        dnbr = np.zeros((10, 10))
        result = detect_wildfire(dnbr)
        with pytest.raises(ValueError):
            find_burned_regions(result, dnbr, _square_aoi(), pixel_resolution_m=10.0, min_region_pixels=0)


class TestWildfireResultToEvents:
    def test_no_burned_regions_gives_no_events(self) -> None:
        dnbr = np.zeros((20, 20))
        result = detect_wildfire(dnbr)
        events = wildfire_result_to_events(
            result, dnbr, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        assert events == []

    def test_one_event_per_region_not_per_pixel(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5:10, 5:10] = 0.6  # 25 burned pixels, one contiguous region
        result = detect_wildfire(dnbr)
        events = wildfire_result_to_events(
            result, dnbr, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        assert len(events) == 1  # not 25

    def test_event_has_correct_type_and_evidence_level(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5:10, 5:10] = 0.6
        result = detect_wildfire(dnbr)
        events = wildfire_result_to_events(
            result, dnbr, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        assert events[0].event_type == EventType.WILDFIRE
        assert events[0].evidence_level == EvidenceLevel.DETECTED

    def test_event_confidence_within_bounds(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5:10, 5:10] = 0.6
        result = detect_wildfire(dnbr)
        events = wildfire_result_to_events(
            result, dnbr, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        assert 0.3 <= events[0].confidence.value <= 0.95

    def test_higher_dnbr_region_gets_higher_confidence(self) -> None:
        dnbr_low = np.zeros((20, 20))
        dnbr_low[5:10, 5:10] = 0.15  # just past default threshold (0.10)
        dnbr_high = np.zeros((20, 20))
        dnbr_high[5:10, 5:10] = 0.9  # deep into high severity

        result_low = detect_wildfire(dnbr_low)
        result_high = detect_wildfire(dnbr_high)

        events_low = wildfire_result_to_events(
            result_low, dnbr_low, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        events_high = wildfire_result_to_events(
            result_high, dnbr_high, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        assert events_high[0].confidence.value > events_low[0].confidence.value

    def test_event_id_deterministic_across_reruns(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5:10, 5:10] = 0.6
        result = detect_wildfire(dnbr)
        events_1 = wildfire_result_to_events(
            result, dnbr, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        events_2 = wildfire_result_to_events(
            result, dnbr, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        assert events_1[0].event_id == events_2[0].event_id

    def test_severity_field_set_on_event(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5:10, 5:10] = 0.9
        result = detect_wildfire(dnbr)
        events = wildfire_result_to_events(
            result, dnbr, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        assert events[0].severity == "high"

    def test_metadata_includes_area_and_pixel_count(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5:10, 5:10] = 0.6
        result = detect_wildfire(dnbr)
        events = wildfire_result_to_events(
            result, dnbr, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        assert events[0].metadata["pixel_count"] == 25
        assert events[0].metadata["area_hectares"] == pytest.approx(0.25)

    def test_evidence_entry_included(self) -> None:
        dnbr = np.zeros((20, 20))
        dnbr[5:10, 5:10] = 0.6
        result = detect_wildfire(dnbr)
        events = wildfire_result_to_events(
            result, dnbr, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        assert len(events[0].evidence) == 1
        assert events[0].evidence[0].level == EvidenceLevel.DETECTED

    def test_multiple_regions_produce_multiple_events(self) -> None:
        dnbr = np.zeros((30, 30))
        dnbr[0:5, 0:5] = 0.6
        dnbr[20:25, 20:25] = 0.9
        result = detect_wildfire(dnbr)
        events = wildfire_result_to_events(
            result, dnbr, _square_aoi(), OBS_TIME, "test-source", pixel_resolution_m=10.0
        )
        assert len(events) == 2
        assert {e.event_id for e in events} == {e.event_id for e in events}  # unique, no crash
        assert len({e.event_id for e in events}) == 2  # genuinely unique IDs

    def test_end_to_end_from_synthetic_bands(self) -> None:
        # Full realistic pipeline: bands -> NBR -> dNBR -> detection -> events.
        from src.remote_sensing.nbr import compute_dnbr, compute_nbr

        size = 20
        nir_pre = np.full((size, size), 0.5)
        swir_pre = np.full((size, size), 0.15)
        nir_post = nir_pre.copy()
        swir_post = swir_pre.copy()
        nir_post[5:10, 5:10] = 0.1
        swir_post[5:10, 5:10] = 0.3

        nbr_pre = compute_nbr(nir_pre, swir_pre)
        nbr_post = compute_nbr(nir_post, swir_post)
        dnbr = compute_dnbr(nbr_pre, nbr_post)

        result = detect_wildfire(dnbr)
        events = wildfire_result_to_events(
            result, dnbr, _square_aoi(), OBS_TIME, "Sentinel-2 (test scene)", pixel_resolution_m=10.0
        )
        assert len(events) == 1
        assert events[0].metadata["pixel_count"] == 25
