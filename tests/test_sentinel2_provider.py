"""
Tests for src/ingestion/sentinel2.py

Earth Search is not reachable from this test environment's network, so
search_observations()/get_observation()/download_asset() are tested
with mocked HTTP responses shaped exactly like real Earth Search v1
STAC items (asset keys confirmed against Earth Search's actual
sentinel-2-l2a collection: 'red', 'nir', 'swir16', etc. — not raw band
codes like 'B04'). read_scene_bands() is tested against a real local
GeoTIFF (see tests/test_raster_imagery.py for the underlying raster
read logic, tested independently there).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.geospatial.aoi import AOI
from src.ingestion.sentinel2 import BAND_ASSET_KEYS, Sentinel2Error, Sentinel2Provider

# Realistic STAC item shape, asset keys taken directly from Earth Search
# v1's actual sentinel-2-l2a collection structure.
SAMPLE_STAC_ITEM = {
    "id": "S2B_35KMS_20250615_0_L2A",
    "type": "Feature",
    "bbox": [28.9, -18.5, 30.0, -17.5],
    "properties": {
        "datetime": "2025-06-15T08:12:34Z",
        "eo:cloud_cover": 12.4,
    },
    "assets": {
        "red": {"href": "https://sentinel-cogs.s3.amazonaws.com/.../B04.tif", "type": "image/tiff"},
        "nir": {"href": "https://sentinel-cogs.s3.amazonaws.com/.../B08.tif", "type": "image/tiff"},
        "swir16": {"href": "https://sentinel-cogs.s3.amazonaws.com/.../B11.tif", "type": "image/tiff"},
        "green": {"href": "https://sentinel-cogs.s3.amazonaws.com/.../B03.tif", "type": "image/tiff"},
    },
}

SAMPLE_SEARCH_RESPONSE = {
    "type": "FeatureCollection",
    "features": [SAMPLE_STAC_ITEM],
    "links": [],
}


def _mock_session(json_body, status_code: int = 200) -> MagicMock:
    session = MagicMock()
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests

        response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    session.post.return_value = response
    session.get.return_value = response
    return session


class TestSearchObservations:
    def test_valid_query_returns_features(self) -> None:
        session = _mock_session(SAMPLE_SEARCH_RESPONSE)
        provider = Sentinel2Provider(session=session)

        results = provider.search_observations(
            bbox=(28.0, -19.0, 31.0, -17.0),
            start_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
        )
        assert len(results) == 1
        assert results[0]["id"] == "S2B_35KMS_20250615_0_L2A"

    def test_search_payload_uses_correct_collection(self) -> None:
        session = _mock_session(SAMPLE_SEARCH_RESPONSE)
        provider = Sentinel2Provider(session=session)

        provider.search_observations(
            bbox=(28.0, -19.0, 31.0, -17.0),
            start_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
        )
        call_kwargs = session.post.call_args
        assert call_kwargs.kwargs["json"]["collections"] == ["sentinel-2-l2a"]

    def test_cloud_cover_filter_included_when_specified(self) -> None:
        session = _mock_session(SAMPLE_SEARCH_RESPONSE)
        provider = Sentinel2Provider(session=session)

        provider.search_observations(
            bbox=(28.0, -19.0, 31.0, -17.0),
            start_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
            max_cloud_cover=20.0,
        )
        payload = session.post.call_args.kwargs["json"]
        assert payload["query"]["eo:cloud_cover"]["lt"] == 20.0

    def test_no_cloud_cover_filter_when_not_specified(self) -> None:
        session = _mock_session(SAMPLE_SEARCH_RESPONSE)
        provider = Sentinel2Provider(session=session)

        provider.search_observations(
            bbox=(28.0, -19.0, 31.0, -17.0),
            start_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
        )
        payload = session.post.call_args.kwargs["json"]
        assert "query" not in payload

    def test_end_before_start_rejected(self) -> None:
        session = _mock_session(SAMPLE_SEARCH_RESPONSE)
        provider = Sentinel2Provider(session=session)
        with pytest.raises(ValueError):
            provider.search_observations(
                bbox=(28.0, -19.0, 31.0, -17.0),
                start_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
                end_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            )

    def test_invalid_bbox_rejected(self) -> None:
        session = _mock_session(SAMPLE_SEARCH_RESPONSE)
        provider = Sentinel2Provider(session=session)
        with pytest.raises(ValueError):
            provider.search_observations(
                bbox=(31.0, -19.0, 28.0, -17.0),  # west > east
                start_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
                end_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
            )

    def test_network_failure_raises_sentinel2_error(self) -> None:
        import requests

        session = MagicMock()
        session.post.side_effect = requests.ConnectionError("unreachable")
        provider = Sentinel2Provider(session=session)
        with pytest.raises(Sentinel2Error):
            provider.search_observations(
                bbox=(28.0, -19.0, 31.0, -17.0),
                start_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
                end_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
            )

    def test_http_error_status_raises_sentinel2_error(self) -> None:
        session = _mock_session({}, status_code=503)
        provider = Sentinel2Provider(session=session)
        with pytest.raises(Sentinel2Error):
            provider.search_observations(
                bbox=(28.0, -19.0, 31.0, -17.0),
                start_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
                end_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
            )

    def test_empty_features_returns_empty_list(self) -> None:
        session = _mock_session({"type": "FeatureCollection", "features": []})
        provider = Sentinel2Provider(session=session)
        results = provider.search_observations(
            bbox=(28.0, -19.0, 31.0, -17.0),
            start_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
        )
        assert results == []


class TestGetObservation:
    def test_returns_stac_item(self) -> None:
        session = _mock_session(SAMPLE_STAC_ITEM)
        provider = Sentinel2Provider(session=session)
        item = provider.get_observation("S2B_35KMS_20250615_0_L2A")
        assert item["id"] == "S2B_35KMS_20250615_0_L2A"

    def test_network_failure_raises_sentinel2_error(self) -> None:
        import requests

        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("unreachable")
        provider = Sentinel2Provider(session=session)
        with pytest.raises(Sentinel2Error):
            provider.get_observation("some-id")


class TestGetMetadata:
    def test_returns_properties_dict(self) -> None:
        session = _mock_session(SAMPLE_STAC_ITEM)
        provider = Sentinel2Provider(session=session)
        metadata = provider.get_metadata("S2B_35KMS_20250615_0_L2A")
        assert metadata["eo:cloud_cover"] == 12.4


class TestDownloadAsset:
    def test_unknown_band_rejected(self) -> None:
        session = _mock_session(SAMPLE_STAC_ITEM)
        provider = Sentinel2Provider(session=session)
        with pytest.raises(ValueError):
            provider.download_asset("some-id", "not_a_real_band", "/tmp/out.tif")

    def test_missing_asset_raises_sentinel2_error(self) -> None:
        session = _mock_session(SAMPLE_STAC_ITEM)  # no 'swir22' asset in the sample
        provider = Sentinel2Provider(session=session)
        with pytest.raises(Sentinel2Error):
            provider.download_asset("S2B_35KMS_20250615_0_L2A", "swir22", "/tmp/out.tif")

    def test_downloads_and_writes_file(self, tmp_path) -> None:
        session = MagicMock()

        item_response = MagicMock()
        item_response.json.return_value = SAMPLE_STAC_ITEM
        item_response.raise_for_status = MagicMock()

        download_response = MagicMock()
        download_response.raise_for_status = MagicMock()
        download_response.iter_content.return_value = [b"fake-tiff-bytes"]

        session.get.side_effect = [item_response, download_response]

        provider = Sentinel2Provider(session=session)
        dest = tmp_path / "band.tif"
        result_path = provider.download_asset("S2B_35KMS_20250615_0_L2A", "red", str(dest))

        assert result_path == str(dest)
        assert dest.read_bytes() == b"fake-tiff-bytes"


class TestReadSceneBands:
    def test_unknown_band_rejected(self) -> None:
        provider = Sentinel2Provider(session=MagicMock())
        aoi = AOI(label="test", west=10.0, south=-20.0, east=11.0, north=-19.0)
        with pytest.raises(ValueError):
            provider.read_scene_bands(SAMPLE_STAC_ITEM, ["not_a_real_band"], aoi)

    def test_missing_asset_in_item_raises_sentinel2_error(self) -> None:
        provider = Sentinel2Provider(session=MagicMock())
        aoi = AOI(label="test", west=10.0, south=-20.0, east=11.0, north=-19.0)
        # SAMPLE_STAC_ITEM has no 'swir22' asset.
        with pytest.raises(Sentinel2Error):
            provider.read_scene_bands(SAMPLE_STAC_ITEM, ["swir22"], aoi)

    def test_reads_real_bands_from_local_geotiffs(self, tmp_path) -> None:
        # End-to-end with real local GeoTIFFs standing in for COGs, exercising
        # the full read_scene_bands -> read_band_window -> rasterio path.
        size, resolution = 50, 10.0
        transform = from_origin(500_000.0, 7_800_000.0, resolution, resolution)

        red_path = tmp_path / "red.tif"
        nir_path = tmp_path / "nir.tif"
        for path, fill_value in [(red_path, 800), (nir_path, 3000)]:
            with rasterio.open(
                path, "w", driver="GTiff", height=size, width=size, count=1,
                dtype=np.uint16, crs="EPSG:32735", transform=transform,
            ) as dst:
                dst.write(np.full((size, size), fill_value, dtype=np.uint16), 1)

        import pyproj

        utm_to_wgs84 = pyproj.Transformer.from_crs("EPSG:32735", "EPSG:4326", always_xy=True)
        lon_w, lat_s = utm_to_wgs84.transform(500_000.0, 7_800_000.0 - size * resolution)
        lon_e, lat_n = utm_to_wgs84.transform(500_000.0 + size * resolution, 7_800_000.0)
        aoi = AOI(label="local test", west=lon_w, south=lat_s, east=lon_e, north=lat_n)

        item = {
            "id": "local-test-item",
            "assets": {
                "red": {"href": str(red_path)},
                "nir": {"href": str(nir_path)},
            },
        }

        provider = Sentinel2Provider(session=MagicMock())
        bands = provider.read_scene_bands(item, ["red", "nir"], aoi)

        assert set(bands.keys()) == {"red", "nir"}
        assert bands["red"].mean() == pytest.approx(800, abs=1)
        assert bands["nir"].mean() == pytest.approx(3000, abs=1)

    def test_mixed_resolution_bands_return_identical_shapes(self, tmp_path) -> None:
        # Regression test for a real bug caught by a live Sentinel-2 query:
        # RED/NIR are natively 10m, SWIR16 is natively 20m. Reading the
        # identical AOI from each independently used to return arrays of
        # DIFFERENT pixel dimensions (a 2x mismatch), which broke
        # compute_nbr()'s shape check. read_scene_bands() must resample
        # every band to a single common shape.
        origin_x, origin_y = 500_000.0, 7_800_000.0
        aoi_size_m = 500.0  # 500m x 500m AOI

        # RED and NIR: 10m native resolution -> 50x50 pixels over the AOI.
        red_path = tmp_path / "red_10m.tif"
        nir_path = tmp_path / "nir_10m.tif"
        fine_size = int(aoi_size_m / 10.0)
        fine_transform = from_origin(origin_x, origin_y, 10.0, 10.0)
        for path, fill_value in [(red_path, 800), (nir_path, 3000)]:
            with rasterio.open(
                path, "w", driver="GTiff", height=fine_size, width=fine_size, count=1,
                dtype=np.uint16, crs="EPSG:32735", transform=fine_transform,
            ) as dst:
                dst.write(np.full((fine_size, fine_size), fill_value, dtype=np.uint16), 1)

        # SWIR16: 20m native resolution -> only 25x25 pixels over the SAME
        # real-world AOI. This is the actual source of the bug.
        swir_path = tmp_path / "swir16_20m.tif"
        coarse_size = int(aoi_size_m / 20.0)
        coarse_transform = from_origin(origin_x, origin_y, 20.0, 20.0)
        with rasterio.open(
            swir_path, "w", driver="GTiff", height=coarse_size, width=coarse_size, count=1,
            dtype=np.uint16, crs="EPSG:32735", transform=coarse_transform,
        ) as dst:
            dst.write(np.full((coarse_size, coarse_size), 1400, dtype=np.uint16), 1)

        assert fine_size != coarse_size  # sanity check the test setup itself is valid

        import pyproj

        utm_to_wgs84 = pyproj.Transformer.from_crs("EPSG:32735", "EPSG:4326", always_xy=True)
        lon_w, lat_s = utm_to_wgs84.transform(origin_x, origin_y - aoi_size_m)
        lon_e, lat_n = utm_to_wgs84.transform(origin_x + aoi_size_m, origin_y)
        aoi = AOI(label="mixed resolution test", west=lon_w, south=lat_s, east=lon_e, north=lat_n)

        item = {
            "id": "mixed-res-test-item",
            "assets": {
                "red": {"href": str(red_path)},
                "nir": {"href": str(nir_path)},
                "swir16": {"href": str(swir_path)},
            },
        }

        provider = Sentinel2Provider(session=MagicMock())
        bands = provider.read_scene_bands(item, ["red", "nir", "swir16"], aoi, target_resolution_m=10.0)

        # The whole point of the fix: all three must come back identically shaped,
        # even though swir16's source file has half the pixel dimensions of red/nir.
        assert bands["red"].shape == bands["nir"].shape == bands["swir16"].shape

        # And the values should still be sane after resampling.
        assert bands["red"].mean() == pytest.approx(800, abs=5)
        assert bands["nir"].mean() == pytest.approx(3000, abs=5)
        assert bands["swir16"].mean() == pytest.approx(1400, abs=5)

    def test_downstream_nbr_calculation_works_after_fix(self, tmp_path) -> None:
        # The actual failure mode from the live bug report: compute_nbr()
        # raising a shape-mismatch ValueError. Confirm the full real pipeline
        # (read_scene_bands -> normalize_band -> compute_nbr) now succeeds.
        from src.remote_sensing.nbr import compute_nbr
        from src.remote_sensing.spectral import normalize_band

        origin_x, origin_y = 500_000.0, 7_800_000.0
        aoi_size_m = 300.0

        nir_path = tmp_path / "nir.tif"
        fine_size = int(aoi_size_m / 10.0)
        with rasterio.open(
            nir_path, "w", driver="GTiff", height=fine_size, width=fine_size, count=1,
            dtype=np.uint16, crs="EPSG:32735", transform=from_origin(origin_x, origin_y, 10.0, 10.0),
        ) as dst:
            dst.write(np.full((fine_size, fine_size), 3200, dtype=np.uint16), 1)

        swir_path = tmp_path / "swir16.tif"
        coarse_size = int(aoi_size_m / 20.0)
        with rasterio.open(
            swir_path, "w", driver="GTiff", height=coarse_size, width=coarse_size, count=1,
            dtype=np.uint16, crs="EPSG:32735", transform=from_origin(origin_x, origin_y, 20.0, 20.0),
        ) as dst:
            dst.write(np.full((coarse_size, coarse_size), 1400, dtype=np.uint16), 1)

        import pyproj

        utm_to_wgs84 = pyproj.Transformer.from_crs("EPSG:32735", "EPSG:4326", always_xy=True)
        lon_w, lat_s = utm_to_wgs84.transform(origin_x, origin_y - aoi_size_m)
        lon_e, lat_n = utm_to_wgs84.transform(origin_x + aoi_size_m, origin_y)
        aoi = AOI(label="nbr pipeline test", west=lon_w, south=lat_s, east=lon_e, north=lat_n)

        item = {
            "id": "nbr-pipeline-test",
            "assets": {"nir": {"href": str(nir_path)}, "swir16": {"href": str(swir_path)}},
        }

        provider = Sentinel2Provider(session=MagicMock())
        bands = provider.read_scene_bands(item, ["nir", "swir16"], aoi)

        nir = normalize_band(bands["nir"])
        swir16 = normalize_band(bands["swir16"])
        nbr = compute_nbr(nir, swir16)  # this line is exactly what raised ValueError before the fix

        assert nbr.shape == bands["nir"].shape
        assert not np.all(np.isnan(nbr))


class TestBandAssetKeys:
    def test_covers_bands_needed_for_ndvi_and_nbr(self) -> None:
        # NDVI needs red+nir, NBR needs nir+swir16 -- confirm all four are mapped.
        assert {"red", "nir", "swir16"}.issubset(BAND_ASSET_KEYS.keys())
