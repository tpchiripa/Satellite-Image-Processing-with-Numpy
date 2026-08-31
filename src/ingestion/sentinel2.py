"""
Sentinel-2 provider via Earth Search (Element84's public STAC API on AWS).

    https://earth-search.aws.element84.com/v1

Unlike NASA FIRMS, this requires NO authentication and NO API key —
Earth Search's STAC catalog is fully public, and its backing COG assets
live in a public, unsigned-access S3 bucket. This is deliberately why
Earth Search was chosen over Copernicus Data Space (OAuth2) or
Microsoft Planetary Computer (SAS token signing) — it fits GeoWatch's
"keep the MVP lightweight" principle far better.

Two-step access pattern, matching how COGs are meant to be used:

    1. search_observations() -> STAC search, returns scene metadata
       (id, datetime, cloud cover, and asset URLs — no pixels yet).
    2. read_scene_bands() -> for a chosen scene + AOI, read only the
       pixels that fall within the AOI directly from the COG (via
       src/preprocessing/imagery.py), without downloading the full
       ~100MB+ per-band file.

download_asset() (the full-file download from SatelliteDataProvider's
interface) is also implemented, for cases where the whole band is
genuinely needed — but read_scene_bands() is the practical entry point
most GeoWatch code should use.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import requests

from src.geospatial.aoi import AOI
from src.ingestion.base import SatelliteDataProvider
from src.preprocessing.imagery import compute_output_shape, read_band_window

EARTH_SEARCH_BASE_URL = "https://earth-search.aws.element84.com/v1"
SENTINEL2_COLLECTION = "sentinel-2-l2a"

# GeoWatch's canonical band names -> Earth Search's STAC asset keys for
# the sentinel-2-l2a collection. See module docstring for the source of
# this mapping (confirmed against Earth Search v1's actual asset keys).
BAND_ASSET_KEYS = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir": "nir",
    "swir16": "swir16",  # used for NBR
    "swir22": "swir22",
    "scl": "scl",  # Scene Classification Layer (cloud/shadow/etc masking)
}


class Sentinel2Error(Exception):
    """Raised for Earth Search API failures GeoWatch cannot recover from automatically."""


class Sentinel2Provider(SatelliteDataProvider):
    """SatelliteDataProvider implementation for Sentinel-2 L2A via Earth Search."""

    name = "Sentinel-2 (Earth Search)"

    def __init__(self, base_url: str = EARTH_SEARCH_BASE_URL, session: Optional[requests.Session] = None):
        """
        Args:
            base_url: Earth Search API root. Overridable for testing or
                pointing at a different STAC API entirely.
            session: optional requests.Session, mainly for test injection.
        """
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()

    def search_observations(
        self,
        bbox: tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: Optional[float] = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search for Sentinel-2 L2A scenes covering a bbox and date range.

        Args:
            bbox: (west, south, east, north) in decimal degrees (WGS84).
            start_date: earliest acquisition date of interest.
            end_date: latest acquisition date of interest. Must be >= start_date.
            max_cloud_cover: if given, only return scenes with
                eo:cloud_cover strictly less than this percentage (0-100).
            limit: maximum number of scenes to return. Earth Search
                paginates beyond this, but GeoWatch's AOI-scale queries
                shouldn't need to page through results.
            **kwargs: unused, accepted for interface consistency.

        Returns:
            List of raw STAC Item dicts (GeoJSON Features), each
            containing 'id', 'properties' (datetime, eo:cloud_cover,
            etc.), 'assets' (band URLs), and 'bbox'/'geometry'.

        Raises:
            ValueError: for an invalid bbox or date range.
            Sentinel2Error: on network failure or an unparseable response.
        """
        if end_date < start_date:
            raise ValueError(
                f"search_observations: end_date ({end_date}) is before start_date ({start_date})"
            )
        west, south, east, north = bbox
        AOI(label="query", west=west, south=south, east=east, north=north)  # validates bounds

        payload: dict[str, Any] = {
            "collections": [SENTINEL2_COLLECTION],
            "bbox": [west, south, east, north],
            "datetime": f"{start_date.isoformat()}/{end_date.isoformat()}",
            "limit": limit,
        }
        if max_cloud_cover is not None:
            payload["query"] = {"eo:cloud_cover": {"lt": max_cloud_cover}}

        try:
            response = self._session.post(f"{self._base_url}/search", json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise Sentinel2Error(f"Earth Search query failed: {exc}") from exc
        except ValueError as exc:  # response.json() failed to parse
            raise Sentinel2Error(f"Earth Search returned an unparseable response: {exc}") from exc

        return data.get("features", [])

    def get_observation(self, observation_id: str) -> dict[str, Any]:
        """Fetch a single STAC item by its scene ID.

        Args:
            observation_id: the STAC item ID, as returned in a
                search_observations() result's 'id' field.

        Returns:
            The raw STAC Item dict.

        Raises:
            Sentinel2Error: on network failure, a 404 (unknown scene
                ID), or an unparseable response.
        """
        url = f"{self._base_url}/collections/{SENTINEL2_COLLECTION}/items/{observation_id}"
        try:
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise Sentinel2Error(f"Earth Search item lookup failed for {observation_id!r}: {exc}") from exc
        except ValueError as exc:
            raise Sentinel2Error(f"Earth Search returned an unparseable response: {exc}") from exc

    def get_metadata(self, observation_id: str) -> dict[str, Any]:
        """Return the STAC item's properties dict (datetime, cloud cover, etc.)."""
        item = self.get_observation(observation_id)
        return item.get("properties", {})

    def download_asset(self, observation_id: str, asset_key: str, destination: str) -> str:
        """Download a full band asset to local disk.

        Most GeoWatch code should prefer read_scene_bands() instead,
        which reads only the AOI's pixels directly from the COG without
        downloading the whole file (often 100MB+ per band). This method
        exists for the (less common) case where the full file is
        genuinely needed.

        Args:
            observation_id: STAC item ID.
            asset_key: one of GeoWatch's canonical band names (see
                BAND_ASSET_KEYS), e.g. 'nir', 'red', 'swir16'.
            destination: local file path to write to.

        Returns:
            The destination path.

        Raises:
            ValueError: if asset_key is not a recognized band name.
            Sentinel2Error: on network failure or a missing asset.
        """
        if asset_key not in BAND_ASSET_KEYS:
            raise ValueError(
                f"download_asset: unknown asset_key {asset_key!r}. "
                f"Valid options: {sorted(BAND_ASSET_KEYS)}"
            )

        item = self.get_observation(observation_id)
        stac_asset_key = BAND_ASSET_KEYS[asset_key]
        try:
            href = item["assets"][stac_asset_key]["href"]
        except KeyError as exc:
            raise Sentinel2Error(
                f"Scene {observation_id!r} has no {asset_key!r} asset "
                f"(STAC key {stac_asset_key!r} not found)."
            ) from exc

        try:
            response = self._session.get(href, timeout=120, stream=True)
            response.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        except requests.RequestException as exc:
            raise Sentinel2Error(f"Failed to download {asset_key!r} for {observation_id!r}: {exc}") from exc

        return destination

    def read_scene_bands(
        self,
        item: dict[str, Any],
        bands: list[str],
        aoi: AOI,
        target_resolution_m: float = 10.0,
    ) -> dict[str, np.ndarray]:
        """Read only the AOI's pixels for one or more bands, directly from the COGs.

        This is the practical entry point for GeoWatch's detection
        pipelines — no full-file download, no local disk usage beyond
        what rasterio/GDAL buffers internally.

        IMPORTANT: Sentinel-2 bands are NOT all the same native
        resolution — RED/NIR/GREEN/BLUE are 10m, SWIR16/SWIR22 are 20m.
        Reading two such bands independently for the identical AOI
        returns arrays of DIFFERENT pixel dimensions, which silently
        breaks any element-wise calculation (NDVI, NBR, ...) between
        them. This was caught by a real live query during development,
        not by testing alone. To prevent it, this method always
        resamples every requested band to a single common shape at
        target_resolution_m, computed once from the first band and
        applied to all — so the returned arrays are always guaranteed
        to be identically shaped, regardless of which bands were asked for.

        Args:
            item: a STAC item dict, as returned by search_observations()
                or get_observation().
            bands: list of GeoWatch canonical band names (see
                BAND_ASSET_KEYS), e.g. ['nir', 'red'] for NDVI, or
                ['nir', 'swir16'] for NBR.
            aoi: the Area of Interest to read.
            target_resolution_m: pixel resolution (in meters — Sentinel-2
                tiles use a UTM CRS) every returned band is resampled to.
                Defaults to 10.0, Sentinel-2's finest common optical
                resolution, so 20m bands (like swir16) get upsampled
                rather than 10m bands being downsampled.

        Returns:
            Dict mapping each requested band name to its windowed pixel
            array, all sharing the same (height, width) — raw digital
            numbers (see src/preprocessing/imagery.py's module docstring
            on normalizing to reflectance). Resampled bands come back as
            float64 rather than the source's native integer dtype
            (bilinear resampling is not integer-exact).

        Raises:
            ValueError: if any band name is not recognized, or
                target_resolution_m is not positive.
            Sentinel2Error: if the item is missing a requested asset.
            RasterReadError: if a COG can't be read or the AOI doesn't
                overlap it.
        """
        unknown = [b for b in bands if b not in BAND_ASSET_KEYS]
        if unknown:
            raise ValueError(
                f"read_scene_bands: unknown band(s) {unknown}. Valid options: {sorted(BAND_ASSET_KEYS)}"
            )
        if not bands:
            return {}

        def _href_for(band: str) -> str:
            stac_key = BAND_ASSET_KEYS[band]
            try:
                return item["assets"][stac_key]["href"]
            except KeyError as exc:
                item_id = item.get("id", "<unknown>")
                raise Sentinel2Error(
                    f"Scene {item_id!r} has no {band!r} asset (STAC key {stac_key!r} not found)."
                ) from exc

        # Establish one common output shape from the first band, then apply
        # it to every band — this is what guarantees same-shaped arrays
        # regardless of each band's individual native resolution.
        out_shape = compute_output_shape(_href_for(bands[0]), aoi, target_resolution_m)

        result: dict[str, np.ndarray] = {}
        for band in bands:
            result[band] = read_band_window(_href_for(band), aoi, out_shape=out_shape)

        return result
