# GeoWatch

### Open Satellite Environmental Intelligence Platform

*Monitor our changing planet using open Earth-observation data.*

---

## Project overview

GeoWatch is an evolving, open-source platform for monitoring wildfires,
vegetation change, and land disturbance using free and open
Earth-observation data (NASA FIRMS, Sentinel, Landsat). It started as a
small NumPy image-processing exercise and is being built up
**incrementally, milestone by milestone**, into a small but credible
environmental intelligence tool — not a collection of disconnected
notebooks.

This repository was originally **Satellite-Image-Processing-with-Numpy**.
That original work is preserved unchanged and now lives at
[`notebooks/01_image_processing.ipynb`](notebooks/01_image_processing.ipynb)
as **Phase 1** of GeoWatch.

## The core question

Every part of this system exists to answer one question, responsibly:

> **What has changed on the Earth, where did it change, when did it
> change, how confident are we, and what evidence supports the
> detection?**

GeoWatch deliberately distinguishes between an **observation**, a
**detected** change, an **inferred** pattern, a **predicted**
classification, and a **confirmed** event. It will never claim more
certainty than the underlying evidence supports — for example, land
disturbance near a mining area is reported as *"potential
mining-related land disturbance,"* never as a confirmed illegal
activity. This distinction is enforced in code via the `EvidenceLevel`
type in [`src/types.py`](src/types.py), not just described in prose.

## Current status: MVP complete (Milestones 0-5) + vegetation decline engine + event time-series + real Sentinel-2 imagery + live Sentinel-2 wildfire detection

This repository is being built one working, tested milestone at a
time. **Nothing here is skipped or faked** — each milestone below is
a real, runnable checkpoint before the next one starts.

| Milestone | Scope | Status |
|---|---|---|
| 0 | Repo restructure, project skeleton, evidence-level types, test harness | done |
| 1 | Spectral index engine: NDVI, NBR, dNBR (deterministic, unit-tested) | done |
| 2 | Wildfire detection: burned-area mask, configurable severity, affected area | done |
| 3 | NASA FIRMS live ingestion, PostGIS-backed event store, interactive map | done |
| 4 | Streamlit dashboard (Overview, Live Map, Fire Monitor) | done |
| 5 | Automated intelligence report (JSON + CSV export) | done |
| 6 (post-MVP) | Vegetation decline detection engine (dNDVI-based) | engine done, not yet in dashboard |
| 7 (post-MVP) | Event time-series tracking (real FIRMS data, live in dashboard) | done |
| 8 (post-MVP) | Sentinel-2 provider via Earth Search (real optical imagery, zero auth) | done |
| 9 (post-MVP) | Sentinel-2 -> wildfire detection -> event -> live map pipeline | done |

**The original Milestone 0-5 MVP is complete.** Post-MVP work has added
three things, each honestly scoped to what it actually does:

- **A vegetation-change detection engine**
  (`src/detection/vegetation.py`, `compute_dndvi()`), built and fully
  tested the same way every wildfire module was.
- **Live event time-series tracking**
  (`src/monitoring/timeseries.py`) — genuinely live, aggregating real
  stored FIRMS events by day into a trend chart in the Fire Monitor
  tab. The trend label is an explicitly simple heuristic (first-half
  vs second-half mean comparison), not a statistical test — see the
  module docstring before treating it as more than a rough signal.
- **A real Sentinel-2 provider** (`src/ingestion/sentinel2.py`), via
  [Earth Search](https://earth-search.aws.element84.com/v1) —
  Element84's free, public STAC API. Unlike NASA FIRMS, this requires
  **zero authentication** — no API key, no registration. See
  [`notebooks/07_sentinel2_real_imagery.ipynb`](notebooks/07_sentinel2_real_imagery.ipynb),
  which searches and reads real Sentinel-2 pixels directly if you have
  network access (falling back to clearly-labeled synthetic data if not).

**What this does and doesn't unlock, precisely:** the Sentinel-2
provider is now wired into a real end-to-end pipeline — search for a
pre-fire and post-fire scene, read NIR/SWIR16 bands directly from the
COGs, compute dNBR, run burned-area detection
(`src/detection/wildfire.py`), cluster the result into geolocated
events (`src/detection/wildfire_events.py`), and store them in
PostGIS — all runnable from the dashboard's **Wildfire (Sentinel-2)**
tab, no API key required. This closes the loop the Fire Monitor tab
already established for point-based FIRMS detections, but for
imagery-derived burned-area regions instead.

**Vegetation decline is not yet part of this live pipeline** — the
engine exists and is fully tested (`src/detection/vegetation.py`), but
nothing yet converts a `VegetationDeclineResult` into map events the
way `wildfire_result_to_events()` now does for fire. That's the
natural next piece. See
[`notebooks/06_vegetation_decline.ipynb`](notebooks/06_vegetation_decline.ipynb)
for the validated-but-not-yet-live engine.

Everything beyond this — Sentinel-1/SAR, machine learning, time-series
recovery tracking, an AI explainer layer, event streaming — is
documented **roadmap**, not current functionality. See
[Roadmap](#roadmap) below.

## Architecture

```
data/                   raw / processed / sample imagery (not committed; see .gitignore)
notebooks/
    01_image_processing.ipynb     Phase 1 — original NumPy foundation
    02_ndvi.ipynb                  Phase 2 — NDVI/NBR/dNBR spectral engine demo
    03_wildfire_detection.ipynb    Phase 3 — burned-area + severity detection demo
    04_live_fire_monitoring.ipynb  Phase 4 — FIRMS ingestion + PostGIS + map demo
    05_intelligence_report.ipynb   Phase 5 — automated report generation demo
    06_vegetation_decline.ipynb    Phase 7 — vegetation decline engine demo (post-MVP)
    07_sentinel2_real_imagery.ipynb Phase 8 — real Sentinel-2 imagery via Earth Search (post-MVP)
src/
    types.py             EvidenceLevel, GeoWatchEvent, ConfidenceScore — shared contracts
    ingestion/           SatelliteDataProvider abstraction; firms.py (fire points) and
                          sentinel2.py (real optical imagery, zero auth) are the two providers
    preprocessing/       imagery.py: COG windowed reads with automatic CRS reprojection
    remote_sensing/      NDVI, NBR, spectral index calculations
    geospatial/          AOI, geometry, area calculations
    detection/           wildfire, vegetation, disturbance detection
    models/              interpretable ML baselines (roadmap)
    monitoring/          events.py (EventStore interface + in-memory store),
                          postgres_store.py (real PostGIS-backed store),
                          map_view.py (Folium event map), time-series tracking
    reporting/           automated intelligence reports (JSON/CSV export), reports.py
    ai/                  optional AI explanation layer (roadmap)
app/
    dashboard.py         Streamlit dashboard entry point
tests/                   pytest suite, mirrors src/ structure
```

Scientific/analytical logic in `src/` is kept independent of the
dashboard in `app/`, so the detection engine can be tested, reused, or
exposed via an API without depending on Streamlit.

## Data sources

GeoWatch prioritizes open, free Earth-observation data and avoids
scraping — only official APIs and catalogues:

- **NASA FIRMS** — near-real-time active-fire detections (free `MAP_KEY`, 5,000 requests / 10-min window). Live as of Milestone 3 via `src/ingestion/firms.py`.
- **Sentinel-2 (via Earth Search)** — optical imagery (NIR/RED/SWIR bands). Live via `src/ingestion/sentinel2.py`; zero authentication required. Not yet wired into detection/dashboard — see status section above.
- **Sentinel-1 SAR** — radar imagery (planned; Earth Search also hosts this collection, so a future provider would reuse the same pattern)
- **USGS Landsat** — optical imagery (planned)
- **OpenStreetMap** — infrastructure/context data (planned)

The system is designed around a `SatelliteDataProvider` abstraction
(see `src/ingestion/base.py`) so no module is hard-coded to a single
provider.

## Installation

```bash
git clone https://github.com/tpchiripa/Satellite-Image-Processing-with-Numpy.git
cd Satellite-Image-Processing-with-Numpy
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick start

```bash
pytest tests/ -v
```

That covers all pure-Python tests (spectral engine, wildfire detection,
FIRMS provider parsing with mocked responses). Two additional pieces
need setup before you get the full picture:

**PostGIS database** (required for `test_postgres_store.py` and the
Milestone 3 notebook's persistence step):

```bash
docker compose up -d db
TEST_DATABASE_URL=postgresql://geowatch:geowatch@localhost:5439/geowatch pytest tests/test_postgres_store.py -v
```

**NASA FIRMS MAP_KEY** (optional — enables live fire data instead of the
notebook's labeled sample rows): get a free key at
https://firms.modaps.eosdis.nasa.gov/api/area/ (see "Map Key"), then:

```bash
# Windows (persists across sessions, restart terminal after):
setx FIRMS_MAP_KEY "your-key-here"
# macOS/Linux:
export FIRMS_MAP_KEY="your-key-here"
```

Never commit a MAP_KEY to git or paste it into a chat — treat it like
any other API credential.

**Run the dashboard:**

```bash
streamlit run app/dashboard.py
```

Works with zero configuration (falls back to an in-memory store and an
empty state with setup instructions), but is more useful with both
`DATABASE_URL` and `FIRMS_MAP_KEY` set — persistence plus live fire
data. Four tabs: Overview, Live Event Map, Fire Monitor, Wildfire (Sentinel-2).

## Responsible use

Environmental intelligence can have real-world consequences.
GeoWatch is built around the following non-negotiable principles:

- Never present a model classification or detected change as a confirmed fact.
- Always distinguish **observed** / **detected** / **inferred** / **predicted** / **confirmed**.
- Never label a location as "illegal" activity from satellite imagery alone — only "potential" activity, with supporting evidence and a confidence score.
- Burn-severity thresholds and other classification cutoffs are methodology-dependent, not universal ground truth, and are documented as such wherever they appear.

## Roadmap

Beyond the current milestone plan, the long-term vision includes:
Sentinel-1/SAR-based disturbance detection, an interpretable
machine-learning baseline (Random Forest → XGBoost → CNN/segmentation),
time-series vegetation-recovery tracking, an optional AI explanation
layer that narrates — but never calculates — GeoWatch's deterministic
metrics, and eventual event-driven/streaming architecture for
near-real-time processing.

## Contributing

This is currently a personal, incrementally-built project. Issues and
suggestions are welcome; larger contributions will be easier to accept
once the Milestone 0-5 MVP lands and the module interfaces stabilize.

## License

No license file has been added yet — treat this repository as
"all rights reserved" until one is added.

## Disclaimer

This project is created for **educational purposes** to demonstrate
satellite image processing and environmental monitoring techniques.
It is not an operational disaster-response or law-enforcement tool.
Outputs should be independently verified before being acted upon.
