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

## Current status: Milestone 3 complete

This repository is being built one working, tested milestone at a
time. **Nothing here is skipped or faked** — each milestone below is
a real, runnable checkpoint before the next one starts.

| Milestone | Scope | Status |
|---|---|---|
| 0 | Repo restructure, project skeleton, evidence-level types, test harness | done |
| 1 | Spectral index engine: NDVI, NBR, dNBR (deterministic, unit-tested) | done |
| 2 | Wildfire detection: burned-area mask, configurable severity, affected area | done |
| 3 | NASA FIRMS live ingestion, PostGIS-backed event store, interactive map | done |
| 4 | Streamlit dashboard (Overview, Live Map, Fire Monitor) | next |
| 5 | Basic automated intelligence report (JSON export) | planned |

Everything beyond Milestone 5 — Sentinel-1/SAR, machine learning,
time-series recovery tracking, an AI explainer layer, event streaming —
is documented **roadmap**, not current functionality. See
[Roadmap](#roadmap) below.

## Architecture

```
data/                   raw / processed / sample imagery (not committed; see .gitignore)
notebooks/
    01_image_processing.ipynb     Phase 1 — original NumPy foundation
    02_ndvi.ipynb                  Phase 2 — NDVI/NBR/dNBR spectral engine demo
    03_wildfire_detection.ipynb    Phase 3 — burned-area + severity detection demo
    04_live_fire_monitoring.ipynb  Phase 4 — FIRMS ingestion + PostGIS + map demo
src/
    types.py             EvidenceLevel, GeoWatchEvent, ConfidenceScore — shared contracts
    ingestion/           SatelliteDataProvider abstraction; firms.py is the first real provider
    preprocessing/       image loading, band stacking, masking helpers
    remote_sensing/      NDVI, NBR, spectral index calculations
    geospatial/          AOI, geometry, area calculations
    detection/           wildfire, vegetation, disturbance detection
    models/              interpretable ML baselines (roadmap)
    monitoring/          events.py (EventStore interface + in-memory store),
                          postgres_store.py (real PostGIS-backed store),
                          map_view.py (Folium event map), time-series tracking
    reporting/           intelligence report generation
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
- **Copernicus Sentinel-1 / Sentinel-2** — SAR and optical imagery (planned)
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
