# GeoWatch application image.
# Milestone 0: builds and installs dependencies only — no app code runs yet
# (the Streamlit dashboard lands in Milestone 4). This exists now so the
# Docker/Compose workflow is exercised early rather than bolted on later.

FROM python:3.11-slim

# System libs required by rasterio/geopandas/shapely once Milestone 1+ lands.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Milestone 4+ will change this to launch the Streamlit dashboard.
CMD ["python3", "-m", "pytest", "tests/"]
