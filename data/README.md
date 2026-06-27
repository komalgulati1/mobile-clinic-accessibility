# Data

This folder holds input data required by the pipeline.
**Neither subfolder is tracked by git** (data is large / proprietary).

```
data/
├── raw/                        ← source data, never modified
│   ├── rockingham_POIs.csv     ← Dewey POI visit data (Rockingham County)
│   ├── acs_rockingham.json     ← ACS population by block group (Census API)
│   ├── acs_vehicles_tract.json ← ACS vehicle availability by tract (Census API)
│   ├── rockingham.osm.pbf      ← OpenStreetMap extract (for transit routing)
│   └── SKAT.zip                ← SKAT GTFS feed (for transit routing)
│
└── processed/                  ← derived from raw, created by preprocessing
    ├── rockingham_hex_r7.gpkg           ← H3 res-7 hex grid (GeoPackage)
    ├── rockingham_hex_centroids_r7.csv  ← hex centroids (hex_id, lat, lon)
    ├── tt_car_r7.csv                    ← car travel time OD matrix (min)
    ├── tt_transit_r7.csv                ← transit OD matrix, single run (min)
    ├── tt_transit_r7_avg.csv            ← transit OD matrix, 5-departure avg ★
    ├── hospital_access_summary.csv      ← per-hex hospital reachability flags
    └── rockingham_hospitals.csv         ← hospital locations (lat, lon)
```

★ `tt_transit_r7_avg.csv` is produced by `scripts/00_rerun_transit.py`.
  If you skip this step, the pipeline falls back to `tt_transit_r7.csv`.

## How to obtain the data

### Hex grid and travel times (processed/)
These are produced by the preprocessing scripts in this repo
(not included in the public release due to size).
Contact the authors or see the methods section of the paper.

### POI data (raw/)
Obtained from SafeGraph/Dewey via academic access.
See: https://www.safegraph.com/academics

### ACS data (raw/)
Downloaded from the US Census Bureau API:
```python
# Example — ACS 5-year 2021, block group level, Rockingham County NC
# State=37 (NC), County=157 (Rockingham)
import requests
url = ("https://api.census.gov/data/2021/acs/acs5"
       "?get=B01003_001E,B01001_020E,...&for=block%20group:*"
       "&in=state:37%20county:157&key=YOUR_API_KEY")
```

### OpenStreetMap (raw/)
```bash
# Download NC extract from Geofabrik and clip to Rockingham County
wget https://download.geofabrik.de/north-america/us/north-carolina-latest.osm.pbf
osmium extract --bbox=-80.1,36.1,-79.5,36.6 north-carolina-latest.osm.pbf -o rockingham.osm.pbf
```

### SKAT GTFS (raw/)
Obtained from the SKAT transit agency (Rockingham County, NC).
Public GTFS feeds may be available via transit.land or the agency directly.
