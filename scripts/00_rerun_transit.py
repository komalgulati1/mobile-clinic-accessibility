"""
00_rerun_transit.py
===================
Compute transit travel time matrix averaged over 5 departure times.

Run ONCE before the main pipeline if you have r5py installed and your own
GTFS feed. Otherwise use the pre-computed tt_transit_r7_avg.csv provided
in data/processed/.

Requires:
    r5py              (pip install r5py)
    data/raw/rockingham.osm.pbf
    data/raw/SKAT.zip  (or any GTFS feed covering your study area)

Output:
    data/processed/tt_transit_r7_avg.csv

Usage:
    python scripts/00_rerun_transit.py

References:
    Departure time averaging follows Conway et al. (2017) — sampling 5
    departure times at 15-min intervals within the AM peak window.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime

from config import (
    OSM_PBF, GTFS_ZIP,
    HEX_CENTROIDS, DATA_PRO,
)

# ── Departure times — 5 × 15-min intervals in AM peak ─────────────────────────
# Adjust date to a weekday within your GTFS feed's service calendar.
DEPARTURE_TIMES = [
    datetime(2024, 10, 15, 9,  0),
    datetime(2024, 10, 15, 9, 15),
    datetime(2024, 10, 15, 9, 30),
    datetime(2024, 10, 15, 9, 45),
    datetime(2024, 10, 15, 10, 0),
]

OUT_CSV = DATA_PRO / 'tt_transit_r7_avg.csv'

# ── Check inputs exist ─────────────────────────────────────────────────────────
for fpath in [OSM_PBF, GTFS_ZIP, HEX_CENTROIDS]:
    if not fpath.exists():
        raise FileNotFoundError(
            f"Required input not found: {fpath}\n"
            f"See README.md — Data Setup section.")

# ── Load hexagon centroids ─────────────────────────────────────────────────────
print('Loading hex centroids...')
hexes = pd.read_csv(HEX_CENTROIDS)
print(f'  {len(hexes)} hexagons')

# ── Build r5py transport network ───────────────────────────────────────────────
try:
    import r5py
except ImportError:
    raise ImportError(
        "r5py not found. Install with: pip install r5py\n"
        "Also requires Java 11+. See https://r5py.readthedocs.io")

print('Building r5py transport network...')
t0 = time.time()
transport_network = r5py.TransportNetwork(str(OSM_PBF), [str(GTFS_ZIP)])
print(f'  Network built in {time.time()-t0:.1f}s')

# ── Build GeoDataFrame of origins/destinations ────────────────────────────────
origins = gpd.GeoDataFrame(
    hexes,
    geometry=gpd.points_from_xy(hexes['lon'], hexes['lat']),
    crs='EPSG:4326'
).rename(columns={'hex_id': 'id'})
destinations = origins.copy()

# ── Compute travel time matrix at each departure time ─────────────────────────
all_matrices = []

for idx, dep_time in enumerate(DEPARTURE_TIMES):
    print(f'\nDeparture {idx+1}/{len(DEPARTURE_TIMES)}: '
          f'{dep_time.strftime("%H:%M")}')
    t0 = time.time()

    ttm = r5py.TravelTimeMatrixComputer(
        transport_network,
        origins=origins,
        destinations=destinations,
        departure=dep_time,
        transport_modes=[r5py.TransportMode.TRANSIT,
                         r5py.TransportMode.WALK],
        max_time=pd.Timedelta(hours=3),
    )
    res = ttm.compute_travel_times()
    res.columns = ['from_hex', 'to_hex', f'tt_{idx}']
    res[f'tt_{idx}'] = res[f'tt_{idx}'].replace(-1, np.nan)  # -1 = unreachable
    all_matrices.append(res)

    reachable = res[f'tt_{idx}'].notna().sum()
    print(f'  Done in {time.time()-t0:.1f}s  |  '
          f'Reachable pairs: {reachable:,}')

# ── Average across departure times ────────────────────────────────────────────
print('\nAveraging across departure times...')
merged = all_matrices[0][['from_hex', 'to_hex', 'tt_0']]
for i in range(1, len(DEPARTURE_TIMES)):
    merged = merged.merge(all_matrices[i], on=['from_hex', 'to_hex'],
                          how='outer')

tt_cols = [f'tt_{i}' for i in range(len(DEPARTURE_TIMES))]
merged['travel_time_min'] = merged[tt_cols].mean(axis=1, skipna=True)
output = merged[['from_hex', 'to_hex', 'travel_time_min']].copy()

# ── Summary stats ──────────────────────────────────────────────────────────────
total      = len(output)
reachable  = output['travel_time_min'].notna().sum()
mean_tt    = output['travel_time_min'].mean()
pct_30min  = (output['travel_time_min'] <= 30).sum() / total * 100

print(f'\nSummary:')
print(f'  Total OD pairs:           {total:,}')
print(f'  Reachable (avg):          {reachable:,}  ({reachable/total*100:.1f}%)')
print(f'  Mean transit time:        {mean_tt:.1f} min')
print(f'  Reachable within 30 min:  {pct_30min:.2f}% of all pairs')

# ── Save ───────────────────────────────────────────────────────────────────────
output.to_csv(OUT_CSV, index=False)
print(f'\nSaved: {OUT_CSV}')
print('Next: python scripts/01_accessibility_pipeline.py')
