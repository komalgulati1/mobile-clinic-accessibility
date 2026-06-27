"""
config.py  —  Central path configuration for the mobile-clinic-accessibility repo.

All scripts import from here. You ONLY need to edit this file if your data
layout differs from the default (see DATA LAYOUT below).

Default layout (everything relative to repo root):

    mobile-clinic-accessibility/
    ├── config.py               ← this file
    ├── scripts/
    │   ├── 00_rerun_transit.py
    │   ├── 01_accessibility_pipeline.py
    │   ├── 02_generate_heatmaps.py
    │   └── 03_generate_difference_maps.py
    ├── data/
    │   ├── raw/                ← census data, POIs  (not tracked by git)
    │   └── processed/          ← hex grid, travel times  (not tracked by git)
    └── outputs/
        ├── maps/               ← PNG heatmaps and difference maps
        ├── plots/              ← step plots
        └── tables/             ← CSVs (results, selected stops)
"""

import pathlib

# ── Repo root (folder containing this file) ───────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent

# ── Data folders ──────────────────────────────────────────────────────────────
DATA_RAW = ROOT / 'data' / 'raw'        # census JSON, POIs CSV
DATA_PRO = ROOT / 'data' / 'processed'  # hex grid, travel time matrices

# ── Output folders (created automatically) ────────────────────────────────────
OUT_ROOT  = ROOT / 'outputs'
OUT_MAPS  = OUT_ROOT / 'maps'
OUT_PLOTS = OUT_ROOT / 'plots'
OUT_TABS  = OUT_ROOT / 'tables'
OUT_SLIDE = OUT_ROOT / 'slide_maps'

for _d in [OUT_MAPS, OUT_PLOTS, OUT_TABS, OUT_SLIDE]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Raw data files ─────────────────────────────────────────────────────────────
POIS     = DATA_RAW / 'rockingham_POIs.csv'
ACS_POP  = DATA_RAW / 'acs_rockingham.json'
ACS_VEH  = DATA_RAW / 'acs_vehicles_tract.json'

# ── Processed data files ───────────────────────────────────────────────────────
HEX_GRID      = DATA_PRO / 'rockingham_hex_r7.gpkg'
HEX_CENTROIDS = DATA_PRO / 'rockingham_hex_centroids_r7.csv'
TT_CAR        = DATA_PRO / 'tt_car_r7.csv'
TT_TRANSIT    = DATA_PRO / 'tt_transit_r7_avg.csv'  # produced by 00_rerun_transit.py
                                                     # if not yet run, falls back:
TT_TRANSIT_SINGLE = DATA_PRO / 'tt_transit_r7.csv'  # single-run fallback
HOSP_SUMMARY  = DATA_PRO / 'hospital_access_summary.csv'
HOSPITALS_CSV = DATA_PRO / 'rockingham_hospitals.csv'

# ── OSM / GTFS (only needed for 00_rerun_transit.py) ─────────────────────────
OSM_PBF  = DATA_RAW / 'rockingham.osm.pbf'
GTFS_ZIP = DATA_RAW / 'SKAT.zip'

# ── Model parameters ──────────────────────────────────────────────────────────
# Impedance — Verma et al. (2025) Table 2, non-work trips
ALPHA_CAR     = -0.020097;  BETA_CAR     = 1.361630
ALPHA_TRANSIT = -0.002062;  BETA_TRANSIT = 1.608027

# Candidate NAICS codes for mobile clinic stop locations
CANDIDATE_NAICS = [
    813110,  # Religious organisations
    611110,  # Elementary & secondary schools
    624410,  # Child day care services
    621111,  # Offices of physicians
    922110,  # Courts
    922120,  # Police protection
    922130,  # Legal counsel & prosecution
    922140,  # Correctional institutions
    922150,  # Parole offices & probation offices
    922160,  # Fire protection
    922190,  # Other justice, public order & safety
]

# Usage probabilities π_g per population segment
# ec = elderly with car, enc = elderly no car,
# nec = non-elderly with car, nenc = non-elderly no car
PI = {'ec': 0.85, 'enc': 0.90, 'nec': 0.55, 'nenc': 0.60}

# County-level population totals (ACS 2021, Rockingham County NC)
COUNTY_TOTAL   = 91585
COUNTY_ELDERLY = 19008    # population aged 65+
ZV_SHARE       = 0.065    # share of households with zero vehicles

# Weibull parameters for microtransit stochastic model
# Wait time W ~ Weibull(k=2.084, λ=19.901 min) — Yang & Gao (2025)
# Detour ratio δ ~ Weibull(k=1.069, λ=1.3) clipped [1,2] — Hu & Xu (2025)
W_SHAPE = 2.084;  W_SCALE = 19.901
D_SHAPE = 1.069;  D_SCALE = 1.3
N_MC    = 200     # Monte Carlo samples for microtransit uncertainty band

# P values to evaluate
P_VALS = [1, 2, 3, 4]

# NC A&T brand colours
NAVY = '#1a3a5c'
GOLD = '#F0AB00'
TEAL = '#2e7d6e'
