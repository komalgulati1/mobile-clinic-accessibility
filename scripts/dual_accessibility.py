"""
dual_accessibility.py
=====================
Dual accessibility model — Rockingham County, NC
Komal Gulati, CR2C2 / NC A&T

Two complementary travel time minimization models, run across
all 6 transport modes and P=1..50.

Model 1 — Population-weighted travel time minimization (P-median):
    minimize  Σ_i n_i · d_i
    s.t.      d_i <= t_ij^m + BIG_M·(1 - y_j)    ∀i,j
              Σ_j y_j = P
              y_j ∈ {0,1},  d_i >= 0

    n_i = total population in hex i (all segments combined)
    t_ij^m = travel time from hex i to stop j under mode m

Model 2 — True single-BLP min-max fairness (Dr. Pandey correction):
    minimize  T_max
    s.t.      T_max >= t_ij^{m,g} - BIG_M·(1 - y_j)
                       ∀ reachable (i,g,j): n_ig > 0, t_ij < BIG_M
              Σ_j y_j = P
              y_j ∈ {0,1},  T_max >= 0

    ONE BLP per mode per P covering ALL population groups simultaneously.
    A single shared T_max is minimised across all segments (ec, enc, nec, nenc)
    and all reachable hexes at once -- NOT 4 separate group BLPs.
    Per-group T_max and mean_tt are reported post-hoc by evaluating the
    joint-optimal stop set against each group's travel times separately.

Price of Fairness:
    PoF(P,g) = [A_tilde_primal(P) - A_tilde_fair(P,g)] / A_tilde_primal(P)
    where A_tilde_primal comes from primal_accessibility_results.csv
    and A_tilde_fair is the primal accessibility evaluated at the
    stop set chosen by Model 2 for group g.

Runs P=1..50, all 6 transport modes (car_only, car_for_all,
car_microtrans, transit_only, transit_for_all, car_transit).

Outputs:
    outputs/dual_model1_results.csv         — pw_mean_tt, max_tt per mode × P
    outputs/dual_model2_results.csv         — T_max, mean_tt per mode × group × P
    outputs/dual_mayodan_entry.csv          — first P at which Mayodan selected
    outputs/price_of_fairness.csv           — PoF per mode × group × P
    outputs/dual_model1_step_plot.png       — pw_mean_tt vs P all modes
    outputs/dual_model2_step_plot.png       — T_max vs P per group (2x3 grid)
    outputs/dual_pof_plot.png               — PoF curves per group
    outputs/dual_heatmap_{mode}_P{p}.png    — per-hex min travel time maps
"""

import os, json, warnings, time, math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pulp import (LpProblem, LpMinimize, LpVariable, lpSum,
                  LpBinary, value, PULP_CBC_CMD)

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
DATA    = '/Users/komalgulati/Documents/project_3_2/DC_2026'
OUTPUTS = f'{DATA}/outputs'

HEX_CENTROIDS  = f'{OUTPUTS}/rockingham_hex_centroids_r7.csv'
HEX_GRID       = f'{OUTPUTS}/rockingham_hex_r7.gpkg'
POIS           = f'{DATA}/rockingham_POIs.csv'
ACS_POP        = f'{DATA}/acs_rockingham_with_moe.json'
ACS_VEH        = f'{DATA}/acs_vehicles_tract_with_moe.json'
TT_CAR         = f'{OUTPUTS}/tt_car_r7.csv'
TT_TRANSIT     = f'{OUTPUTS}/tt_transit_r7_avg.csv'
HEX_ELDERLY_SE = f'{OUTPUTS}/hexagon_elderly_SE.csv'
HEX_ZVEH_SHARE = f'{OUTPUTS}/hexagon_zveh_share.csv'
PRIMAL_RESULTS = f'{OUTPUTS}/primal_accessibility_results.csv'

CANDIDATE_NAICS = [813110, 611110, 624410, 621111,
                   922110, 922120, 922130, 922140, 922150, 922160, 922190]

# Impedance parameters
ALPHA_CAR = -0.020097;  BETA_CAR = 1.361630
ALPHA_TR  = -0.002062;  BETA_TR  = 1.608027
ALPHA_MC  = ALPHA_CAR;  BETA_MC  = BETA_CAR
SHAPE_W   = 1.79;       SCALE_W  = 4.2
W_MEAN    = SCALE_W * math.gamma(1 + 1/SHAPE_W)   # mean Weibull waiting time ≈ 3.74 min

PI       = {'ec': 0.85, 'enc': 0.90, 'nec': 0.55, 'nenc': 0.60}
P_LIST   = list(range(1, 51))
BIG_M    = 200.0    # max possible travel time (min)
POP_MIN  = 1.0      # min segment population to include hex in Model 2 constraint

# Mayodan town centre
MAYODAN_LAT = 36.4132
MAYODAN_LON = -79.9693

MODE_LABELS = {
    'car_only':        'Car only',
    'car_for_all':     'Car for all',
    'car_microtrans':  'Car + Microtransit',
    'transit_only':    'Transit only',
    'transit_for_all': 'Transit for all',
    'car_transit':     'Car + Transit (SKAT)',
}
MODE_COLORS = {
    'car_only':        '#e74c3c',
    'car_for_all':     '#2ecc71',
    'car_microtrans':  '#e67e22',
    'transit_only':    '#3498db',
    'transit_for_all': '#1abc9c',
    'car_transit':     '#9b59b6',
}
GROUP_LABELS = {
    'ec':   'Elderly + car',
    'enc':  'Elderly + no-car',
    'nec':  'Non-elderly + car',
    'nenc': 'Non-elderly + no-car',
}
GROUP_COLORS = {
    'ec': '#c0392b', 'enc': '#8e44ad',
    'nec': '#2980b9', 'nenc': '#16a085',
}

def tick(label):
    print(f'\n[START] {label}')
    return time.time()

def tock(t0, label):
    print(f'[DONE]  {label}  ({time.time()-t0:.1f}s)')

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Load data')
hexes   = pd.read_csv(HEX_CENTROIDS)
hex_gdf = gpd.read_file(HEX_GRID).to_crs('EPSG:4326')
hex_col = next((c for c in hex_gdf.columns
                if 'hex' in c.lower() or c == 'h3_index'), hex_gdf.columns[0])
hex_gdf = hex_gdf.rename(columns={hex_col: 'hex_id'})
hex_ids = hexes['hex_id'].tolist()
N_HEX   = len(hex_ids)
pois    = pd.read_csv(POIS, low_memory=False)
tt_car  = pd.read_csv(TT_CAR)

with open(ACS_POP) as f: acs_pop = json.load(f)
with open(ACS_VEH) as f: acs_veh = json.load(f)
print(f'  Hexagons: {N_HEX}')
tock(t0, 'Load data')

# ══════════════════════════════════════════════════════════════════════════════
# 2. Build candidate set J
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Build candidate set J')
pois['naics_code'] = pd.to_numeric(pois['naics_code'], errors='coerce')
cand = pois[pois['naics_code'].isin(CANDIDATE_NAICS)].copy()

poi_gdf = gpd.GeoDataFrame(cand,
    geometry=gpd.points_from_xy(cand['longitude'], cand['latitude']),
    crs='EPSG:4326')
if 'index_right' in poi_gdf.columns:
    poi_gdf = poi_gdf.drop(columns=['index_right'])

joined = gpd.sjoin(poi_gdf,
                   hex_gdf[['hex_id','geometry']].reset_index(drop=True),
                   how='left', predicate='within')
joined['total_visit_2025'] = pd.to_numeric(
    joined['total_visit_2025'], errors='coerce').fillna(0)
hv = joined.groupby('hex_id')['total_visit_2025'].sum().reset_index()
hv.columns = ['hex_id', 'total_visits']
hv = hv[hv['total_visits'] > 0]

J_df  = hv[hv['hex_id'].isin(hex_ids)].copy().reset_index(drop=True)
J_df  = J_df.merge(hexes[['hex_id','lon','lat']], on='hex_id', how='left')
J_ids = J_df['hex_id'].tolist()
print(f'  |J| = {len(J_ids)} candidate stops')
tock(t0, 'Build candidate set J')

# ══════════════════════════════════════════════════════════════════════════════
# 3. Build n_ig per hex (spatially disaggregated, with fallback)
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Build n_ig')
eld_E = ['B01001_020E','B01001_021E','B01001_022E','B01001_023E',
         'B01001_024E','B01001_025E','B01001_044E','B01001_045E',
         'B01001_046E','B01001_047E','B01001_048E','B01001_049E']

headers_pop = acs_pop[0]
COUNTY_TOTAL = COUNTY_ELDERLY = 0
for rec in acs_pop[1:]:
    d = dict(zip(headers_pop, rec))
    COUNTY_TOTAL   += float(d.get('B01003_001E', 0) or 0)
    COUNTY_ELDERLY += sum(float(d.get(c, 0) or 0) for c in eld_E)
COUNTY_TOTAL   = max(COUNTY_TOTAL, 91585)
COUNTY_ELDERLY = max(COUNTY_ELDERLY, 19008)

headers_veh = acs_veh[0]
COUNTY_ZV  = sum(float(dict(zip(headers_veh,r)).get('B08201_002E',0) or 0)
                 for r in acs_veh[1:])
COUNTY_HH  = max(sum(float(dict(zip(headers_veh,r)).get('B08201_001E',0) or 0)
                     for r in acs_veh[1:]), 1)
COUNTY_ZV_SHARE = COUNTY_ZV / COUNTY_HH

hex_eld_est = {}; hex_total_pop = {}; hex_zv_share_map = {}

if os.path.exists(HEX_ELDERLY_SE):
    _df = pd.read_csv(HEX_ELDERLY_SE)
    if 'elderly_est' in _df.columns:
        _p = _df[_df['elderly_est'] > 0]
        if len(_p):
            hex_eld_est = dict(zip(_p['hex_id'], _p['elderly_est']))
            print(f'  Loaded hex elderly_est: {len(hex_eld_est)} hexes')
        else:
            print('  WARNING: elderly_est all zero -- using county average fallback')
    if 'total_pop' in _df.columns:
        _p = _df[_df['total_pop'] > 0]
        if len(_p):
            hex_total_pop = dict(zip(_p['hex_id'], _p['total_pop']))
            print(f'  Loaded hex total_pop: {len(hex_total_pop)} hexes')
        else:
            print('  WARNING: total_pop all zero -- using county average fallback')
    else:
        print('  WARNING: total_pop column missing -- using county average fallback')
else:
    print('  WARNING: HEX_ELDERLY_SE not found -- using county average fallback')

if os.path.exists(HEX_ZVEH_SHARE):
    _df = pd.read_csv(HEX_ZVEH_SHARE)
    if 'zv_share' in _df.columns:
        hex_zv_share_map = dict(zip(_df['hex_id'], _df['zv_share']))
        print(f'  Loaded hex zv_share: {len(hex_zv_share_map)} hexes')

n_per_hex = COUNTY_TOTAL / N_HEX
n_ig = {}
n_fallback = 0
for hid in hex_ids:
    n_h  = hex_total_pop.get(hid, n_per_hex)
    e_h  = min(hex_eld_est.get(hid, COUNTY_ELDERLY/N_HEX), n_h)
    zv_h = hex_zv_share_map.get(hid, COUNTY_ZV_SHARE)
    nn_h = max(n_h - e_h, 0)
    n_ig[hid] = {
        'ec':   e_h  * (1 - zv_h),
        'enc':  e_h  * zv_h,
        'nec':  nn_h * (1 - zv_h),
        'nenc': nn_h * zv_h,
    }
    if hid not in hex_total_pop:
        n_fallback += 1

n_total = {hid: sum(n_ig[hid].values()) for hid in hex_ids}
print(f'  Total pop: {sum(n_total.values()):.0f}  '
      f'Uniform fallback: {n_fallback}/{N_HEX} hexes')
if n_fallback == N_HEX:
    print('  WARNING: ALL hexes using uniform fallback -- run compute_hex_se_from_vrt.py first')
tock(t0, 'Build n_ig')

# ══════════════════════════════════════════════════════════════════════════════
# 4. Build travel time matrices (raw minutes, not impedance)
#    T_car[i][j]     = car travel time (min)
#    T_transit[i][j] = transit travel time (min)
#    T_mc[i][j]      = microtransit travel time = t_car + W_MEAN (min)
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Build travel time matrices')

# Car
car_sub = tt_car[tt_car['to_hex'].isin(J_ids)].copy()
T_car = {}
for row in car_sub.itertuples(index=False):
    t = float(row.travel_time_min) if pd.notna(row.travel_time_min) else BIG_M
    T_car.setdefault(row.from_hex, {})[row.to_hex] = min(t, BIG_M)
# Fill missing with BIG_M
for i in hex_ids:
    T_car.setdefault(i, {})
    for j in J_ids:
        T_car[i].setdefault(j, BIG_M)
print(f'  Car OD pairs: {len(car_sub):,}')

# Transit
T_transit = {}
if os.path.exists(TT_TRANSIT):
    tr_sub = pd.read_csv(TT_TRANSIT)
    tr_sub = tr_sub[tr_sub['to_hex'].isin(J_ids)].copy()
    for row in tr_sub.itertuples(index=False):
        t = float(row.travel_time_min) if pd.notna(row.travel_time_min) else BIG_M
        T_transit.setdefault(row.from_hex, {})[row.to_hex] = min(t, BIG_M)
    for i in hex_ids:
        T_transit.setdefault(i, {})
        for j in J_ids:
            T_transit[i].setdefault(j, BIG_M)
    print(f'  Transit OD pairs: {len(tr_sub):,}')
else:
    print(f'  Transit OD file not found -- transit modes will be skipped')
    for i in hex_ids:
        T_transit[i] = {j: BIG_M for j in J_ids}

# Microtransit (car leg + mean Weibull waiting time)
T_mc = {}
for i in hex_ids:
    T_mc[i] = {}
    for j in J_ids:
        t_c = T_car[i].get(j, BIG_M)
        T_mc[i][j] = min(t_c + W_MEAN, BIG_M) if t_c < BIG_M else BIG_M

tock(t0, 'Build travel time matrices')

# ══════════════════════════════════════════════════════════════════════════════
# 5. Mode definitions for travel time
#    Each mode maps each segment to the travel time matrix it uses.
#    Segments with no access under a mode get BIG_M (excluded from constraints).
# ══════════════════════════════════════════════════════════════════════════════
# T_none: no-access placeholder (all BIG_M)
T_none = {i: {j: BIG_M for j in J_ids} for i in hex_ids}

MODES_TT = {
    'car_only': {
        'ec': T_car, 'enc': T_none, 'nec': T_car, 'nenc': T_none,
    },
    'car_for_all': {
        'ec': T_car, 'enc': T_car, 'nec': T_car, 'nenc': T_car,
    },
    'car_microtrans': {
        'ec': T_car, 'enc': T_mc,  'nec': T_car, 'nenc': T_mc,
    },
    'transit_only': {
        'ec': T_none, 'enc': T_transit, 'nec': T_none, 'nenc': T_transit,
    },
    'transit_for_all': {
        'ec': T_transit, 'enc': T_transit, 'nec': T_transit, 'nenc': T_transit,
    },
    'car_transit': {
        'ec': T_car, 'enc': T_transit, 'nec': T_car, 'nenc': T_transit,
    },
}

# Skip transit modes if transit OD not loaded
if not os.path.exists(TT_TRANSIT):
    for mode in ['transit_only', 'transit_for_all', 'car_transit']:
        del MODES_TT[mode]
    print('  Transit modes excluded (OD file missing)')

print(f'  Active modes: {list(MODES_TT.keys())}')

# ══════════════════════════════════════════════════════════════════════════════
# 6. Identify Mayodan candidate stops
# ══════════════════════════════════════════════════════════════════════════════
from math import radians, cos, sin, asin, sqrt as msqrt

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lo1,la1,lo2,la2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((la2-la1)/2)**2 + cos(la1)*cos(la2)*sin((lo2-lo1)/2)**2
    return R * 2 * asin(msqrt(a))

J_df['dist_mayodan_km'] = J_df.apply(
    lambda r: haversine_km(r['lon'], r['lat'], MAYODAN_LON, MAYODAN_LAT), axis=1)
J_df['is_mayodan'] = J_df['dist_mayodan_km'] <= 5.0
may_hex_set = set(J_df[J_df['is_mayodan']]['hex_id'].tolist())
print(f'\n  Mayodan candidates: {len(may_hex_set)} stops within 5 km')

# ══════════════════════════════════════════════════════════════════════════════
# 7. Model 1 — Population-weighted travel time minimization
#    For each mode, the effective travel time for hex i is:
#      t_i = weighted average across segments of their mode-specific travel time
#      teff_i = Σ_g (n_ig / n_i) · T_mode[g][i][nearest_j]
#    This gives a single effective travel time per hex per mode,
#    consistent with how each segment accesses the system.
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Model 1: Pop-weighted travel time minimization (all modes, P=1..50)')

m1_results = []

for mode_name, T_by_seg in MODES_TT.items():
    print(f'\n  Mode: {MODE_LABELS[mode_name]}')

    # Compute effective travel time per hex: population-weighted across segments
    # For hexes where a segment has no access (T=BIG_M), those segment-hexes
    # are excluded from the population weight for that hex.
    T_eff = {}
    for i in hex_ids:
        n_i = n_total[i]
        if n_i <= 0:
            T_eff[i] = {j: BIG_M for j in J_ids}
            continue
        T_eff[i] = {}
        for j in J_ids:
            # Weight each segment's travel time by its population share
            num = sum(n_ig[i][g] * T_by_seg[g][i][j] for g in PI
                      if T_by_seg[g][i][j] < BIG_M)
            den = sum(n_ig[i][g] for g in PI
                      if T_by_seg[g][i][j] < BIG_M)
            T_eff[i][j] = num / den if den > 0 else BIG_M

    # Identify hexes that can be reached (at least one j with T_eff < BIG_M)
    reachable = [i for i in hex_ids
                 if any(T_eff[i][j] < BIG_M for j in J_ids)
                 and n_total[i] > 0]
    print(f'    Reachable hexes: {len(reachable)}/{N_HEX}')

    for P in P_LIST:
        t_solve = time.time()
        prob = LpProblem(f'M1_{mode_name}_P{P}', LpMinimize)
        y    = {j: LpVariable(f'y_{j}', cat=LpBinary) for j in J_ids}
        d    = {i: LpVariable(f'd_{i}', lowBound=0)   for i in reachable}

        # Objective: minimize population-weighted mean travel time
        prob += lpSum(n_total[i] * d[i] for i in reachable)

        # Cardinality
        prob += lpSum(y[j] for j in J_ids) == P

        # Assignment: d_i <= t_ij + BIG_M*(1-y_j)
        # Only add for reachable pairs (T_eff < BIG_M) to keep problem tight
        for i in reachable:
            reachable_j = [j for j in J_ids if T_eff[i][j] < BIG_M]
            if reachable_j:
                for j in reachable_j:
                    prob += d[i] <= T_eff[i][j] + BIG_M * (1 - y[j])

        prob.solve(PULP_CBC_CMD(msg=0))

        sel = [j for j in J_ids
               if value(y[j]) is not None and value(y[j]) > 0.5]

        # Compute pw_mean_tt and T_max from solution directly
        if sel:
            tt_each = {}
            for i in reachable:
                reachable_j_sel = [j for j in sel if T_eff[i][j] < BIG_M]
                tt_each[i] = min((T_eff[i][j] for j in reachable_j_sel),
                                 default=BIG_M)
            pw_mean_tt = (sum(n_total[i] * tt_each[i] for i in reachable) /
                          max(sum(n_total[i] for i in reachable), 1))
            T_max_val  = max(tt_each.values()) if tt_each else BIG_M
        else:
            pw_mean_tt = BIG_M
            T_max_val  = BIG_M

        has_may = any(j in may_hex_set for j in sel)
        may_nos = [int(J_df[J_df['hex_id']==j]['index'].values[0]+1)
                   if 'index' in J_df.columns
                   else j for j in sel if j in may_hex_set]

        m1_results.append({
            'mode':        mode_name,
            'P':           P,
            'pw_mean_tt':  pw_mean_tt,
            'T_max':       T_max_val,
            'has_mayodan': has_may,
            'selected_stops': '|'.join(sel),
            'solve_s':     time.time() - t_solve,
        })
        flag = '  [MAYODAN]' if has_may else ''
        print(f'    P={P:>3}: pw_mean={pw_mean_tt:.2f} min  '
              f'T_max={T_max_val:.1f} min  ({time.time()-t_solve:.1f}s){flag}')

tock(t0, 'Model 1')

# Save Model 1 results
m1_df = pd.DataFrame(m1_results)
m1_df.to_csv(f'{OUTPUTS}/dual_model1_results.csv', index=False)
print(f'\n  Saved: dual_model1_results.csv')

# ══════════════════════════════════════════════════════════════════════════════
# 8. Model 2 — True single-BLP min-max fairness (Dr. Pandey correction)
#
#    For each mode and P, ONE BLP minimizes the worst-case travel time
#    across ALL population segments and ALL reachable hexes simultaneously:
#
#      minimize  T_max
#      s.t.      T_max >= t_ij^{m,g} - BIG_M*(1-y_j)
#                         ∀ reachable (i,g,j): n_ig > POP_MIN, t_ij < BIG_M
#                Σ_j y_j = P
#                y_j ∈ {0,1},  T_max >= 0
#
#    This is the correct formulation per Dr. Pandey:
#    - A SINGLE shared T_max variable covers all groups simultaneously
#    - The optimizer must find P stops that minimise the worst-case
#      travel time for ANYONE (any segment, any hex) in the county
#    - NOT 4 separate BLPs (one per group) -- that was the old incorrect version
#
#    After solving, we report per-group T_max and mean_tt by evaluating
#    the SAME selected stop set against each group's travel times.
#    This shows which group faces the tightest constraint at the solution.
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Model 2: True single-BLP min-max across ALL groups (all modes, P=1..50)')

m2_results = []

for mode_name, T_by_seg in MODES_TT.items():
    print(f'\n  Mode: {MODE_LABELS[mode_name]}')

    # Build reachable (i, g, j) triples for this mode
    # Covers ALL groups simultaneously -- this is the key fix
    reachable_igj = [
        (i, g, j)
        for i in hex_ids
        for g in PI
        if n_ig[i][g] > POP_MIN
        for j in J_ids
        if T_by_seg[g][i][j] < BIG_M
    ]
    print(f'    Reachable (i,g,j) triples: {len(reachable_igj):,}  '
          f'(all groups combined)')

    if not reachable_igj:
        print(f'    Skipped -- no reachable triples for this mode')
        for P in P_LIST:
            for g in PI:
                m2_results.append({
                    'mode': mode_name, 'group': g, 'P': P,
                    'T_max': BIG_M, 'mean_tt': BIG_M,
                    'has_mayodan': False, 'selected_stops': '',
                    'solve_s': 0.0,
                })
        continue

    for P in P_LIST:
        t_solve = time.time()
        prob  = LpProblem(f'M2_{mode_name}_P{P}', LpMinimize)
        y     = {j: LpVariable(f'y_{j}', cat=LpBinary) for j in J_ids}
        T_max = LpVariable('T_max', lowBound=0)

        # Objective: minimize worst-case travel time across ALL groups and hexes
        prob += T_max

        # Cardinality
        prob += lpSum(y[j] for j in J_ids) == P

        # Single set of constraints covering all (i, g, j) simultaneously
        # T_max >= t_ij^{m,g} - BIG_M*(1-y_j)  for all reachable (i,g,j)
        for i, g, j in reachable_igj:
            prob += T_max >= T_by_seg[g][i][j] - BIG_M * (1 - y[j])

        prob.solve(PULP_CBC_CMD(msg=0))

        sel = [j for j in J_ids
               if value(y[j]) is not None and value(y[j]) > 0.5]

        # Evaluate the SAME stop set against each group's travel times
        # to report per-group T_max and mean_tt
        if sel:
            # Global T_max from solution (worst case across all groups)
            T_max_global = value(T_max) if value(T_max) is not None else BIG_M

            for g in PI:
                T_mode_g = T_by_seg[g]
                active_i_g = [i for i in hex_ids
                               if n_ig[i][g] > POP_MIN
                               and any(T_mode_g[i][j] < BIG_M for j in J_ids)]
                tt_by_hex_g = {}
                for i in active_i_g:
                    reach_sel = [j for j in sel if T_mode_g[i][j] < BIG_M]
                    tt_by_hex_g[i] = (min(T_mode_g[i][j] for j in reach_sel)
                                      if reach_sel else BIG_M)

                # Per-group T_max: worst case for this group under the joint solution
                T_max_g = (max(v for v in tt_by_hex_g.values() if v < BIG_M)
                           if any(v < BIG_M for v in tt_by_hex_g.values())
                           else BIG_M)

                # Per-group mean_tt
                mean_tt_g = (
                    sum(n_ig[i][g] * tt_by_hex_g[i]
                        for i in active_i_g if tt_by_hex_g.get(i, BIG_M) < BIG_M) /
                    max(sum(n_ig[i][g] for i in active_i_g
                            if tt_by_hex_g.get(i, BIG_M) < BIG_M), 1)
                ) if active_i_g else BIG_M

                has_may = any(j in may_hex_set for j in sel)
                m2_results.append({
                    'mode':           mode_name,
                    'group':          g,
                    'P':              P,
                    'T_max_global':   T_max_global,   # worst case across all groups
                    'T_max':          T_max_g,        # worst case for this group
                    'mean_tt':        mean_tt_g,
                    'has_mayodan':    has_may,
                    'selected_stops': '|'.join(sel),
                    'solve_s':        time.time() - t_solve,
                })
        else:
            for g in PI:
                m2_results.append({
                    'mode': mode_name, 'group': g, 'P': P,
                    'T_max_global': BIG_M, 'T_max': BIG_M, 'mean_tt': BIG_M,
                    'has_mayodan': False, 'selected_stops': '',
                    'solve_s': time.time() - t_solve,
                })

        # Print summary: global T_max + per-group breakdown
        flag = '  [MAYODAN]' if sel and any(j in may_hex_set for j in sel) else ''
        T_global = value(T_max) if value(T_max) is not None else BIG_M
        print(f'    P={P:>3}: T_max_global={T_global:.1f} min  '
              f'({time.time()-t_solve:.1f}s){flag}')

tock(t0, 'Model 2')

# Save Model 2 results
m2_df = pd.DataFrame(m2_results)
m2_df.to_csv(f'{OUTPUTS}/dual_model2_results.csv', index=False)
print(f'\n  Saved: dual_model2_results.csv')

# ══════════════════════════════════════════════════════════════════════════════
# 9. Mayodan entry summary
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('MAYODAN ENTRY POINTS')
print('='*65)

entry_rows = []

# Model 1
for mode_name in MODES_TT:
    sub = m1_df[m1_df['mode']==mode_name].sort_values('P')
    first = sub[sub['has_mayodan']].head(1)
    if len(first):
        fp = int(first['P'].values[0])
        print(f'  M1 {mode_name:<18}: P={fp}')
        entry_rows.append({'model':'M1','mode':mode_name,'group':'all','first_P':fp})
    else:
        print(f'  M1 {mode_name:<18}: not selected (P<=50)')
        entry_rows.append({'model':'M1','mode':mode_name,'group':'all','first_P':None})

# Model 2
for mode_name in MODES_TT:
    for g in PI:
        sub = m2_df[(m2_df['mode']==mode_name) & (m2_df['group']==g)].sort_values('P')
        first = sub[sub['has_mayodan']].head(1)
        if len(first):
            fp = int(first['P'].values[0])
            print(f'  M2 {mode_name:<18} [{g}]: P={fp}')
            entry_rows.append({'model':'M2','mode':mode_name,'group':g,'first_P':fp})
        else:
            print(f'  M2 {mode_name:<18} [{g}]: not selected (P<=50)')
            entry_rows.append({'model':'M2','mode':mode_name,'group':g,'first_P':None})

pd.DataFrame(entry_rows).to_csv(f'{OUTPUTS}/dual_mayodan_entry.csv', index=False)
print(f'\n  Saved: dual_mayodan_entry.csv')

# ══════════════════════════════════════════════════════════════════════════════
# 10. Price of Fairness
#     PoF(P, mode, g) = [A_tilde_primal(P, mode) - A_tilde_fair(P, mode, g)]
#                       / A_tilde_primal(P, mode)
#
#     A_tilde_primal: from primal_accessibility_results.csv
#     A_tilde_fair:   primal accessibility evaluated at M2-selected stop set
#                     (uses primal gravity objective, not travel time)
#
#     To evaluate primal at M2 stops we need the primal's gravity-based A_max.
#     We load it from the primal CSV (A_max column is constant per mode).
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Price of Fairness')

pof_results = []

if os.path.exists(PRIMAL_RESULTS):
    primal_df = pd.read_csv(PRIMAL_RESULTS)
    print(f'  Loaded primal results: {len(primal_df)} rows')

    # Build impedance matrices needed for primal objective evaluation
    # (same as primal_accessibility.py -- gravity weights not travel times)
    def imp(t, a, b):
        t = np.asarray(t, dtype=float)
        w = np.zeros_like(t)
        m = np.isfinite(t) & (t > 0)
        w[m] = np.exp(a * (t[m] ** b))
        return w

    car_sub2 = tt_car[tt_car['to_hex'].isin(J_ids)].copy()
    car_sub2['w'] = imp(car_sub2['travel_time_min'].values, ALPHA_CAR, BETA_CAR)
    W_car_grav = {}
    for row in car_sub2.itertuples(index=False):
        W_car_grav.setdefault(row.from_hex, {})[row.to_hex] = row.w

    W_transit_grav = {}
    if os.path.exists(TT_TRANSIT):
        tr_sub2 = pd.read_csv(TT_TRANSIT)
        tr_sub2 = tr_sub2[tr_sub2['to_hex'].isin(J_ids)].copy()
        tr_sub2['w'] = imp(tr_sub2['travel_time_min'].values, ALPHA_TR, BETA_TR)
        for row in tr_sub2.itertuples(index=False):
            W_transit_grav.setdefault(row.from_hex, {})[row.to_hex] = row.w

    W_mc_grav = {}
    for i in hex_ids:
        W_mc_grav[i] = {}
        for j in J_ids:
            t_c = T_car[i].get(j, BIG_M)
            if t_c < BIG_M:
                t_mt = t_c + W_MEAN
                W_mc_grav[i][j] = np.exp(ALPHA_MC * (t_mt ** BETA_MC))

    W_none_grav = {}

    MODES_GRAV = {
        'car_only':        {'ec': W_car_grav, 'enc': W_none_grav,
                            'nec': W_car_grav, 'nenc': W_none_grav},
        'car_for_all':     {'ec': W_car_grav, 'enc': W_car_grav,
                            'nec': W_car_grav, 'nenc': W_car_grav},
        'car_microtrans':  {'ec': W_car_grav, 'enc': W_mc_grav,
                            'nec': W_car_grav, 'nenc': W_mc_grav},
        'transit_only':    {'ec': W_none_grav,    'enc': W_transit_grav,
                            'nec': W_none_grav,    'nenc': W_transit_grav},
        'transit_for_all': {'ec': W_transit_grav, 'enc': W_transit_grav,
                            'nec': W_transit_grav, 'nenc': W_transit_grav},
        'car_transit':     {'ec': W_car_grav, 'enc': W_transit_grav,
                            'nec': W_car_grav, 'nenc': W_transit_grav},
    }

    # K=15 nearest per hex for allocation-based A_max evaluation
    K_SPARSE = 15
    nearest_j = {}
    for i in hex_ids:
        scored = [(j, W_car_grav.get(i,{}).get(j,0)) for j in J_ids]
        scored.sort(key=lambda x: -x[1])
        nearest_j[i] = [j for j,_ in scored[:K_SPARSE]]

    def eval_primal_at_stops(sel_hexes, mode_name):
        """Evaluate allocation-based primal accessibility at a given stop set."""
        if mode_name not in MODES_GRAV:
            return 0.0
        W_by_seg = MODES_GRAV[mode_name]
        sel_set  = set(sel_hexes)
        obj = 0.0
        for i in hex_ids:
            for g in PI:
                if n_ig[i][g] <= 0: continue
                avail = [j for j in nearest_j[i] if j in sel_set]
                if not avail: continue
                Wg = W_by_seg[g]
                best_w = max(Wg.get(i,{}).get(j,0) for j in avail)
                obj += PI[g] * n_ig[i][g] * best_w
        return obj

    for mode_name in MODES_TT:
        # A_max for this mode from primal results
        primal_mode = primal_df[primal_df['mode']==mode_name]
        if len(primal_mode) == 0:
            print(f'  No primal results for mode {mode_name} -- skipping PoF')
            continue
        A_max_mode = primal_mode['A_max'].iloc[0]

        # A_tilde_primal per P
        A_opt = dict(zip(primal_mode['P'], primal_mode['A_tilde']))

        for g in PI:
            m2_sub = m2_df[(m2_df['mode']==mode_name) & (m2_df['group']==g)]
            for _, row in m2_sub.iterrows():
                P = int(row['P'])
                sel = [s for s in row['selected_stops'].split('|') if s] \
                      if row['selected_stops'] else []
                at_fair  = eval_primal_at_stops(sel, mode_name) / A_max_mode \
                           if A_max_mode > 0 and sel else 0.0
                at_opt   = A_opt.get(P, 0.0)
                pof      = (at_opt - at_fair) / at_opt if at_opt > 0 else 0.0
                pof_results.append({
                    'mode':             mode_name,
                    'group':            g,
                    'P':                P,
                    'A_tilde_primal':   at_opt,
                    'A_tilde_fair':     at_fair,
                    'PoF':              pof,
                    'PoF_pct':          pof * 100,
                    'has_mayodan':      row['has_mayodan'],
                })

    pof_df = pd.DataFrame(pof_results)
    pof_df.to_csv(f'{OUTPUTS}/price_of_fairness.csv', index=False)
    print(f'\n  Saved: price_of_fairness.csv')

    # Summary at key P values
    print('\n  PoF summary (car_only mode):')
    for p_val in [4, 12, 26, 32, 50]:
        sub = pof_df[(pof_df['mode']=='car_only') & (pof_df['P']==p_val)]
        if len(sub):
            at_opt = sub['A_tilde_primal'].iloc[0]
            print(f'  P={p_val}  A_opt={at_opt:.4f}')
            for _, r in sub.iterrows():
                may = '[Mayodan in]' if r['has_mayodan'] else ''
                print(f'    {r["group"]:4s}: A_fair={r["A_tilde_fair"]:.4f}  '
                      f'PoF={r["PoF_pct"]:.1f}%  {may}')
else:
    print(f'  primal_accessibility_results.csv not found -- skipping PoF')
    print(f'  Run primal_accessibility.py first, then rerun this script')

tock(t0, 'Price of Fairness')

# ══════════════════════════════════════════════════════════════════════════════
# 11. Plots
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Generating plots')

# ── Plot 1: Model 1 step plot — pw_mean_tt vs P, all modes ───────────────────
fig, ax = plt.subplots(figsize=(14, 7))
for mode_name in MODES_TT:
    sub = m1_df[m1_df['mode']==mode_name].sort_values('P')
    color = MODE_COLORS.get(mode_name, '#888')
    label = MODE_LABELS.get(mode_name, mode_name)
    ax.step(sub['P'], sub['pw_mean_tt'], where='post',
            color=color, lw=2.2, label=label)
    ax.text(50.5, sub['pw_mean_tt'].iloc[-1],
            f'{sub["pw_mean_tt"].iloc[-1]:.1f}',
            fontsize=7.5, va='center', color=color)

ax.set_xlabel('Number of clinic stops (P)', fontsize=12)
ax.set_ylabel('Population-weighted mean travel time (min)', fontsize=12)
ax.set_title('Model 1: Population-weighted Travel Time vs P\n'
             'Rockingham County, NC  |  Dual Accessibility',
             fontsize=11, pad=6)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(1, 53)
plt.tight_layout()
plt.savefig(f'{OUTPUTS}/dual_model1_step_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: dual_model1_step_plot.png')

# ── Plot 2: Model 2 — T_max vs P, 2×3 grid (one panel per mode) ──────────────
n_modes = len(MODES_TT)
ncols = min(3, n_modes)
nrows = math.ceil(n_modes / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
axes_flat = axes.flatten() if n_modes > 1 else [axes]

for idx, mode_name in enumerate(MODES_TT):
    ax = axes_flat[idx]
    # Plot T_max_global (joint worst case) once -- same for all groups at each P
    # Then overlay per-group T_max to show which group drives the constraint
    sub_any = m2_df[(m2_df['mode']==mode_name) &
                    (m2_df['group']=='ec')].sort_values('P')
    if len(sub_any) and 'T_max_global' in sub_any.columns:
        ax.step(sub_any['P'], sub_any['T_max_global'], where='post',
                color='black', lw=2.5, ls='--', label='T_max (joint, all groups)',
                zorder=5)

    for g in PI:
        sub = m2_df[(m2_df['mode']==mode_name) & (m2_df['group']==g)].sort_values('P')
        if len(sub) == 0: continue
        # Per-group T_max under joint solution
        ax.step(sub['P'], sub['T_max'], where='post',
                color=GROUP_COLORS[g], lw=1.5, ls=':', alpha=0.8,
                label=f'{GROUP_LABELS[g]} (per-group)')
        # Mark Mayodan entry
        may_entry = sub[sub['has_mayodan']].head(1)
        if len(may_entry):
            fp = int(may_entry['P'].values[0])
            tv = float(may_entry['T_max_global'].values[0]) \
                 if 'T_max_global' in sub.columns else float(may_entry['T_max'].values[0])
            ax.scatter([fp], [tv], s=60, color=GROUP_COLORS[g],
                       marker='D', zorder=7)

    ax.set_xlabel('P', fontsize=10)
    ax.set_ylabel('T_max (min)', fontsize=10)
    ax.set_title(f'{MODE_LABELS.get(mode_name, mode_name)}\n'
                 f'Diamond = Mayodan first selected', fontsize=9, pad=4)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, 51)

# Hide unused subplots
for idx in range(len(MODES_TT), len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle('Model 2: Min-Max Fairness — Worst-Case Travel Time vs P\n'
             'Rockingham County, NC  |  Single BLP across all hexes per group',
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUTS}/dual_model2_step_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: dual_model2_step_plot.png')

# ── Plot 3: Price of Fairness — one figure per mode (PoF curves + bar chart) ──
if pof_results:
    for mode_name in MODES_TT:
        pof_mode = pof_df[pof_df['mode']==mode_name]
        if len(pof_mode) == 0:
            continue

        fig, axes_pof = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f'Price of Fairness — {MODE_LABELS.get(mode_name, mode_name)}\n'
            'Rockingham County, NC  |  '
            r'PoF(P,g) = $\frac{A_{primal}(P) - A_{fair}(P,g)}{A_{primal}(P)}$',
            fontsize=11, y=1.02)

        # Left: PoF curves vs P, one line per group
        ax = axes_pof[0]
        for g in PI:
            sub = pof_mode[pof_mode['group']==g].sort_values('P')
            if len(sub) == 0:
                continue
            ax.plot(sub['P'], sub['PoF_pct'], color=GROUP_COLORS[g],
                    lw=2, marker='o', markersize=2.5, label=GROUP_LABELS[g])
            # Mark Mayodan entry with diamond
            may_rows = sub[sub['has_mayodan']]
            if len(may_rows):
                gP     = int(may_rows['P'].min())
                pof_at = float(sub[sub['P']==gP]['PoF_pct'].values[0])
                ax.scatter([gP], [pof_at], s=80, color=GROUP_COLORS[g],
                           marker='D', zorder=7)
                ax.annotate(f'Mayodan\nP={gP}\n{pof_at:.1f}%',
                            xy=(gP, pof_at),
                            xytext=(gP+1.5, pof_at+1.5),
                            fontsize=6.5, color=GROUP_COLORS[g],
                            arrowprops=dict(arrowstyle='->', color=GROUP_COLORS[g],
                                            lw=0.8))

        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlabel('Number of clinic stops (P)', fontsize=11)
        ax.set_ylabel('Price of Fairness (%)', fontsize=11)
        ax.set_title('PoF vs P  |  Diamond = Mayodan first selected',
                     fontsize=10, pad=5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(1, 51)
        # Annotate final P=50 values
        for g in PI:
            sub = pof_mode[pof_mode['group']==g].sort_values('P')
            if len(sub):
                ax.text(50.5, sub['PoF_pct'].iloc[-1],
                        f'{sub["PoF_pct"].iloc[-1]:.1f}%',
                        fontsize=7, va='center', color=GROUP_COLORS[g])

        # Right: PoF at key P values as grouped bar chart
        ax2 = axes_pof[1]
        key_p = [4, 12, 26, 32, 50]
        x     = np.arange(len(key_p))
        width = 0.18
        for ki, g in enumerate(PI):
            vals = []
            for p in key_p:
                match = pof_mode[(pof_mode['P']==p) & (pof_mode['group']==g)]
                vals.append(float(match['PoF_pct'].values[0]) if len(match) else 0.0)
            bars = ax2.bar(x + ki*width, vals, width, color=GROUP_COLORS[g],
                           label=GROUP_LABELS[g], edgecolor='white', linewidth=0.3)
            for bar, v in zip(bars, vals):
                if v > 0.5:
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.2,
                             f'{v:.1f}%', ha='center', fontsize=6,
                             color=GROUP_COLORS[g])

        ax2.set_xticks(x + width*1.5)
        ax2.set_xticklabels([f'P={p}' for p in key_p], fontsize=10)
        ax2.set_ylabel('Price of Fairness (%)', fontsize=11)
        ax2.set_title('PoF at key P values\n(lower = cheaper to be fair)',
                      fontsize=10, pad=5)
        ax2.legend(fontsize=7.5)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        out_pof = f'{OUTPUTS}/dual_pof_{mode_name}.png'
        plt.savefig(out_pof, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: dual_pof_{mode_name}.png')

    # ── Plot 3b: PoF comparison across modes (car_only vs car_for_all, P=4) ──
    # Single figure: PoF at P=4 and P=26 across all modes, stacked bars per group
    fig, axes_pof_cmp = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Price of Fairness — Mode Comparison\n'
                 'Rockingham County, NC  |  How PoF varies by transport mode',
                 fontsize=11, y=1.02)

    for panel_idx, p_val in enumerate([4, 26]):
        ax = axes_pof_cmp[panel_idx]
        mode_list = list(MODES_TT.keys())
        x = np.arange(len(mode_list))
        width = 0.18
        for ki, g in enumerate(PI):
            vals = []
            for mode_name in mode_list:
                match = pof_df[(pof_df['mode']==mode_name) &
                               (pof_df['group']==g) &
                               (pof_df['P']==p_val)]
                vals.append(float(match['PoF_pct'].values[0]) if len(match) else 0.0)
            ax.bar(x + ki*width, vals, width, color=GROUP_COLORS[g],
                   label=GROUP_LABELS[g], edgecolor='white', linewidth=0.3)

        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels([MODE_LABELS.get(m, m) for m in mode_list],
                           fontsize=7.5, rotation=20, ha='right')
        ax.set_ylabel('Price of Fairness (%)', fontsize=11)
        ax.set_title(f'PoF at P={p_val} across all modes', fontsize=10, pad=5)
        ax.legend(fontsize=7.5)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUTS}/dual_pof_mode_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: dual_pof_mode_comparison.png')

# ── Plot 4: Per-hex travel time heatmaps — ALL modes, P=4,12,26,50 ────────────
KEY_P_VIZ = [4, 12, 26, 50]
HEX_PLOT  = hex_gdf[['hex_id','geometry']].copy()

for mode_name in MODES_TT:
    m1_sub = m1_df[m1_df['mode']==mode_name]

    fig, axes_hm = plt.subplots(1, len(KEY_P_VIZ),
                                 figsize=(7*len(KEY_P_VIZ), 6))
    fig.suptitle(
        f'Per-Hex Min Travel Time (min) — {MODE_LABELS.get(mode_name, mode_name)}\n'
        f'Rockingham County, NC  |  Model 1 (pop-weighted) Selected Stops  '
        f'|  Color capped at 60 min',
        fontsize=11, y=1.01)

    T_mode = MODES_TT[mode_name]

    for col, p_val in enumerate(KEY_P_VIZ):
        ax = axes_hm[col]
        row_m1 = m1_sub[m1_sub['P']==p_val]
        if not len(row_m1):
            ax.set_visible(False)
            continue

        sel = [s for s in row_m1['selected_stops'].values[0].split('|') if s]

        # Per-hex min travel time to nearest selected stop
        # For each hex, take the population-weighted average across segments
        tt_hex = {}
        for i in hex_ids:
            tt_vals = []
            for g in PI:
                if n_ig[i][g] <= POP_MIN: continue
                reach_sel = [j for j in sel if T_mode[g][i].get(j, BIG_M) < BIG_M]
                if reach_sel:
                    tt_vals.append(min(T_mode[g][i][j] for j in reach_sel))
            tt_hex[i] = min(tt_vals) if tt_vals else BIG_M

        HEX_PLOT['tt']      = HEX_PLOT['hex_id'].map(tt_hex)
        HEX_PLOT['tt_plot'] = HEX_PLOT['tt'].clip(upper=60).fillna(60)

        HEX_PLOT.plot(column='tt_plot', ax=ax, cmap='RdYlGn_r',
                      vmin=0, vmax=60, edgecolor='white', linewidth=0.2)

        # Mark stops: Mayodan in red, others in black
        sel_coords = J_df[J_df['hex_id'].isin(sel)]
        may_sel    = sel_coords[sel_coords['hex_id'].isin(may_hex_set)]
        non_sel    = sel_coords[~sel_coords['hex_id'].isin(may_hex_set)]
        if len(non_sel):
            ax.scatter(non_sel['lon'], non_sel['lat'], s=80, c='black',
                       marker='*', zorder=7, label=f'Stop ({len(sel)} total)')
        if len(may_sel):
            ax.scatter(may_sel['lon'], may_sel['lat'], s=140, c='#e74c3c',
                       marker='*', zorder=8, label='Mayodan stop')

        # Stats annotation
        reachable_tt = [v for v in tt_hex.values() if v < BIG_M]
        pop_wt_mean  = float(row_m1['pw_mean_tt'].values[0])
        t_max_val    = float(row_m1['T_max'].values[0])
        ax.text(0.02, 0.02,
                f'pw_mean={pop_wt_mean:.1f} min\n'
                f'T_max={t_max_val:.0f} min\n'
                f'Unreachable: {tt_hex.values().__class__.__name__}\n'
                f'  {sum(1 for v in tt_hex.values() if v>=BIG_M)}/{N_HEX} hexes',
                transform=ax.transAxes, fontsize=7, va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        has_may = bool(row_m1['has_mayodan'].values[0])
        ax.set_title(f'P={p_val}  |  pw_mean={pop_wt_mean:.1f} min\n'
                     f'{"[Mayodan selected]" if has_may else ""}',
                     fontsize=9, pad=4)
        ax.legend(fontsize=6.5, loc='lower right')
        ax.set_axis_off()

        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=mcolors.Normalize(0, 60))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.6, label='Min tt to nearest stop (min)')

    plt.tight_layout()
    out_hm = f'{OUTPUTS}/dual_heatmap_{mode_name}.png'
    plt.savefig(out_hm, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: dual_heatmap_{mode_name}.png')

# ── Plot 5: Model 2 heatmaps — T_max geography at key P values ───────────────
# For each mode: 4 panels (one per group) at P=26 showing per-hex min tt
# under the fairness-optimal stop set
for mode_name in MODES_TT:
    m2_sub = m2_df[m2_df['mode']==mode_name]
    T_mode = MODES_TT[mode_name]

    fig, axes_m2hm = plt.subplots(1, len(PI),
                                   figsize=(7*len(PI), 6))
    fig.suptitle(
        f'Model 2 Min-Max Fairness — Per-Hex Travel Time at P=26\n'
        f'{MODE_LABELS.get(mode_name, mode_name)}  |  Rockingham County, NC',
        fontsize=11, y=1.01)

    for col, g in enumerate(PI):
        ax = axes_m2hm[col]
        row_m2 = m2_sub[(m2_sub['group']==g) & (m2_sub['P']==26)]
        if not len(row_m2):
            ax.set_visible(False)
            continue

        sel = [s for s in row_m2['selected_stops'].values[0].split('|') if s]

        tt_hex = {}
        for i in hex_ids:
            if n_ig[i][g] <= POP_MIN:
                tt_hex[i] = BIG_M
                continue
            reach_sel = [j for j in sel if T_mode[g][i].get(j, BIG_M) < BIG_M]
            tt_hex[i] = min(T_mode[g][i][j] for j in reach_sel) \
                        if reach_sel else BIG_M

        HEX_PLOT['tt']      = HEX_PLOT['hex_id'].map(tt_hex)
        HEX_PLOT['tt_plot'] = HEX_PLOT['tt'].clip(upper=60).fillna(60)

        HEX_PLOT.plot(column='tt_plot', ax=ax, cmap='RdYlGn_r',
                      vmin=0, vmax=60, edgecolor='white', linewidth=0.2)

        sel_coords = J_df[J_df['hex_id'].isin(sel)]
        may_sel    = sel_coords[sel_coords['hex_id'].isin(may_hex_set)]
        non_sel    = sel_coords[~sel_coords['hex_id'].isin(may_hex_set)]
        if len(non_sel):
            ax.scatter(non_sel['lon'], non_sel['lat'], s=80, c='black',
                       marker='*', zorder=7)
        if len(may_sel):
            ax.scatter(may_sel['lon'], may_sel['lat'], s=140, c='#e74c3c',
                       marker='*', zorder=8)

        tmax_v = float(row_m2['T_max'].values[0])
        has_may = bool(row_m2['has_mayodan'].values[0])
        ax.set_title(f'{GROUP_LABELS[g]}\nT_max={tmax_v:.0f} min  '
                     f'{"[Mayodan]" if has_may else ""}',
                     fontsize=9, pad=4)
        ax.set_axis_off()

        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=mcolors.Normalize(0, 60))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.6, label='Min tt (min, capped 60)')

    plt.tight_layout()
    out_m2hm = f'{OUTPUTS}/dual_m2_heatmap_{mode_name}_P26.png'
    plt.savefig(out_m2hm, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: dual_m2_heatmap_{mode_name}_P26.png')

# ── Plot 6: Primal vs Dual tradeoff — gravity accessibility vs travel time ────
# Shows what you give up in gravity-based coverage to gain travel time fairness
# X axis: pw_mean_tt (Model 1), Y axis: A_tilde_primal — one point per P per mode
if pof_results:
    fig, axes_td = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Primal-Dual Tradeoff: Gravity Accessibility vs Travel Time\n'
                 'Rockingham County, NC  |  Each point = one P value',
                 fontsize=11, y=1.02)

    # Left: A_tilde_primal vs pw_mean_tt (M1) — Pareto frontier per mode
    ax = axes_td[0]
    for mode_name in MODES_TT:
        m1_sub    = m1_df[m1_df['mode']==mode_name].sort_values('P')
        primal_sub = primal_df[primal_df['mode']==mode_name].sort_values('P') \
                    if os.path.exists(PRIMAL_RESULTS) else pd.DataFrame()
        if len(m1_sub) == 0 or len(primal_sub) == 0:
            continue
        color = MODE_COLORS.get(mode_name, '#888')
        label = MODE_LABELS.get(mode_name, mode_name)
        ax.scatter(m1_sub['pw_mean_tt'], primal_sub['A_tilde'],
                   c=color, s=15, alpha=0.7)
        ax.plot(m1_sub['pw_mean_tt'].values, primal_sub['A_tilde'].values,
                color=color, lw=1.5, label=label, alpha=0.8)
        # Annotate P=4, P=26, P=50
        for p_ann in [4, 26, 50]:
            r_m1 = m1_sub[m1_sub['P']==p_ann]
            r_pr = primal_sub[primal_sub['P']==p_ann]
            if len(r_m1) and len(r_pr):
                ax.annotate(f'P={p_ann}',
                            xy=(r_m1['pw_mean_tt'].values[0],
                                r_pr['A_tilde'].values[0]),
                            fontsize=6, color=color)

    ax.set_xlabel('Pop-weighted mean travel time (min) [Model 1]', fontsize=11)
    ax.set_ylabel(r'Gravity accessibility $\tilde{\mathcal{A}}$ [Primal]', fontsize=11)
    ax.set_title('Pareto frontier: more stops\n'
                 'reduces travel time AND increases accessibility',
                 fontsize=9, pad=5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: T_max (M2, car_only, worst group) vs A_tilde_primal
    ax2 = axes_td[1]
    if os.path.exists(PRIMAL_RESULTS):
        primal_car = primal_df[primal_df['mode']=='car_only'].sort_values('P')
        for g in PI:
            m2_sub = m2_df[(m2_df['mode']=='car_only') &
                           (m2_df['group']==g)].sort_values('P')
            if len(m2_sub) == 0: continue
            merged = m2_sub.merge(primal_car[['P','A_tilde']], on='P', how='inner')
            ax2.scatter(merged['T_max'], merged['A_tilde'],
                        c=GROUP_COLORS[g], s=15, alpha=0.7)
            ax2.plot(merged['T_max'].values, merged['A_tilde'].values,
                     color=GROUP_COLORS[g], lw=1.5, label=GROUP_LABELS[g],
                     alpha=0.8)
            for p_ann in [4, 26, 50]:
                rm = merged[merged['P']==p_ann]
                if len(rm):
                    ax2.annotate(f'P={p_ann}',
                                 xy=(rm['T_max'].values[0], rm['A_tilde'].values[0]),
                                 fontsize=6, color=GROUP_COLORS[g])

    ax2.set_xlabel('Worst-case travel time T_max (min) [Model 2, car only]',
                   fontsize=11)
    ax2.set_ylabel(r'Gravity accessibility $\tilde{\mathcal{A}}$ [Primal]', fontsize=11)
    ax2.set_title('Equity-efficiency tradeoff per group\n'
                  '(car only mode, lower-right = better)',
                  fontsize=9, pad=5)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUTS}/dual_primal_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: dual_primal_tradeoff.png')

tock(t0, 'Plots')

# ══════════════════════════════════════════════════════════════════════════════
# 12. Summary table
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('DUAL ACCESSIBILITY SUMMARY')
print('='*70)
print(f'\n  Model 1 — Population-weighted mean travel time (min)')
print(f'  {"Mode":<22}  {"P=4":>6}  {"P=12":>6}  {"P=26":>6}  {"P=50":>6}')
print(f'  {"-"*22}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}')
for mode_name in MODES_TT:
    sub = m1_df[m1_df['mode']==mode_name].set_index('P')
    vals = [sub.loc[p,'pw_mean_tt'] if p in sub.index else float('nan')
            for p in [4, 12, 26, 50]]
    print(f'  {MODE_LABELS.get(mode_name,mode_name):<22}  '
          + '  '.join(f'{v:>6.1f}' for v in vals))

print(f'\n  Model 2 — Global worst-case T_max (min) across all groups — car_only mode')
print(f'  {"Group":<25}  {"P=4":>6}  {"P=12":>6}  {"P=26":>6}  {"P=50":>6}')
print(f'  {"-"*25}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}')
for g in PI:
    sub = m2_df[(m2_df['mode']=='car_only') & (m2_df['group']==g)].set_index('P')
    # T_max_global is same for all groups at same P (joint solution)
    # T_max is per-group worst case under joint solution
    col = 'T_max_global' if 'T_max_global' in sub.columns else 'T_max'
    vals = [sub.loc[p, col] if p in sub.index else float('nan')
            for p in [4, 12, 26, 50]]
    print(f'  {GROUP_LABELS[g]:<25}  '
          + '  '.join(f'{v:>6.1f}' for v in vals))

print(f'\n  Model 2 — Per-group T_max (min) under joint solution — car_only mode')
print(f'  {"Group":<25}  {"P=4":>6}  {"P=12":>6}  {"P=26":>6}  {"P=50":>6}')
print(f'  {"-"*25}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}')
for g in PI:
    sub = m2_df[(m2_df['mode']=='car_only') & (m2_df['group']==g)].set_index('P')
    vals = [sub.loc[p,'T_max'] if p in sub.index else float('nan')
            for p in [4, 12, 26, 50]]
    print(f'  {GROUP_LABELS[g]:<25}  '
          + '  '.join(f'{v:>6.1f}' for v in vals))

print('\nDone.')