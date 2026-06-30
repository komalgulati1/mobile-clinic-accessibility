"""
primal_accessibility.py
=======================
Primal accessibility model — Rockingham County, NC
Komal Gulati, CR2C2 / NC A&T

Computes corrected gravity-based accessibility using allocation BLP:

    maximize  Σ_i Σ_g π_g * n_ig * Σ_j o_j * w_ij(g) * z_ij(g)
    s.t.      Σ_j z_ij(g) <= 1    ∀i,g   [each hex assigned to at most one stop]
              z_ij(g) <= y_j       ∀i,j,g [can only assign to open stop]
              Σ_j y_j = P
              y_j, z_ij ∈ {0,1}

Uses sparse z_ij (k=15 nearest per hex) for tractability.
Runs P=1..50, 5 transport modes, uniform o_j.

Key difference from original pipeline:
    Original: A(y) = Σ_j c_j * y_j  [greedy-equivalent, no stop interaction]
    This:     A(y) = Σ_i Σ_g π_g * n_ig * max_{j∈S}(o_j * w_ij)  [correct, submodular]

Outputs:
    outputs/primal_accessibility_results.csv
    outputs/primal_accessibility_step_plot.png
    outputs/primal_vs_linear_comparison.png
"""

import os, json, warnings, time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pulp import (LpProblem, LpMaximize, LpVariable, lpSum,
                  LpBinary, value, PULP_CBC_CMD)

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
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

CANDIDATE_NAICS = [813110, 611110, 624410, 621111,
                   922110, 922120, 922130, 922140, 922150, 922160, 922190]

# Impedance parameters
ALPHA_CAR = -0.020097;  BETA_CAR = 1.361630
ALPHA_TR  = -0.002062;  BETA_TR  = 1.608027
ALPHA_MC  = ALPHA_CAR;  BETA_MC  = BETA_CAR

import math

# ── Microtransit (RCATS) range parameters ─────────────────────────────────────
# t_ij^mt = delta * t_ij^car + W
# delta: detour ratio (Weibull-AFT, k=1.069, hu2025)
# W:     waiting time (Weibull, k=2.084, lambda=19.901 min, yang2025)
# Mean Weibull wait = lambda * Gamma(1 + 1/k) = 19.901 * Gamma(1.480) ≈ 17.67 min
# Three scenarios: optimistic / baseline / pessimistic
MC_SCENARIOS = {
    'car_microtrans_opt':  {'delta': 1.0, 'W': 3.74,  'label': 'Car + Microtransit (optimistic)',  'ls': ':',  'lw': 1.8},
    'car_microtrans':      {'delta': 1.3, 'W': 10.00, 'label': 'Car + Microtransit (baseline)',    'ls': '--', 'lw': 2.2},
    'car_microtrans_pess': {'delta': 2.0, 'W': 17.67, 'label': 'Car + Microtransit (pessimistic)', 'ls': ':',  'lw': 1.8},
}
# optimistic:  delta=1.0 (no shared detour), W=3.74 min (low wait)
# baseline:    delta=1.3 (empirical median detour, hu2025), W=10 min (moderate wait)
# pessimistic: delta=2.0 (upper bound detour, hu2025), W=17.67 min (mean Weibull, yang2025)

PI = {'ec': 0.85, 'enc': 0.90, 'nec': 0.55, 'nenc': 0.60}
MOE_Z    = 1.645
P_LIST   = list(range(1, 51))
K_SPARSE = 15    # nearest candidates per hex for sparse z_ij

def impedance(t, a, b):
    """Weibull-exponential impedance function."""
    t = np.asarray(t, dtype=float)
    w = np.zeros_like(t)
    m = np.isfinite(t) & (t > 0)
    w[m] = np.exp(a * (t[m] ** b))
    return w

def tick(label): print(f'\n[START] {label}'); return time.time()
def tock(t0, label): print(f'[DONE]  {label}  ({time.time()-t0:.1f}s)')

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
# 2. Build candidate set J (uniform o_j = 1)
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
# 3. Build n_ig per hex
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
    if 'SE_elderly_vrt' in _df.columns:
        _p = _df[_df['SE_elderly_vrt'] > 0]
        if len(_p): hex_eld_est = dict(zip(_p['hex_id'], _p['elderly_est']))
    if 'total_pop' in _df.columns:
        _p = _df[_df['total_pop'] > 0]
        if len(_p): hex_total_pop = dict(zip(_p['hex_id'], _p['total_pop']))
if os.path.exists(HEX_ZVEH_SHARE):
    _df = pd.read_csv(HEX_ZVEH_SHARE)
    if 'zv_share' in _df.columns:
        hex_zv_share_map = dict(zip(_df['hex_id'], _df['zv_share']))

n_per_hex = COUNTY_TOTAL / N_HEX
n_ig = {}
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
print(f'  Total pop: {sum(sum(v.values()) for v in n_ig.values()):.0f}')
tock(t0, 'Build n_ig')

# ══════════════════════════════════════════════════════════════════════════════
# 4. Build impedance matrices
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Build impedance matrices')

# Car impedance
car_df = tt_car[tt_car['to_hex'].isin(J_ids)].copy()
car_df['w'] = impedance(car_df['travel_time_min'].values, ALPHA_CAR, BETA_CAR)
W_car = {}
for row in car_df.itertuples(index=False):
    W_car.setdefault(row.from_hex, {})[row.to_hex] = row.w

# Transit impedance
W_transit = {}
if os.path.exists(TT_TRANSIT):
    tr_df = pd.read_csv(TT_TRANSIT)
    tr_df = tr_df[tr_df['to_hex'].isin(J_ids)].copy()
    tr_df['w'] = impedance(tr_df['travel_time_min'].values, ALPHA_TR, BETA_TR)
    for row in tr_df.itertuples(index=False):
        W_transit.setdefault(row.from_hex, {})[row.to_hex] = row.w
    print(f'  Transit OD pairs: {len(tr_df):,}')
else:
    print(f'  Transit OD file not found — transit modes skipped')

# Microtransit impedance — three scenarios (delta * t_car + W)
raw_tt_car = {}
for row in car_df.itertuples(index=False):
    raw_tt_car.setdefault(row.from_hex, {})[row.to_hex] = row.travel_time_min

W_mc_scenarios = {}
for sc_name, sc in MC_SCENARIOS.items():
    W_mc_s = {}
    for i in hex_ids:
        W_mc_s[i] = {}
        for j in J_ids:
            t_c = raw_tt_car.get(i, {}).get(j, np.nan)
            if np.isfinite(t_c):
                t_mt = sc['delta'] * t_c + sc['W']
                W_mc_s[i][j] = np.exp(ALPHA_MC * (t_mt ** BETA_MC))
    W_mc_scenarios[sc_name] = W_mc_s
    d_val = sc['delta']; w_val = sc['W']
    print(f'  Built microtransit impedance: {sc_name}  (delta={d_val}, W={w_val} min)')

print(f'  Car OD pairs: {len(car_df):,}')
tock(t0, 'Build impedance matrices')

# ══════════════════════════════════════════════════════════════════════════════
# 5. Mode definitions
#    {mode_name: {group: W_dict}}
#    car users (ec, nec): W_car
#    no-car users (enc, nenc): mode-specific
# ══════════════════════════════════════════════════════════════════════════════
MODES = {
    'car_only':            {'ec': W_car, 'enc': {},    'nec': W_car, 'nenc': {}},
    'car_for_all':         {'ec': W_car, 'enc': W_car, 'nec': W_car, 'nenc': W_car},
    'car_microtrans_opt':  {'ec': W_car, 'enc': W_mc_scenarios['car_microtrans_opt'],
                            'nec': W_car, 'nenc': W_mc_scenarios['car_microtrans_opt']},
    'car_microtrans':      {'ec': W_car, 'enc': W_mc_scenarios['car_microtrans'],
                            'nec': W_car, 'nenc': W_mc_scenarios['car_microtrans']},
    'car_microtrans_pess': {'ec': W_car, 'enc': W_mc_scenarios['car_microtrans_pess'],
                            'nec': W_car, 'nenc': W_mc_scenarios['car_microtrans_pess']},
}
if W_transit:
    # transit_only removed per model update -- 5 scenarios only
    MODES['transit_for_all'] = {'ec': W_transit, 'enc': W_transit,
                                 'nec': W_transit, 'nenc': W_transit}
    MODES['car_transit']     = {'ec': W_car,      'enc': W_transit,
                                 'nec': W_car,      'nenc': W_transit}

MODE_COLORS = {
    'car_only':            '#e74c3c',
    'car_for_all':         '#2ecc71',
    'car_microtrans_opt':  '#f39c12',
    'car_microtrans':      '#e67e22',
    'car_microtrans_pess': '#d35400',
    'transit_for_all':     '#1abc9c',
    'car_transit':         '#9b59b6',
}
MODE_LABELS = {
    'car_only':            'Car only',
    'car_for_all':         'Car for all',
    'car_microtrans_opt':  'Car + Microtransit (optimistic)',
    'car_microtrans':      'Car + Microtransit (baseline)',
    'car_microtrans_pess': 'Car + Microtransit (pessimistic)',
    'transit_for_all':     'Transit for all',
    'car_transit':         'Car + Transit (SKAT)',
}
MODE_LS = {
    'car_only':            '-',
    'car_for_all':         '--',
    'car_microtrans_opt':  ':',
    'car_microtrans':      '--',
    'car_microtrans_pess': ':',
    'transit_for_all':     '-',
    'car_transit':         '-.',
}
MODE_LW = {
    'car_only':            2.8,
    'car_for_all':         2.2,
    'car_microtrans_opt':  1.6,
    'car_microtrans':      2.2,
    'car_microtrans_pess': 1.6,
    'transit_for_all':     2.2,
    'car_transit':         2.2,
}

print(f'  Active modes ({len(MODES)}): {list(MODES.keys())}')

# ══════════════════════════════════════════════════════════════════════════════
# 6. Pre-compute k nearest candidates per hex (sparse z_ij)
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick(f'Pre-compute k={K_SPARSE} nearest candidates per hex')
nearest_j = {}
for i in hex_ids:
    scored = [(j, W_car.get(i,{}).get(j,0)) for j in J_ids]
    scored.sort(key=lambda x: -x[1])
    nearest_j[i] = [j for j,_ in scored[:K_SPARSE]]
tock(t0, 'Pre-compute nearest candidates')

# ══════════════════════════════════════════════════════════════════════════════
# 7. A_max per mode (allocation-based upper bound)
#    A_max = Σ_i Σ_g π_g * n_ig * max_j(o_j * w_ij(g))   [all J open]
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Compute A_max per mode')
A_max = {}
for mode_name, W_by_seg in MODES.items():
    total = 0.0
    for i in hex_ids:
        for g, Wm in W_by_seg.items():
            if not Wm: continue
            best = max((Wm.get(i,{}).get(j,0)) for j in nearest_j[i])
            total += PI[g] * n_ig[i][g] * best
    A_max[mode_name] = total
    print(f'  A_max [{mode_name}]: {total:.2f}')
tock(t0, 'Compute A_max')

# ══════════════════════════════════════════════════════════════════════════════
# 8. Solve allocation BLP for each mode × P
#
#    maximize  Σ_i Σ_g π_g * n_ig * Σ_{j∈nearest(i)} w_ij(g) * z_ij(g)
#    s.t.      Σ_j z_ij(g) <= 1    ∀i,g   (nearest j only)
#              z_ij(g) <= y_j       ∀i,j,g
#              Σ_j y_j = P
#              y_j, z_ij ∈ {0,1}
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Solve allocation BLP P=1..50 all modes')

results = []

for mode_name, W_by_seg in MODES.items():
    print(f'\n  Mode: {MODE_LABELS[mode_name]}')
    A_max_val = A_max[mode_name]

    for P in P_LIST:
        t_solve = time.time()
        prob = LpProblem(f'{mode_name}_P{P}', LpMaximize)

        # Facility variables
        y = {j: LpVariable(f'y_{j}', cat=LpBinary) for j in J_ids}

        # Sparse allocation variables: z[g][i][j]
        z = {}
        for g, Wm in W_by_seg.items():
            if not Wm: continue
            z[g] = {}
            for i in hex_ids:
                if n_ig[i][g] <= 0: continue
                z[g][i] = {}
                for j in nearest_j[i]:
                    if Wm.get(i,{}).get(j,0) > 0:
                        z[g][i][j] = LpVariable(f'z_{g}_{i}_{j}', cat=LpBinary)

        # Objective
        obj_terms = []
        for g, zi_dict in z.items():
            Wm = W_by_seg[g]
            for i, zij in zi_dict.items():
                for j, zvar in zij.items():
                    coeff = PI[g] * n_ig[i][g] * Wm.get(i,{}).get(j,0)
                    if coeff > 0:
                        obj_terms.append(coeff * zvar)
        prob += lpSum(obj_terms)

        # P stops constraint
        prob += lpSum(y[j] for j in J_ids) == P

        # Allocation constraints
        for g, zi_dict in z.items():
            for i, zij in zi_dict.items():
                if zij:
                    prob += lpSum(zij.values()) <= 1
                    for j, zvar in zij.items():
                        prob += zvar <= y[j]

        prob.solve(PULP_CBC_CMD(msg=0))

        sel     = [j for j in J_ids if value(y[j]) is not None and value(y[j]) > 0.5]
        raw_obj = value(prob.objective) or 0.0
        A_tilde = raw_obj / A_max_val if A_max_val > 0 else 0

        results.append({
            'mode':       mode_name,
            'P':          P,
            'objective':  raw_obj,
            'A_tilde':    A_tilde,
            'A_max':      A_max_val,
            'n_selected': len(sel),
            'selected_stops': '|'.join(sel),
            'solve_s':    time.time() - t_solve,
        })
        print(f'    P={P:>3}: A_tilde={A_tilde:.4f}  ({time.time()-t_solve:.2f}s)')

tock(t0, 'Solve allocation BLP')

# ══════════════════════════════════════════════════════════════════════════════
# 9. Save results
# ══════════════════════════════════════════════════════════════════════════════
res_df = pd.DataFrame(results)
out_csv = f'{OUTPUTS}/primal_accessibility_results.csv'
res_df.drop(columns=['selected_stops']).to_csv(out_csv, index=False)
print(f'\n  Saved: {out_csv}')

# Also save with selected stops for mapping
out_full = f'{OUTPUTS}/primal_accessibility_results_full.csv'
res_df.to_csv(out_full, index=False)
print(f'  Saved: {out_full}')

# ══════════════════════════════════════════════════════════════════════════════
# 10. Step plot — A_tilde vs P for all modes
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating plots...')

# Draw order: transit first, car cluster on top, MC scenarios grouped
DRAW_ORDER = ['transit_for_all', 'car_transit',
              'car_microtrans_pess', 'car_microtrans_opt',
              'car_microtrans', 'car_for_all', 'car_only']

# Annotation offsets to prevent stacking
ANNOT_OFFSET = {
    'car_only':            -0.015,
    'car_for_all':         +0.015,
    'car_microtrans':       0.000,
    'car_microtrans_opt':  +0.030,
    'car_microtrans_pess': -0.030,
    'transit_for_all':     +0.010,
    'car_transit':         -0.010,
}

fig, ax = plt.subplots(figsize=(16, 8))

# Shade microtransit band (opt to pess)
if all(m in res_df['mode'].unique()
       for m in ['car_microtrans_opt','car_microtrans_pess']):
    opt_sub  = res_df[res_df['mode']=='car_microtrans_opt'].sort_values('P')
    pess_sub = res_df[res_df['mode']=='car_microtrans_pess'].sort_values('P')
    ax.fill_between(opt_sub['P'], pess_sub['A_tilde'], opt_sub['A_tilde'],
                    alpha=0.15, color='#e67e22',
                    label='Car + Microtransit range (opt–pess)')

for mode_name in DRAW_ORDER:
    if mode_name not in res_df['mode'].unique():
        continue
    sub   = res_df[res_df['mode']==mode_name].sort_values('P')
    color = MODE_COLORS.get(mode_name, '#888')
    label = MODE_LABELS.get(mode_name, mode_name)
    lw    = MODE_LW.get(mode_name, 2.0)
    ls    = MODE_LS.get(mode_name, '-')

    ax.step(sub['P'], sub['A_tilde'], where='post',
            color=color, lw=lw, ls=ls, label=label)

    # Right-side annotation
    final = float(sub['A_tilde'].iloc[-1])
    yann  = final + ANNOT_OFFSET.get(mode_name, 0)
    ax.text(51.0, yann, f'{final:.4f}',
            fontsize=9, va='center', color=color,
            fontweight='bold' if mode_name in ['car_only','transit_for_all'] else 'normal')

ax.set_xlabel('Number of clinic stops (P)', fontsize=14)
ax.set_ylabel(r'$\tilde{\mathcal{A}}$ (normalised accessibility)', fontsize=14)
ax.set_title('Primal Accessibility vs $P$ — Allocation BLP\n'
             'Rockingham County, NC  |  Uniform $o_j$  |  Correct submodular formulation\n'
             'Car + Microtransit shown as band: optimistic ($\\delta=1.0$, $W=3.74$ min) '
             'to pessimistic ($\\delta=2.0$, $W=17.67$ min)',
             fontsize=12, pad=8)
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_xlim(1, 55)
ax.set_ylim(0, 1.05)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
out_plot = f'{OUTPUTS}/primal_accessibility_step_plot.png'
plt.savefig(out_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {out_plot}')

# ══════════════════════════════════════════════════════════════════════════════
# 11. Comparison: primal allocation vs original linear (greedy)
#     Load linear results from mayodan_first_P.csv if available
# ══════════════════════════════════════════════════════════════════════════════
linear_csv = f'{OUTPUTS}/mayodan_first_P.csv'
if os.path.exists(linear_csv):
    linear_df = pd.read_csv(linear_csv)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Primal Accessibility: Allocation BLP vs Original Linear Formulation\n'
                 'Car only, Uniform $o_j$  |  Rockingham County, NC',
                 fontsize=13, y=1.02)

    # Left: side by side step plots
    ax = axes[0]
    car_sub = res_df[res_df['mode']=='car_only'].sort_values('P')
    ax.step(car_sub['P'], car_sub['A_tilde'], where='post',
            color='#e74c3c', lw=2.5, label='Allocation BLP (correct)')
    ax.step(linear_df['P'], linear_df['A_tilde'], where='post',
            color='#95a5a6', lw=2, ls='--', label='Linear greedy (original, incorrect)')

    ax.set_xlabel('P', fontsize=13)
    ax.set_ylabel(r'$\tilde{\mathcal{A}}$', fontsize=13)
    ax.set_title('$\\tilde{\\mathcal{A}}$ vs $P$', fontsize=12, pad=5)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim(1, 51); ax.set_ylim(0)

    # Right: difference plot
    ax2 = axes[1]
    alloc_vals  = car_sub.set_index('P')['A_tilde']
    linear_vals = linear_df.set_index('P')['A_tilde']
    common_P    = sorted(set(alloc_vals.index) & set(linear_vals.index))
    diff        = [alloc_vals[p] - linear_vals[p] for p in common_P]

    colors_diff = ['#e74c3c' if d > 0 else '#3498db' for d in diff]
    ax2.bar(common_P, diff, color=colors_diff, edgecolor='white', linewidth=0.3)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xlabel('P', fontsize=13)
    ax2.set_ylabel('Allocation minus Linear', fontsize=13)
    ax2.set_title('Difference: Allocation BLP $-$ Linear\nRed = allocation higher  Blue = linear higher',
                  fontsize=12, pad=5)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='both', labelsize=11)
    ax2.set_xlim(0.3, 51)

    plt.tight_layout()
    out_comp = f'{OUTPUTS}/primal_vs_linear_comparison.png'
    plt.savefig(out_comp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_comp}')
else:
    print(f'  Linear results not found at {linear_csv} — skipping comparison plot')

# ══════════════════════════════════════════════════════════════════════════════
# 12. Summary table
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PRIMAL ACCESSIBILITY SUMMARY (Allocation BLP)')
print('='*65)
print(f'\n  {"Mode":<22}  {"P=1":>6}  {"P=4":>6}  {"P=10":>6}  '
      f'{"P=20":>6}  {"P=50":>6}')
print(f'  {"-"*22}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}')
for mode_name in MODES:
    sub = res_df[res_df['mode']==mode_name].set_index('P')
    vals = [sub.loc[p,'A_tilde'] if p in sub.index else float('nan')
            for p in [1,4,10,20,50]]
    print(f'  {MODE_LABELS[mode_name]:<22}  '
          + '  '.join(f'{v:>6.3f}' for v in vals))

print('\nDone.')