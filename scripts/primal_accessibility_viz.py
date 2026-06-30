"""
primal_accessibility_viz.py
============================
Standalone visualization for primal accessibility results.
Reads outputs from primal_accessibility.py — run that first.

Inputs (from outputs/):
    primal_accessibility_results_full.csv  — A_tilde + selected stops per mode per P
    mayodan_first_P.csv                    — linear greedy results for comparison

Outputs (to outputs/):
    primal_accessibility_step_plot.png        — step plot, all modes
    primal_microtransit_range_plot.png        — standalone car+microtransit zoom
    primal_accessibility_combined.png         — all-modes plot with microtransit
                                                 zoom shown as an inset (recommended
                                                 single figure for the paper)
    primal_vs_linear_comparison.png           — allocation vs linear car only
    primal_heatmap_{mode}.png                 — per-hex heatmap at P=4,12,26
    no_car_accessibility_results.csv          — joint-optimum sites, rescored
                                                 for the no-car population
    no_car_accessibility_plot.png
    no_car_heatmap_{mode}.png

Optional (run no_car_facility_optimization.py first to enable):
    no_car_optimized_vs_rescored.png          — joint-optimum sites vs sites
                                                 re-optimized specifically for
                                                 the no-car population
"""

import os, json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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
HOSPITALS_CSV  = f'{OUTPUTS}/rockingham_hospitals.csv'

RESULTS_FULL = f'{OUTPUTS}/primal_accessibility_results_full.csv'
LINEAR_CSV   = f'{OUTPUTS}/mayodan_first_P.csv'

CANDIDATE_NAICS = [813110, 611110, 624410, 621111,
                   922110, 922120, 922130, 922140, 922150, 922160, 922190]

ALPHA_CAR = -0.020097; BETA_CAR = 1.361630
ALPHA_TR  = -0.002062; BETA_TR  = 1.608027
ALPHA_MC  = ALPHA_CAR; BETA_MC  = BETA_CAR
import math

# Microtransit range: t_ij^mt = delta * t_car + W
MC_SCENARIOS = {
    'car_microtrans_opt':  {'delta': 1.0, 'W': 3.74,  'label': 'Car + Microtransit (optimistic)',  'color': '#f39c12', 'ls': ':', 'lw': 1.8},
    'car_microtrans':      {'delta': 1.3, 'W': 10.00, 'label': 'Car + Microtransit (baseline)',    'color': '#e67e22', 'ls': '--','lw': 2.2},
    'car_microtrans_pess': {'delta': 2.0, 'W': 17.67, 'label': 'Car + Microtransit (pessimistic)', 'color': '#d35400', 'ls': ':', 'lw': 1.8},
}

PI       = {'ec': 0.85, 'enc': 0.90, 'nec': 0.55, 'nenc': 0.60}
K_SPARSE = 15
KEY_P    = [4, 12, 26]  # P=50 dropped for now; add back later if needed

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
    'car_microtrans_opt':  'Car + Microtransit (optimistic, δ=1.0, W=3.74 min)',
    'car_microtrans':      'Car + Microtransit (baseline, δ=1.3, W=10 min)',
    'car_microtrans_pess': 'Car + Microtransit (pessimistic, δ=2.0, W=17.67 min)',
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
    'car_microtrans_opt':  1.8,
    'car_microtrans':      2.2,
    'car_microtrans_pess': 1.8,
    'transit_for_all':     2.2,
    'car_transit':         2.2,
}

# Heatmap colormap: single-hue sequential ramp (light = low accessibility,
# dark = high accessibility). Switched from a diverging purple-green map,
# since accessibility is a one-directional quantity with no meaningful
# midpoint/sign-flip, which is what diverging colormaps are meant to encode.
# Swap to 'Greens' (or any other single-hue sequential map) if preferred.
HEAT_CMAP = 'Purples'

def impedance(t, a, b):
    t = np.asarray(t, dtype=float); w = np.zeros_like(t)
    m = np.isfinite(t) & (t > 0); w[m] = np.exp(a*(t[m]**b))
    return w

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load results CSV
# ══════════════════════════════════════════════════════════════════════════════
print('[1] Loading primal accessibility results...')
if not os.path.exists(RESULTS_FULL):
    print(f'ERROR: {RESULTS_FULL} not found. Run primal_accessibility.py first.')
    exit(1)

res_df   = pd.read_csv(RESULTS_FULL)
modes    = res_df['mode'].unique().tolist()
print(f'  Modes: {modes}')
print(f'  P range: {res_df["P"].min()} to {res_df["P"].max()}')
print(f'  Total rows: {len(res_df)}')

# ══════════════════════════════════════════════════════════════════════════════
# 2. Load spatial data for heatmaps
# ══════════════════════════════════════════════════════════════════════════════
print('[2] Loading spatial data...')
hexes    = pd.read_csv(HEX_CENTROIDS)
hex_gdf  = gpd.read_file(HEX_GRID).to_crs('EPSG:4326')
hex_col  = next((c for c in hex_gdf.columns
                 if 'hex' in c.lower() or c == 'h3_index'), hex_gdf.columns[0])
hex_gdf  = hex_gdf.rename(columns={hex_col: 'hex_id'})
hex_ids  = hexes['hex_id'].tolist()
hospitals = pd.read_csv(HOSPITALS_CSV) if os.path.exists(HOSPITALS_CSV) else None
pois     = pd.read_csv(POIS, low_memory=False)

with open(ACS_POP) as f: acs_pop = json.load(f)
with open(ACS_VEH) as f: acs_veh = json.load(f)

# ── n_ig ──────────────────────────────────────────────────────────────────────
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
headers_veh  = acs_veh[0]
COUNTY_ZV    = sum(float(dict(zip(headers_veh,r)).get('B08201_002E',0) or 0) for r in acs_veh[1:])
COUNTY_HH    = max(sum(float(dict(zip(headers_veh,r)).get('B08201_001E',0) or 0) for r in acs_veh[1:]),1)
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

n_per_hex = COUNTY_TOTAL / len(hex_ids)
n_ig = {}
for hid in hex_ids:
    n_h  = hex_total_pop.get(hid, n_per_hex)
    e_h  = min(hex_eld_est.get(hid, COUNTY_ELDERLY/len(hex_ids)), n_h)
    zv_h = hex_zv_share_map.get(hid, COUNTY_ZV_SHARE)
    nn_h = max(n_h - e_h, 0)
    n_ig[hid] = {'ec': e_h*(1-zv_h), 'enc': e_h*zv_h,
                 'nec': nn_h*(1-zv_h), 'nenc': nn_h*zv_h}

# ── Candidate set J ───────────────────────────────────────────────────────────
pois['naics_code'] = pd.to_numeric(pois['naics_code'], errors='coerce')
cand = pois[pois['naics_code'].isin(CANDIDATE_NAICS)].copy()
poi_gdf = gpd.GeoDataFrame(cand,
    geometry=gpd.points_from_xy(cand['longitude'], cand['latitude']),
    crs='EPSG:4326')
if 'index_right' in poi_gdf.columns:
    poi_gdf = poi_gdf.drop(columns=['index_right'])
joined = gpd.sjoin(poi_gdf, hex_gdf[['hex_id','geometry']].reset_index(drop=True),
                   how='left', predicate='within')
joined['total_visit_2025'] = pd.to_numeric(
    joined['total_visit_2025'], errors='coerce').fillna(0)
hv = joined.groupby('hex_id')['total_visit_2025'].sum().reset_index()
hv.columns = ['hex_id','total_visits']
hv = hv[hv['total_visits'] > 0]
J_df  = hv[hv['hex_id'].isin(hex_ids)].copy().reset_index(drop=True)
J_df  = J_df.merge(hexes[['hex_id','lon','lat']], on='hex_id', how='left')
J_ids = J_df['hex_id'].tolist()
print(f'  |J| = {len(J_ids)} candidate stops')

# ── Impedance matrices ────────────────────────────────────────────────────────
tt_car_df = pd.read_csv(TT_CAR)
car_df    = tt_car_df[tt_car_df['to_hex'].isin(J_ids)].copy()
car_df['w'] = impedance(car_df['travel_time_min'].values, ALPHA_CAR, BETA_CAR)
W_car = {}
for row in car_df.itertuples(index=False):
    W_car.setdefault(row.from_hex, {})[row.to_hex] = row.w

W_transit = {}
if os.path.exists(TT_TRANSIT):
    tr_df = pd.read_csv(TT_TRANSIT)
    tr_df = tr_df[tr_df['to_hex'].isin(J_ids)].copy()
    tr_df['w'] = impedance(tr_df['travel_time_min'].values, ALPHA_TR, BETA_TR)
    for row in tr_df.itertuples(index=False):
        W_transit.setdefault(row.from_hex, {})[row.to_hex] = row.w

raw_tt_car = {}
for row in car_df.itertuples(index=False):
    raw_tt_car.setdefault(row.from_hex, {})[row.to_hex] = row.travel_time_min

# Build one impedance dict per MC scenario
W_mc_dict = {}
for sc_name, sc in MC_SCENARIOS.items():
    W_mc_s = {}
    for i in hex_ids:
        W_mc_s[i] = {}
        for j in J_ids:
            t_c = raw_tt_car.get(i,{}).get(j, np.nan)
            if np.isfinite(t_c):
                t_mt = sc['delta'] * t_c + sc['W']
                W_mc_s[i][j] = np.exp(ALPHA_MC * (t_mt ** BETA_MC))
    W_mc_dict[sc_name] = W_mc_s

MODES = {
    'car_only':            {'ec': W_car, 'enc': {},    'nec': W_car, 'nenc': {}},
    'car_for_all':         {'ec': W_car, 'enc': W_car, 'nec': W_car, 'nenc': W_car},
    'car_microtrans_opt':  {'ec': W_car, 'enc': W_mc_dict['car_microtrans_opt'],
                            'nec': W_car, 'nenc': W_mc_dict['car_microtrans_opt']},
    'car_microtrans':      {'ec': W_car, 'enc': W_mc_dict['car_microtrans'],
                            'nec': W_car, 'nenc': W_mc_dict['car_microtrans']},
    'car_microtrans_pess': {'ec': W_car, 'enc': W_mc_dict['car_microtrans_pess'],
                            'nec': W_car, 'nenc': W_mc_dict['car_microtrans_pess']},
}
if W_transit:
    MODES['transit_for_all'] = {'ec': W_transit, 'enc': W_transit,
                                 'nec': W_transit, 'nenc': W_transit}
    MODES['car_transit']     = {'ec': W_car,      'enc': W_transit,
                                 'nec': W_car,      'nenc': W_transit}

# k nearest per hex
nearest_j = {}
for i in hex_ids:
    scored = [(j, W_car.get(i,{}).get(j,0)) for j in J_ids]
    scored.sort(key=lambda x: -x[1])
    nearest_j[i] = [j for j,_ in scored[:K_SPARSE]]

def per_hex_A(sel_hexes, W_by_seg, PI_weights=None):
    if PI_weights is None:
        PI_weights = PI
    sel_set = set(sel_hexes)
    A_hex = {}
    for i in hex_ids:
        total = 0.0
        for g, Wm in W_by_seg.items():
            if PI_weights.get(g, 0) <= 0: continue
            if not Wm or n_ig[i][g] <= 0: continue
            avail = [j for j in nearest_j[i] if j in sel_set]
            if not avail: continue
            best_w = max(Wm.get(i,{}).get(j,0) for j in avail)
            total += PI_weights[g] * n_ig[i][g] * best_w
        A_hex[i] = total
    return A_hex

print('  Spatial data loaded.')

# ══════════════════════════════════════════════════════════════════════════════
# 2b. No-car (transit-dependent) restricted accessibility
#     Population-weighted A_tilde dilutes mode differences because car-owning
#     households (ec, nec) get identical car access in car_only, car_for_all,
#     and every car_microtrans scenario -- only the no-car households (enc,
#     nenc) actually change between those modes. Zeroing out the car-owning
#     segments isolates the effect on the population these interventions are
#     meant to serve.
# ══════════════════════════════════════════════════════════════════════════════
print('[2b] Computing no-car restricted accessibility...')

NO_CAR_PI = {'ec': 0.0, 'enc': PI['enc'], 'nec': 0.0, 'nenc': PI['nenc']}

nocar_rows = []
for mode_name, W_by_seg in MODES.items():
    # Ceiling: every candidate stop open, restricted to the no-car population
    A_max_hex = per_hex_A(J_ids, W_by_seg, PI_weights=NO_CAR_PI)
    A_max_nocar = sum(A_max_hex.values())

    mode_rows = res_df[res_df['mode'] == mode_name]
    for row in mode_rows.itertuples(index=False):
        sel_str   = getattr(row, 'selected_stops', '')
        sel_hexes = [s for s in str(sel_str).split('|') if s and s != 'nan']
        A_hex     = per_hex_A(sel_hexes, W_by_seg, PI_weights=NO_CAR_PI)
        A_raw     = sum(A_hex.values())
        A_tilde_nocar = A_raw / A_max_nocar if A_max_nocar > 0 else 0.0
        nocar_rows.append({'mode': mode_name, 'P': row.P,
                            'A_tilde_nocar': A_tilde_nocar})

nocar_df = pd.DataFrame(nocar_rows)
nocar_out = f'{OUTPUTS}/no_car_accessibility_results.csv'
nocar_df.to_csv(nocar_out, index=False)
print(f'  Saved: {nocar_out}')

fig, ax = plt.subplots(figsize=(13, 8))
NOCAR_DRAW_ORDER = ['transit_for_all', 'car_transit',
                     'car_microtrans_pess', 'car_microtrans_opt',
                     'car_microtrans', 'car_for_all', 'car_only']
for mode_name in NOCAR_DRAW_ORDER:
    if mode_name not in nocar_df['mode'].unique():
        continue
    sub = nocar_df[nocar_df['mode']==mode_name].sort_values('P')
    ax.step(sub['P'], sub['A_tilde_nocar'], where='post',
            color=MODE_COLORS.get(mode_name, '#888'),
            lw=MODE_LW.get(mode_name, 2.0),
            ls=MODE_LS.get(mode_name, '-'),
            label=MODE_LABELS.get(mode_name, mode_name))

ax.set_xlabel('Number of clinic stops ($P$)', fontsize=14)
ax.set_ylabel(r'$\tilde{\mathcal{A}}_{\mathrm{no\ car}}$ (no-car population only)', fontsize=14)
ax.set_title('Accessibility for the No-Car Population vs $P$\n'
             'Rockingham County, NC  |  Car-owning segments excluded',
             fontsize=13, pad=8)
ax.set_xlim(1, res_df['P'].max())
ax.set_ylim(0, 1.05)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=9.5, loc='upper center', bbox_to_anchor=(0.5, -0.13),
          ncol=3, framealpha=0.95)
plt.tight_layout()
out_nocar = f'{OUTPUTS}/no_car_accessibility_plot.png'
plt.savefig(out_nocar, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {out_nocar}')

# ══════════════════════════════════════════════════════════════════════════════
# 3. Step plot — all modes (combined, with MC band shading)
# ══════════════════════════════════════════════════════════════════════════════
print('[3] Step plot (all modes combined)...')

DRAW_ORDER_ALL = ['transit_for_all', 'car_transit',
                  'car_microtrans_pess', 'car_microtrans_opt',
                  'car_microtrans', 'car_for_all', 'car_only']

fig, ax = plt.subplots(figsize=(15, 8.5))

P_MAX_ALL = res_df['P'].max()
end_vals_all = []
for mode_name in DRAW_ORDER_ALL:
    if mode_name not in res_df['mode'].unique():
        continue
    sub   = res_df[res_df['mode']==mode_name].sort_values('P')
    color = MODE_COLORS.get(mode_name, '#888')
    label = MODE_LABELS.get(mode_name, mode_name)
    lw    = MODE_LW.get(mode_name, 2.0)
    ls    = MODE_LS.get(mode_name, '-')

    ax.step(sub['P'], sub['A_tilde'], where='post',
            color=color, lw=lw, ls=ls, label=label)

    final = float(sub['A_tilde'].iloc[-1])
    end_vals_all.append((mode_name, final))

# End-of-line value labels, auto-spaced so close-together curves never
# produce overlapping text (sorted by value, minimum gap enforced).
end_vals_all.sort(key=lambda x: x[1])
min_gap_all = 0.022
adjusted_all = []
for mode_name, val in end_vals_all:
    y = val
    if adjusted_all and y - adjusted_all[-1][1] < min_gap_all:
        y = adjusted_all[-1][1] + min_gap_all
    adjusted_all.append((mode_name, y))

for mode_name, y_label in adjusted_all:
    val   = dict(end_vals_all)[mode_name]
    color = MODE_COLORS.get(mode_name, '#888')
    ax.annotate(f'{val:.3f}',
                xy=(P_MAX_ALL, val),
                xytext=(P_MAX_ALL * 1.025, y_label),
                fontsize=8.5, color=color, va='center',
                arrowprops=dict(arrowstyle='-', color=color, lw=0.6))

ax.set_xlabel('Number of clinic stops ($P$)', fontsize=14)
ax.set_ylabel(r'$\tilde{\mathcal{A}}$ (normalised accessibility)', fontsize=14)
ax.set_title('Primal Accessibility vs $P$ — Allocation BLP\n'
             'Rockingham County, NC  |  Uniform $o_j$  |  Correct submodular formulation',
             fontsize=13, pad=8)
ax.set_xlim(1, P_MAX_ALL * 1.18)
ax.set_ylim(0, 1.05)
ax.tick_params(axis='both', labelsize=12)

# Legend placed below the axes, outside the plotting area, so it never
# overlaps the curves or the end-of-line value labels.
ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.13),
          ncol=3, framealpha=0.95)

plt.tight_layout()
out = f'{OUTPUTS}/primal_accessibility_step_plot.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 3b. Dedicated Car + Microtransit range plot
#     - single panel, zoomed view by default (P = 1-20)
#     - only the three microtransit scenarios (no car-only reference line)
#     - no parameter text box (parameters are in the legend labels instead)
#     - end-of-line value labels are spaced out so they don't overlap
# ══════════════════════════════════════════════════════════════════════════════
print('[3b] Car + Microtransit range plot...')

mc_modes_list = ['car_microtrans_opt', 'car_microtrans', 'car_microtrans_pess']
mc_available  = [m for m in mc_modes_list if m in res_df['mode'].unique()]

ZOOM_P_MAX = 20  # default zoom window shown to the user

if len(mc_available) >= 2:
    fig, ax_mc = plt.subplots(figsize=(11, 7.5))

    # Shade band (opt -> pess)
    if 'car_microtrans_opt' in mc_available and 'car_microtrans_pess' in mc_available:
        opt_s  = res_df[res_df['mode']=='car_microtrans_opt'].sort_values('P')
        pess_s = res_df[res_df['mode']=='car_microtrans_pess'].sort_values('P')
        ax_mc.fill_between(opt_s['P'], pess_s['A_tilde'], opt_s['A_tilde'],
                           alpha=0.20, color='#e67e22',
                           label='Uncertainty band (opt–pess)')

    # Three MC scenarios only — labels carry delta/W so no separate text box needed
    scenario_data = {}
    for mode_name in mc_available:
        sc    = MC_SCENARIOS[mode_name]
        sub   = res_df[res_df['mode']==mode_name].sort_values('P')
        color = MODE_COLORS[mode_name]
        lw    = MODE_LW[mode_name]
        ls    = MODE_LS[mode_name]
        tag   = {'car_microtrans_opt': 'optimistic',
                 'car_microtrans': 'baseline',
                 'car_microtrans_pess': 'pessimistic'}[mode_name]
        label = f'{tag.capitalize()}: $\\delta={sc["delta"]}$, $W={sc["W"]}$ min'

        ax_mc.step(sub['P'], sub['A_tilde'], where='post',
                   color=color, lw=lw, ls=ls, label=label)
        scenario_data[mode_name] = sub

    # End-of-window value labels (at P = ZOOM_P_MAX), nudged apart so they
    # never overlap regardless of how close the curves are numerically.
    end_vals = []
    for mode_name in mc_available:
        sub = scenario_data[mode_name]
        row = sub[sub['P'] <= ZOOM_P_MAX].sort_values('P')
        if len(row) == 0:
            continue
        val = float(row['A_tilde'].iloc[-1])
        end_vals.append((mode_name, val))

    # Sort by value and enforce a minimum vertical gap between labels
    end_vals.sort(key=lambda x: x[1])
    min_gap = 0.025
    adjusted = []
    for i, (mode_name, val) in enumerate(end_vals):
        y = val
        if adjusted and y - adjusted[-1][1] < min_gap:
            y = adjusted[-1][1] + min_gap
        adjusted.append((mode_name, y))

    for mode_name, y_label in adjusted:
        val   = dict(end_vals)[mode_name]
        color = MODE_COLORS[mode_name]
        ax_mc.annotate(f'{val:.3f}',
                       xy=(ZOOM_P_MAX, val),
                       xytext=(ZOOM_P_MAX * 1.02, y_label),
                       fontsize=9.5, color=color, va='center',
                       arrowprops=dict(arrowstyle='-', color=color, lw=0.7))

    ax_mc.set_xlabel('Number of clinic stops ($P$)', fontsize=14)
    ax_mc.set_ylabel(r'$\tilde{\mathcal{A}}$ (normalised accessibility)', fontsize=14)
    ax_mc.set_title(
        'Car + Microtransit (RCATS) — Accessibility Range\n'
        r'$t_{ij}^{\mathrm{mt}} = \delta \cdot t_{ij}^{\mathrm{car}} + W$  '
        '|  Rockingham County, NC',
        fontsize=13, pad=8)
    ax_mc.set_xlim(1, ZOOM_P_MAX * 1.18)  # room for end labels, no overlap with plot
    ax_mc.tick_params(axis='both', labelsize=12)
    ax_mc.legend(fontsize=10.5, loc='lower right', framealpha=0.92)

    plt.tight_layout()
    out_mc = f'{OUTPUTS}/primal_microtransit_range_plot.png'
    plt.savefig(out_mc, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_mc}')
else:
    print('  MC range skipped -- need opt and pess variants in results CSV')

# ══════════════════════════════════════════════════════════════════════════════
# 3c. Combined figure: all-modes step plot with the Car+Microtransit range
#     shown as an inset zoom (P = 1-20), instead of two separate images.
#     A dashed box on the main plot marks exactly which region the inset
#     is zooming into.
# ══════════════════════════════════════════════════════════════════════════════
print('[3c] Combined plot (all modes + microtransit inset)...')

if len(mc_available) >= 2 and 'car_only' in res_df['mode'].unique():
    fig, ax = plt.subplots(figsize=(15, 9))

    P_MAX_ALL = res_df['P'].max()
    for mode_name in DRAW_ORDER_ALL:
        if mode_name not in res_df['mode'].unique():
            continue
        sub   = res_df[res_df['mode']==mode_name].sort_values('P')
        ax.step(sub['P'], sub['A_tilde'], where='post',
                color=MODE_COLORS.get(mode_name, '#888'),
                lw=MODE_LW.get(mode_name, 2.0),
                ls=MODE_LS.get(mode_name, '-'),
                label=MODE_LABELS.get(mode_name, mode_name))

    ax.set_xlabel('Number of clinic stops ($P$)', fontsize=14)
    ax.set_ylabel(r'$\tilde{\mathcal{A}}$ (normalised accessibility)', fontsize=14)
    ax.set_title('Primal Accessibility vs $P$ — Allocation BLP\n'
                 'Rockingham County, NC  |  Uniform $o_j$  |  Correct submodular formulation',
                 fontsize=13, pad=8)
    ax.set_xlim(1, P_MAX_ALL)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=9.5, loc='upper center', bbox_to_anchor=(0.5, -0.13),
              ncol=3, framealpha=0.95)

    # ── Inset: zoom into Car + Microtransit range, P = 1-20 ──────────────────
    ZOOM_P_MAX_INSET = 20
    axins = ax.inset_axes([0.52, 0.08, 0.46, 0.46])  # [x0, y0, w, h] in axes fraction

    car_ref = res_df[res_df['mode']=='car_only'].sort_values('P')
    axins.step(car_ref['P'], car_ref['A_tilde'], where='post',
              color=MODE_COLORS['car_only'], lw=2.0, ls='-', alpha=0.7)

    for mode_name in mc_modes_list:
        if mode_name not in res_df['mode'].unique():
            continue
        sub = res_df[res_df['mode']==mode_name].sort_values('P')
        axins.step(sub['P'], sub['A_tilde'], where='post',
                  color=MODE_COLORS[mode_name], lw=MODE_LW[mode_name], ls=MODE_LS[mode_name])

    zoom_modes  = mc_modes_list + ['car_only']
    zoom_subset = res_df[(res_df['mode'].isin(zoom_modes)) & (res_df['P'] <= ZOOM_P_MAX_INSET)]
    zoom_y_lo   = zoom_subset['A_tilde'].min() - 0.03
    zoom_y_hi   = zoom_subset['A_tilde'].max() + 0.03

    axins.set_xlim(1, ZOOM_P_MAX_INSET)
    axins.set_ylim(zoom_y_lo, zoom_y_hi)
    axins.set_title('Zoom: Car + Microtransit range ($P=1$–$20$)', fontsize=9.5)
    axins.tick_params(axis='both', labelsize=8)
    for spine in axins.spines.values():
        spine.set_edgecolor('#555')
        spine.set_linewidth(1.0)

    # Dashed box + connector lines linking the inset to the zoomed region
    # of the main plot.
    mark_inset(ax, axins, loc1=2, loc2=3, fc='none', ec='0.5', lw=1.0, ls='--')

    plt.tight_layout()
    out_combined = f'{OUTPUTS}/primal_accessibility_combined.png'
    plt.savefig(out_combined, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_combined}')
else:
    print('  Combined plot skipped -- need car_only + microtransit scenarios in results CSV')

# ══════════════════════════════════════════════════════════════════════════════
# 4. Comparison: allocation vs linear (car only)
# ══════════════════════════════════════════════════════════════════════════════
print('[4] Comparison plot (allocation vs linear)...')
if os.path.exists(LINEAR_CSV):
    linear_df = pd.read_csv(LINEAR_CSV)
    car_sub   = res_df[res_df['mode']=='car_only'].sort_values('P')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Primal Accessibility: Allocation BLP vs Original Linear Formulation\n'
                 'Car only  |  Uniform $o_j$  |  Rockingham County, NC',
                 fontsize=13, y=1.02)

    ax = axes[0]
    ax.step(car_sub['P'], car_sub['A_tilde'], where='post',
            color='#e74c3c', lw=2.5, label='Allocation BLP (correct, submodular)')
    ax.step(linear_df['P'], linear_df['A_tilde'], where='post',
            color='#95a5a6', lw=2, ls='--', label='Linear greedy (original, incorrect)')
    ax.set_xlabel('$P$', fontsize=13)
    ax.set_ylabel(r'$\tilde{\mathcal{A}}$', fontsize=13)
    ax.set_title(r'$\tilde{\mathcal{A}}$ vs $P$', fontsize=12, pad=5)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim(1, 51); ax.set_ylim(0)

    ax2 = axes[1]
    alloc_v  = car_sub.set_index('P')['A_tilde']
    linear_v = linear_df.set_index('P')['A_tilde']
    common_P = sorted(set(alloc_v.index) & set(linear_v.index))
    diff     = [alloc_v[p] - linear_v[p] for p in common_P]
    colors_d = ['#e74c3c' if d > 0 else '#3498db' for d in diff]
    ax2.bar(common_P, diff, color=colors_d, edgecolor='white', linewidth=0.3)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xlabel('$P$', fontsize=13)
    ax2.set_ylabel('Allocation $-$ Linear', fontsize=13)
    ax2.set_title('Difference: Allocation BLP $-$ Linear\n'
                  'Red = allocation higher  Blue = linear higher',
                  fontsize=12, pad=5)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='both', labelsize=11)
    ax2.set_xlim(0.3, 51)

    plt.tight_layout()
    out = f'{OUTPUTS}/primal_vs_linear_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {out}')
else:
    print(f'  {LINEAR_CSV} not found — skipping comparison plot')

# ══════════════════════════════════════════════════════════════════════════════
# 5. Heatmaps — per-hex accessibility at P=4, 12, 26, 50
# ══════════════════════════════════════════════════════════════════════════════
print('[5] Generating heatmaps...')

HEAT_MODES = [m for m in ['car_only','car_for_all','car_microtrans',
                           'car_transit','transit_for_all'] if m in modes]

for mode_name in HEAT_MODES:
    W_by_seg = MODES.get(mode_name, {})
    if not W_by_seg: continue
    mode_res = res_df[res_df['mode']==mode_name]

    fig, axes = plt.subplots(1, len(KEY_P), figsize=(8*len(KEY_P), 7))
    fig.suptitle(f'Per-Hex Accessibility Heatmap — {MODE_LABELS.get(mode_name, mode_name)}\n'
                 f'Rockingham County, NC  |  Allocation BLP  |  Uniform $o_j$',
                 fontsize=19, y=1.04)

    for idx, p_val in enumerate(KEY_P):
        ax = axes[idx]
        row = mode_res[mode_res['P']==p_val]
        if len(row) == 0:
            ax.set_title(f'P={p_val}\n(no data)', fontsize=16); ax.set_axis_off(); continue

        sel_str   = row.iloc[0].get('selected_stops', '')
        sel_hexes = [s for s in str(sel_str).split('|') if s and s != 'nan']

        A_hex    = per_hex_A(sel_hexes, W_by_seg)
        hex_plot = hex_gdf[['hex_id','geometry']].copy()
        hex_plot['A_i'] = hex_plot['hex_id'].map(A_hex).fillna(0)
        vmax = max(hex_plot['A_i'].quantile(0.98), 0.01)

        hex_plot.plot(column='A_i', ax=ax, cmap=HEAT_CMAP,
                      vmin=0, vmax=vmax, edgecolor='white', linewidth=0.2)

        if sel_hexes:
            sel_geom = hex_plot[hex_plot['hex_id'].isin(sel_hexes)]
            sel_geom.plot(ax=ax, facecolor='none', edgecolor='black',
                          linewidth=3.0, hatch='...', zorder=9)
            sc = J_df[J_df['hex_id'].isin(sel_hexes)]
            ax.scatter(sc['lon'], sc['lat'], s=35, c='black',
                       marker='o', zorder=10)

        if hospitals is not None:
            ax.scatter(hospitals['longitude'], hospitals['latitude'],
                       c='blue', marker='+', s=120, linewidths=2.2, zorder=8)

        sm = plt.cm.ScalarMappable(cmap=HEAT_CMAP,
                                    norm=mcolors.Normalize(0, vmax))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, label='A_i')
        cb.ax.tick_params(labelsize=13)
        cb.set_label('A_i', fontsize=15)

        at_val = row.iloc[0]['A_tilde']
        ax.set_title(f'$P={p_val}$  |  $\\tilde{{\\mathcal{{A}}}}={at_val:.3f}$',
                     fontsize=17, pad=6)
        ax.set_axis_off()

        vals = hex_plot['A_i']
        ax.text(0.02, 0.02,
                f'min={vals.min():.2f}\nmax={vals.max():.2f}\nmean={vals.mean():.2f}',
                transform=ax.transAxes, fontsize=12, va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    out = f'{OUTPUTS}/primal_heatmap_{mode_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 5b. Heatmaps — no-car restricted accessibility at P=4, 12, 26, 50
#     Same purple-to-green colormap, same larger fonts, but A_i is computed
#     with the car-owning segments zeroed out (NO_CAR_PI), matching the
#     no_car_accessibility_results.csv metric.
# ══════════════════════════════════════════════════════════════════════════════
print('[5b] Generating no-car restricted heatmaps...')

NOCAR_HEAT_MODES = [m for m in ['car_for_all','car_microtrans','car_transit',
                                 'transit_for_all'] if m in modes]

for mode_name in NOCAR_HEAT_MODES:
    W_by_seg = MODES.get(mode_name, {})
    if not W_by_seg: continue
    mode_res = res_df[res_df['mode']==mode_name]
    mode_nocar_res = nocar_df[nocar_df['mode']==mode_name]

    fig, axes = plt.subplots(1, len(KEY_P), figsize=(8*len(KEY_P), 7))
    fig.suptitle(f'Per-Hex Accessibility Heatmap (No-Car Population) — '
                 f'{MODE_LABELS.get(mode_name, mode_name)}\n'
                 f'Rockingham County, NC  |  Car-owning segments excluded',
                 fontsize=19, y=1.04)

    for idx, p_val in enumerate(KEY_P):
        ax = axes[idx]
        row = mode_res[mode_res['P']==p_val]
        nocar_row = mode_nocar_res[mode_nocar_res['P']==p_val]
        if len(row) == 0:
            ax.set_title(f'P={p_val}\n(no data)', fontsize=16); ax.set_axis_off(); continue

        sel_str   = row.iloc[0].get('selected_stops', '')
        sel_hexes = [s for s in str(sel_str).split('|') if s and s != 'nan']

        A_hex    = per_hex_A(sel_hexes, W_by_seg, PI_weights=NO_CAR_PI)
        hex_plot = hex_gdf[['hex_id','geometry']].copy()
        hex_plot['A_i'] = hex_plot['hex_id'].map(A_hex).fillna(0)
        vmax = max(hex_plot['A_i'].quantile(0.98), 0.01)

        hex_plot.plot(column='A_i', ax=ax, cmap=HEAT_CMAP,
                      vmin=0, vmax=vmax, edgecolor='white', linewidth=0.2)

        if sel_hexes:
            sel_geom = hex_plot[hex_plot['hex_id'].isin(sel_hexes)]
            sel_geom.plot(ax=ax, facecolor='none', edgecolor='black',
                          linewidth=3.0, hatch='...', zorder=9)
            sc = J_df[J_df['hex_id'].isin(sel_hexes)]
            ax.scatter(sc['lon'], sc['lat'], s=35, c='black',
                       marker='o', zorder=10)

        if hospitals is not None:
            ax.scatter(hospitals['longitude'], hospitals['latitude'],
                       c='blue', marker='+', s=120, linewidths=2.2, zorder=8)

        sm = plt.cm.ScalarMappable(cmap=HEAT_CMAP,
                                    norm=mcolors.Normalize(0, vmax))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, label='A_i')
        cb.ax.tick_params(labelsize=13)
        cb.set_label('A_i', fontsize=15)

        at_val = float(nocar_row['A_tilde_nocar'].iloc[0]) if len(nocar_row) else float('nan')
        ax.set_title(f'$P={p_val}$  |  $\\tilde{{\\mathcal{{A}}}}_{{\\mathrm{{no\\ car}}}}={at_val:.3f}$',
                     fontsize=17, pad=6)
        ax.set_axis_off()

        vals = hex_plot['A_i']
        ax.text(0.02, 0.02,
                f'min={vals.min():.2f}\nmax={vals.max():.2f}\nmean={vals.mean():.2f}',
                transform=ax.transAxes, fontsize=12, va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    out = f'{OUTPUTS}/no_car_heatmap_{mode_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 6. No-car OPTIMIZED comparison (linked from no_car_facility_optimization.py)
#    That script re-solves the facility-location problem with the objective
#    restricted to the no-car population, instead of reusing the population-
#    wide site selection (which is what no_car_accessibility_results.csv /
#    section [2b] does). Run no_car_facility_optimization.py first; this
#    section is skipped gracefully if its output isn't found yet.
#
#    Two lines per mode:
#      "rescored"  -- joint-optimum sites (same selected_stops as the main
#                     pipeline), scored only on the no-car population
#                     (= no_car_accessibility_results.csv, already in nocar_df)
#      "optimized" -- sites chosen specifically to maximize no-car
#                     accessibility (= no_car_optimized_results.csv)
#    The gap between the two lines is the accessibility the no-car
#    population is losing because sites are chosen jointly with car owners.
# ══════════════════════════════════════════════════════════════════════════════
NOCAR_OPT_CSV = f'{OUTPUTS}/no_car_optimized_results.csv'

if os.path.exists(NOCAR_OPT_CSV):
    print('[6] No-car OPTIMIZED comparison (found no_car_optimized_results.csv)...')
    nocar_opt_df = pd.read_csv(NOCAR_OPT_CSV)

    compare_modes = [m for m in NOCAR_DRAW_ORDER
                      if m in nocar_opt_df['mode'].unique()
                      and m in nocar_df['mode'].unique()]

    fig, ax = plt.subplots(figsize=(13, 8))
    for mode_name in compare_modes:
        color = MODE_COLORS.get(mode_name, '#888')

        rescored = nocar_df[nocar_df['mode']==mode_name].sort_values('P')
        ax.plot(rescored['P'], rescored['A_tilde_nocar'],
                color=color, lw=2.0, ls='--', alpha=0.55, marker='o', markersize=5,
                label=f'{MODE_LABELS.get(mode_name, mode_name)} (joint-optimum sites)')

        optimized = nocar_opt_df[nocar_opt_df['mode']==mode_name].sort_values('P')
        ax.plot(optimized['P'], optimized['A_tilde_nocar_opt'],
                color=color, lw=2.6, ls='-', marker='^', markersize=7,
                label=f'{MODE_LABELS.get(mode_name, mode_name)} (no-car-optimized sites)')

    ax.set_xlabel('Number of clinic stops ($P$)', fontsize=14)
    ax.set_ylabel(r'$\tilde{\mathcal{A}}_{\mathrm{no\ car}}$ (no-car population only)', fontsize=14)
    ax.set_title('No-Car Accessibility: Joint-Optimum Sites vs No-Car-Optimized Sites\n'
                 'Rockingham County, NC  |  dashed = current sites, rescored  |  '
                 'solid = sites re-chosen for the no-car population',
                 fontsize=13, pad=8)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.13),
              ncol=2, framealpha=0.95)
    plt.tight_layout()
    out_compare = f'{OUTPUTS}/no_car_optimized_vs_rescored.png'
    plt.savefig(out_compare, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_compare}')

    # Print the recoverable-accessibility gap at each P, per mode
    print('\n  Accessibility recoverable by optimizing sites for the no-car')
    print('  population instead of reusing the joint-optimum sites:')
    for mode_name in compare_modes:
        rescored  = nocar_df[nocar_df['mode']==mode_name].set_index('P')['A_tilde_nocar']
        optimized = nocar_opt_df[nocar_opt_df['mode']==mode_name].set_index('P')['A_tilde_nocar_opt']
        common_P  = sorted(set(rescored.index) & set(optimized.index))
        for p in common_P:
            gap = optimized[p] - rescored[p]
            print(f'    {mode_name:22s} P={p:>3d}  rescored={rescored[p]:.3f}  '
                  f'optimized={optimized[p]:.3f}  gap={gap:+.3f}')

    # ──────────────────────────────────────────────────────────────────────
    # 6b. Heatmaps using the ACTUAL no-car-optimized site selections.
    #     Section [5b]'s heatmaps reuse the joint-optimum `selected_stops`
    #     column by design (that's what makes the "same sites, rescored"
    #     comparison meaningful). These heatmaps instead plot
    #     `selected_stops_nocar_opt`, so the star markers show where the
    #     optimizer actually chose to move facilities once the objective was
    #     restricted to the no-car population.
    # ──────────────────────────────────────────────────────────────────────
    print('\n[6b] Generating heatmaps for the no-car-OPTIMIZED site selections...')

    for mode_name in compare_modes:
        W_by_seg = MODES.get(mode_name, {})
        if not W_by_seg: continue
        mode_opt_res = nocar_opt_df[nocar_opt_df['mode']==mode_name]
        opt_P_values = sorted(mode_opt_res['P'].unique())
        if not opt_P_values: continue

        fig, axes = plt.subplots(1, len(opt_P_values), figsize=(8*len(opt_P_values), 7))
        if len(opt_P_values) == 1:
            axes = [axes]
        fig.suptitle(f'Per-Hex Accessibility Heatmap (No-Car OPTIMIZED Sites) — '
                     f'{MODE_LABELS.get(mode_name, mode_name)}\n'
                     f'Rockingham County, NC  |  Sites re-chosen to maximize '
                     f'no-car accessibility specifically',
                     fontsize=19, y=1.04)

        for idx, p_val in enumerate(opt_P_values):
            ax = axes[idx]
            row = mode_opt_res[mode_opt_res['P']==p_val]
            if len(row) == 0:
                ax.set_title(f'P={p_val}\n(no data)', fontsize=16); ax.set_axis_off(); continue

            sel_str   = row.iloc[0].get('selected_stops_nocar_opt', '')
            sel_hexes = [s for s in str(sel_str).split('|') if s and s != 'nan']

            A_hex    = per_hex_A(sel_hexes, W_by_seg, PI_weights=NO_CAR_PI)
            hex_plot = hex_gdf[['hex_id','geometry']].copy()
            hex_plot['A_i'] = hex_plot['hex_id'].map(A_hex).fillna(0)
            vmax = max(hex_plot['A_i'].quantile(0.98), 0.01)

            hex_plot.plot(column='A_i', ax=ax, cmap=HEAT_CMAP,
                          vmin=0, vmax=vmax, edgecolor='white', linewidth=0.2)

            if sel_hexes:
                sel_geom = hex_plot[hex_plot['hex_id'].isin(sel_hexes)]
                sel_geom.plot(ax=ax, facecolor='none', edgecolor='black',
                          linewidth=3.0, hatch='...', zorder=9)
                sc = J_df[J_df['hex_id'].isin(sel_hexes)]
                ax.scatter(sc['lon'], sc['lat'], s=35, c='black',
                           marker='o', zorder=10)

            if hospitals is not None:
                ax.scatter(hospitals['longitude'], hospitals['latitude'],
                           c='blue', marker='+', s=120, linewidths=2.2, zorder=8)

            sm = plt.cm.ScalarMappable(cmap=HEAT_CMAP,
                                        norm=mcolors.Normalize(0, vmax))
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, label='A_i')
            cb.ax.tick_params(labelsize=13)
            cb.set_label('A_i', fontsize=15)

            at_val = float(row['A_tilde_nocar_opt'].iloc[0])
            ax.set_title(f'$P={p_val}$  |  $\\tilde{{\\mathcal{{A}}}}_{{\\mathrm{{no\\ car}}}}={at_val:.3f}$',
                         fontsize=17, pad=6)
            ax.set_axis_off()

            vals = hex_plot['A_i']
            ax.text(0.02, 0.02,
                    f'min={vals.min():.2f}\nmax={vals.max():.2f}\nmean={vals.mean():.2f}',
                    transform=ax.transAxes, fontsize=12, va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        out = f'{OUTPUTS}/no_car_optimized_heatmap_{mode_name}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f'  Saved: {out}')
else:
    print('[6] Skipped -- run no_car_facility_optimization.py first to generate '
          f'{NOCAR_OPT_CSV}')

print('\nAll done.')