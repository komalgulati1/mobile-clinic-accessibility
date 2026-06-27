"""
02_generate_heatmaps.py
=======================
Choropleth accessibility heatmaps for all 5 mode scenarios at P=4.

Outputs (→ outputs/maps/):
    heatmap_a_car_only.png
    heatmap_b_car_transit.png
    heatmap_c_car_microtrans.png
    heatmap_d_car_for_all.png
    heatmap_e_transit_only.png
    heatmap_all5_scenarios.png        ← combined 5-panel for slides

Run after 01_accessibility_pipeline.py:
    python scripts/02_generate_heatmaps.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pulp import (LpProblem, LpMaximize, LpVariable,
                  lpSum, LpBinary, value, PULP_CBC_CMD)

from config import (
    TT_CAR, TT_TRANSIT, TT_TRANSIT_SINGLE,
    HEX_CENTROIDS, HEX_GRID, HOSPITALS_CSV,
    POIS, ACS_POP, ACS_VEH,
    ALPHA_CAR, BETA_CAR, ALPHA_TRANSIT, BETA_TRANSIT,
    CANDIDATE_NAICS, PI,
    COUNTY_TOTAL, COUNTY_ELDERLY, ZV_SHARE,
    W_SHAPE, W_SCALE, D_SHAPE, D_SCALE,
    P_VALS, OUT_MAPS,
)

warnings.filterwarnings('ignore')

P = 4   # show heatmaps for P=4

_TT_TRANSIT = TT_TRANSIT if TT_TRANSIT.exists() else TT_TRANSIT_SINGLE

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading data...')
hexes     = pd.read_csv(HEX_CENTROIDS)
tt_car_df = pd.read_csv(TT_CAR)
tt_tr_df  = pd.read_csv(_TT_TRANSIT)
pois      = pd.read_csv(POIS, low_memory=False)
hex_gdf   = gpd.read_file(HEX_GRID).to_crs('EPSG:4326')
hospitals = pd.read_csv(HOSPITALS_CSV) if HOSPITALS_CSV.exists() else None

hex_ids = hexes['hex_id'].tolist()
N_HEX   = len(hex_ids)
base    = hex_gdf[['hex_id', 'geometry']].copy()

# ── o_j ───────────────────────────────────────────────────────────────────────
pois['naics_code'] = pd.to_numeric(pois['naics_code'], errors='coerce')
cand = pois[pois['naics_code'].isin(CANDIDATE_NAICS)].copy()
poi_gdf = gpd.GeoDataFrame(cand,
    geometry=gpd.points_from_xy(cand['longitude'], cand['latitude']),
    crs='EPSG:4326')
if 'index_right' in poi_gdf.columns:
    poi_gdf = poi_gdf.drop(columns=['index_right'])
hr = hex_gdf[['hex_id', 'geometry']].copy().reset_index(drop=True)
joined = gpd.sjoin(poi_gdf, hr, how='left', predicate='within')
joined['total_visit_2025'] = pd.to_numeric(
    joined['total_visit_2025'], errors='coerce').fillna(0)
hv = joined.groupby('hex_id')['total_visit_2025'].sum().reset_index()
hv.columns = ['hex_id', 'total_visits']
hv = hv[hv['total_visits'] > 0]
hv['o_j'] = hv['total_visits'] / hv['total_visits'].max()
J_df  = hv[hv['hex_id'].isin(hex_ids)].copy()
J_ids = J_df['hex_id'].tolist()
o_j   = dict(zip(J_df['hex_id'], J_df['o_j']))
print(f'  |J| = {len(J_ids)}')

# ── n_ig (area-proportional) ──────────────────────────────────────────────────
n_per_hex  = COUNTY_TOTAL  / N_HEX
e_per_hex  = COUNTY_ELDERLY / N_HEX
ne_per_hex = n_per_hex - e_per_hex
n_ig = {
    hid: {'ec':   e_per_hex*(1-ZV_SHARE),
           'enc':  e_per_hex*ZV_SHARE,
           'nec':  ne_per_hex*(1-ZV_SHARE),
           'nenc': ne_per_hex*ZV_SHARE}
    for hid in hex_ids
}

# ── Impedance matrices ─────────────────────────────────────────────────────────
def impedance(t, a, b):
    t = np.asarray(t, dtype=float); w = np.zeros_like(t)
    m = np.isfinite(t) & (t > 0); w[m] = np.exp(a*(t[m]**b))
    return w

car_df = tt_car_df[tt_car_df['to_hex'].isin(J_ids)].copy()
car_df['w'] = impedance(car_df['travel_time_min'].values, ALPHA_CAR, BETA_CAR)
W_car = {}; raw_tt = {}
for row in car_df.itertuples(index=False):
    W_car.setdefault(row.from_hex, {})[row.to_hex] = row.w
    raw_tt.setdefault(row.from_hex, {})[row.to_hex] = (
        float(row.travel_time_min) if pd.notna(row.travel_time_min) else np.nan)

tr_df = tt_tr_df[tt_tr_df['to_hex'].isin(J_ids)].copy()
tr_df['tt'] = tr_df['travel_time_min'].replace(0, np.nan)
tr_df['w']  = impedance(tr_df['tt'].values, ALPHA_TRANSIT, BETA_TRANSIT)
W_transit = {}
for row in tr_df.itertuples(index=False):
    W_transit.setdefault(row.from_hex, {})[row.to_hex] = row.w

# Microtransit mean impedance
np.random.seed(42)
d_mean = np.clip(np.random.weibull(D_SHAPE, 200)*D_SCALE, 1.0, 2.0).mean()
w_mean = (np.random.weibull(W_SHAPE, 200)*W_SCALE).mean()
W_mt = {i: {j: float(np.exp(ALPHA_CAR*((d_mean*raw_tt.get(i,{}).get(j,np.nan)+w_mean)**BETA_CAR)))
             if np.isfinite(raw_tt.get(i,{}).get(j,np.nan)) else 0.0
             for j in J_ids}
        for i in hex_ids}
W_zero = {i: {j: 0.0 for j in J_ids} for i in hex_ids}

A_max = sum(PI[g]*n_ig[i][g]*sum(o_j.get(j,0)*W_car.get(i,{}).get(j,0)
                                   for j in J_ids)
            for i in hex_ids for g in PI)

# ── Solve BLP helper ───────────────────────────────────────────────────────────
def solve_scenario(W_by_seg, P, name):
    c = {j: sum(PI[g]*n_ig[i][g]*o_j.get(j,0)*Wm.get(i,{}).get(j,0)
                for i in hex_ids for g, Wm in W_by_seg.items())
         if o_j.get(j,0) > 0 else 0.0
         for j in J_ids}
    prob = LpProblem(name, LpMaximize)
    y    = {j: LpVariable(f'y_{j}', cat=LpBinary) for j in J_ids}
    prob += lpSum(c[j]*y[j] for j in J_ids)
    prob += lpSum(y[j] for j in J_ids) == P
    prob.solve(PULP_CBC_CMD(msg=0))
    sel = [j for j in J_ids if value(y[j]) > 0.5]
    At  = value(prob.objective) / A_max
    print(f'  {name} P={P}: Ã={At:.4f}')
    return sel, At

def per_hex_Ai(selected, W_by_seg):
    return {i: sum(PI[g]*n_ig[i][g]*sum(o_j.get(j,0)*Wm.get(i,{}).get(j,0)
                                         for j in selected)
                   for g, Wm in W_by_seg.items())
            for i in hex_ids}

SCENARIOS = {
    'a_car_only':       ({'ec': W_car,     'enc': W_zero,    'nec': W_car,    'nenc': W_zero},
                         'YlOrRd', '(a) Car Only — Baseline'),
    'b_car_transit':    ({'ec': W_car,     'enc': W_transit, 'nec': W_car,    'nenc': W_transit},
                         'Blues',   '(b) Car + Transit (SKAT)'),
    'c_car_microtrans': ({'ec': W_car,     'enc': W_mt,      'nec': W_car,    'nenc': W_mt},
                         'YlGnBu',  '(c) Car + Microtransit (RCATS)'),
    'd_car_for_all':    ({'ec': W_car,     'enc': W_car,     'nec': W_car,    'nenc': W_car},
                         'Greens',  '(d) Car for All — Upper Bound'),
    'e_transit_only':   ({'ec': W_transit, 'enc': W_transit, 'nec': W_transit,'nenc': W_transit},
                         'PuRd',    '(e) Transit Only — Lower Bound'),
}

print('\nSolving scenarios at P=4...')
sc_results = {}
for name, (Wb, cmap, title) in SCENARIOS.items():
    sel, At = solve_scenario(Wb, P, name)
    Ai = per_hex_Ai(sel, Wb)
    sc_results[name] = {'sel': sel, 'At': At, 'Ai': Ai, 'cmap': cmap, 'title': title}
    base[f'Ai_{name}'] = base['hex_id'].map(Ai).fillna(0)

# Common scale
all_vals = [v for nm in sc_results for v in sc_results[nm]['Ai'].values()]
vmax = np.percentile(all_vals, 98)

# ── Plot helpers ───────────────────────────────────────────────────────────────
def add_stops(ax, sel, ms=120):
    sc = hexes[hexes['hex_id'].isin(sel)]
    ax.scatter(sc['lon'], sc['lat'], s=ms, c='black', marker='*', zorder=6,
               label=f'Clinic stop (P={P})')
    if hospitals is not None:
        ax.scatter(hospitals['longitude'], hospitals['latitude'],
                   s=80, c='crimson', marker='+', linewidths=2.5, zorder=6,
                   label='Hospital')

LEG = [
    Line2D([0],[0], marker='*', color='w', markerfacecolor='black',
           markersize=12, label=f'Selected clinic stop (P={P})'),
    Line2D([0],[0], marker='+', color='crimson', markersize=10,
           markeredgewidth=2.5, linestyle='None', label='Hospital'),
]

# ── Individual maps ────────────────────────────────────────────────────────────
print('\nGenerating individual maps...')
for nm, res in sc_results.items():
    fig, ax = plt.subplots(figsize=(8, 7))
    base.plot(column=f'Ai_{nm}', ax=ax, cmap=res['cmap'],
              vmin=0, vmax=vmax, edgecolor='white', linewidth=0.3)
    add_stops(ax, res['sel'])
    sm = plt.cm.ScalarMappable(cmap=res['cmap'],
         norm=mcolors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(r'Absolute accessibility $A_i$', fontsize=9)
    ax.set_title(f'{res["title"]}\n'
                 r'$\tilde{\mathcal{A}}$' + f' = {res["At"]:.3f}',
                 fontsize=10, pad=6)
    ax.set_axis_off()
    ax.legend(handles=LEG, loc='lower left', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out = OUT_MAPS / f'heatmap_{nm}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')

# ── Combined 5-panel ───────────────────────────────────────────────────────────
print('\nGenerating combined 5-panel...')
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for idx, (nm, res) in enumerate(sc_results.items()):
    ax = axes[idx]
    base.plot(column=f'Ai_{nm}', ax=ax, cmap=res['cmap'],
              vmin=0, vmax=vmax, edgecolor='white', linewidth=0.3)
    add_stops(ax, res['sel'], ms=90)
    sm = plt.cm.ScalarMappable(cmap=res['cmap'],
         norm=mcolors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f'{res["title"]}\n'
                 r'$\tilde{\mathcal{A}}$' + f' = {res["At"]:.3f}',
                 fontsize=9.5, pad=4)
    ax.set_axis_off()

axes[5].set_axis_off()
axes[5].legend(handles=LEG + [
    Patch(color='white', label=''),
    Patch(color='white', label=r'Common scale: $A_i$ (absolute)'),
    Patch(color='white', label=f'P={P} stops per scenario'),
], loc='center', fontsize=9.5, framealpha=0, borderaxespad=0)

fig.suptitle(f'Accessibility Heatmaps — Rockingham County, NC  (P={P} stops)\n'
             r'$A_i$ = per-hexagon absolute accessibility',
             fontsize=12, y=1.01)
plt.tight_layout()
out = OUT_MAPS / 'heatmap_all5_scenarios.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {out}')
print('\nNext: python scripts/03_generate_difference_maps.py')
