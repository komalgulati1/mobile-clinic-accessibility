"""
03_generate_difference_maps.py
==============================
Difference maps and absolute accessibility heatmaps.

Shows:
    1. Absolute A_i values for all 5 scenarios on a common scale
    2. (b) Car+Transit minus (a) Car Only   — where transit adds accessibility
    3. (c) Car+Microtransit minus (a) Car Only — where microtransit adds
    4. Slide-ready 3-panel: baseline + 2 difference maps

Outputs (→ outputs/maps/):
    heatmap_absolute_all5.png
    heatmap_difference_maps.png
    heatmap_slide_figure.png    ← use this for slide 13

Run after 01_accessibility_pipeline.py:
    python scripts/03_generate_difference_maps.py
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
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pulp import (LpProblem, LpMaximize, LpVariable,
                  lpSum, LpBinary, value, PULP_CBC_CMD)

from config import (
    TT_CAR, TT_TRANSIT, TT_TRANSIT_SINGLE,
    HEX_CENTROIDS, HEX_GRID, HOSPITALS_CSV,
    POIS, ACS_POP,
    ALPHA_CAR, BETA_CAR, ALPHA_TRANSIT, BETA_TRANSIT,
    CANDIDATE_NAICS, PI,
    COUNTY_TOTAL, COUNTY_ELDERLY, ZV_SHARE,
    W_SHAPE, W_SCALE, D_SHAPE, D_SCALE,
    NAVY, GOLD,
    OUT_MAPS,
)

warnings.filterwarnings('ignore')

P = 4
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

# ── n_ig ──────────────────────────────────────────────────────────────────────
n_per_hex  = COUNTY_TOTAL  / N_HEX
e_per_hex  = COUNTY_ELDERLY / N_HEX
ne_per_hex = n_per_hex - e_per_hex
n_ig = {hid: {'ec': e_per_hex*(1-ZV_SHARE), 'enc': e_per_hex*ZV_SHARE,
               'nec': ne_per_hex*(1-ZV_SHARE), 'nenc': ne_per_hex*ZV_SHARE}
        for hid in hex_ids}

# ── Impedance ─────────────────────────────────────────────────────────────────
def imp(t, a, b):
    t = np.asarray(t, dtype=float); w = np.zeros_like(t)
    m = np.isfinite(t) & (t > 0); w[m] = np.exp(a*(t[m]**b))
    return w

car_df = tt_car_df[tt_car_df['to_hex'].isin(J_ids)].copy()
car_df['w'] = imp(car_df['travel_time_min'].values, ALPHA_CAR, BETA_CAR)
W_car = {}; raw_tt = {}
for row in car_df.itertuples(index=False):
    W_car.setdefault(row.from_hex, {})[row.to_hex] = row.w
    raw_tt.setdefault(row.from_hex, {})[row.to_hex] = (
        float(row.travel_time_min) if pd.notna(row.travel_time_min) else np.nan)

tr_df = tt_tr_df[tt_tr_df['to_hex'].isin(J_ids)].copy()
tr_df['tt'] = tr_df['travel_time_min'].replace(0, np.nan)
tr_df['w']  = imp(tr_df['tt'].values, ALPHA_TRANSIT, BETA_TRANSIT)
W_transit = {}
for row in tr_df.itertuples(index=False):
    W_transit.setdefault(row.from_hex, {})[row.to_hex] = row.w

np.random.seed(42)
d_mean = np.clip(np.random.weibull(D_SHAPE, 200)*D_SCALE, 1.0, 2.0).mean()
w_mean = (np.random.weibull(W_SHAPE, 200)*W_SCALE).mean()
W_mt = {i: {j: float(np.exp(ALPHA_CAR*((d_mean*raw_tt.get(i,{}).get(j,np.nan)+w_mean)**BETA_CAR)))
             if np.isfinite(raw_tt.get(i,{}).get(j,np.nan)) else 0.0
             for j in J_ids}
        for i in hex_ids}
W_zero = {i: {j: 0.0 for j in J_ids} for i in hex_ids}

A_max = sum(PI[g]*n_ig[i][g]*sum(o_j.get(j,0)*W_car.get(i,{}).get(j,0) for j in J_ids)
            for i in hex_ids for g in PI)

# ── Solve BLP ─────────────────────────────────────────────────────────────────
def solve(W_by_seg, P, name):
    c = {j: sum(PI[g]*n_ig[i][g]*o_j.get(j,0)*Wm.get(i,{}).get(j,0)
                for i in hex_ids for g, Wm in W_by_seg.items())
         if o_j.get(j,0) > 0 else 0.0 for j in J_ids}
    prob = LpProblem(name, LpMaximize)
    y    = {j: LpVariable(f'y_{j}', cat=LpBinary) for j in J_ids}
    prob += lpSum(c[j]*y[j] for j in J_ids)
    prob += lpSum(y[j] for j in J_ids) == P
    prob.solve(PULP_CBC_CMD(msg=0))
    sel = [j for j in J_ids if value(y[j]) > 0.5]
    At  = value(prob.objective) / A_max
    print(f'  {name} P={P}: Ã={At:.4f}')
    return sel, At

def per_hex(sel, W_by_seg):
    return {i: sum(PI[g]*n_ig[i][g]*sum(o_j.get(j,0)*Wm.get(i,{}).get(j,0) for j in sel)
                   for g, Wm in W_by_seg.items())
            for i in hex_ids}

print('\nSolving scenarios...')
SCENARIOS = {
    'a_car_only':       {'ec': W_car,     'enc': W_zero,    'nec': W_car,    'nenc': W_zero},
    'b_car_transit':    {'ec': W_car,     'enc': W_transit, 'nec': W_car,    'nenc': W_transit},
    'c_car_microtrans': {'ec': W_car,     'enc': W_mt,      'nec': W_car,    'nenc': W_mt},
    'd_car_for_all':    {'ec': W_car,     'enc': W_car,     'nec': W_car,    'nenc': W_car},
    'e_transit_only':   {'ec': W_transit, 'enc': W_transit, 'nec': W_transit,'nenc': W_transit},
}

SC_META = {
    'a_car_only':       ('YlOrRd', '(a) Car Only — Baseline'),
    'b_car_transit':    ('Blues',   '(b) Car + Transit (SKAT)'),
    'c_car_microtrans': ('YlGnBu',  '(c) Car + Microtransit (RCATS)'),
    'd_car_for_all':    ('Greens',  '(d) Car for All — Upper Bound'),
    'e_transit_only':   ('PuRd',    '(e) Transit Only — Lower Bound'),
}

res = {}
for nm, Wb in SCENARIOS.items():
    sel, At = solve(Wb, P, nm)
    Ai = per_hex(sel, Wb)
    res[nm] = {'sel': sel, 'At': At, 'Ai': Ai}
    base[f'abs_{nm}'] = base['hex_id'].map(Ai).fillna(0)

# Difference maps
base['diff_tr'] = base['hex_id'].map(
    {i: res['b_car_transit']['Ai'][i] - res['a_car_only']['Ai'][i]
     for i in hex_ids}).fillna(0)
base['diff_mt'] = base['hex_id'].map(
    {i: res['c_car_microtrans']['Ai'][i] - res['a_car_only']['Ai'][i]
     for i in hex_ids}).fillna(0)
base['diff_mt_vs_tr'] = base['hex_id'].map(
    {i: res['c_car_microtrans']['Ai'][i] - res['b_car_transit']['Ai'][i]
     for i in hex_ids}).fillna(0)

# ── Common scale ───────────────────────────────────────────────────────────────
abs_vals = [v for nm in SCENARIOS for v in res[nm]['Ai'].values()]
vmax_abs = np.percentile(abs_vals, 98)

# ── Helpers ────────────────────────────────────────────────────────────────────
def add_stops(ax, sel, ms=120):
    sc = hexes[hexes['hex_id'].isin(sel)]
    ax.scatter(sc['lon'], sc['lat'], s=ms, c='black', marker='*', zorder=6,
               label=f'Clinic stop (P={P})')
    if hospitals is not None:
        ax.scatter(hospitals['longitude'], hospitals['latitude'],
                   s=80, c='crimson', marker='+', linewidths=2.5,
                   zorder=6, label='Hospital')

STD_LEG = [
    Line2D([0],[0], marker='*', color='w', markerfacecolor='black',
           markersize=12, label=f'Selected clinic stop (P={P})'),
    Line2D([0],[0], marker='+', color='crimson', markersize=10,
           markeredgewidth=2.5, linestyle='None', label='Hospital'),
]

# ── FIGURE 1 — Absolute accessibility, 5 scenarios ────────────────────────────
print('\nFigure 1: Absolute accessibility...')
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 11))
axes1 = axes1.flatten()

for idx, (nm, (cmap, title)) in enumerate(SC_META.items()):
    ax = axes1[idx]
    At = res[nm]['At']
    base.plot(column=f'abs_{nm}', ax=ax, cmap=cmap,
              vmin=0, vmax=vmax_abs, edgecolor='white', linewidth=0.3)
    add_stops(ax, res[nm]['sel'], ms=90)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, vmax_abs))
    sm.set_array([])
    cb = fig1.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f'{title}\n'
                 r'$\tilde{\mathcal{A}}$' + f' = {At:.3f}  |  '
                 f'Mean $A_i$ = {base[f"abs_{nm}"].mean():.4f}',
                 fontsize=9.5, pad=5)
    ax.set_axis_off()

axes1[5].set_axis_off()
axes1[5].legend(handles=STD_LEG + [
    Patch(color='white', label=''),
    Patch(color='white', label=r'$A_i$ = absolute (not normalized)'),
    Patch(color='white', label='Common scale across all panels'),
], loc='center', fontsize=9.5, framealpha=0)
fig1.suptitle('Absolute Accessibility — All 5 Scenarios  |  Rockingham County, NC',
              fontsize=12, y=1.005)
plt.tight_layout()
out1 = OUT_MAPS / 'heatmap_absolute_all5.png'
plt.savefig(out1, dpi=150, bbox_inches='tight');  plt.close()
print(f'  Saved: {out1}')

# ── FIGURE 2 — Three difference maps ──────────────────────────────────────────
print('Figure 2: Difference maps...')
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

diff_configs = [
    ('diff_tr',       'b_car_transit',    '(b)−(a): Transit Gain\n(Car+Transit minus Car Only)',
     'RdBu', 'Transit accessibility gain\n(Blue = gain, Red = loss)'),
    ('diff_mt',       'c_car_microtrans', '(c)−(a): Microtransit Gain\n(Car+Microtransit minus Car Only)',
     'RdBu', 'Microtransit accessibility gain\n(Blue = gain, Red = loss)'),
    ('diff_mt_vs_tr', 'c_car_microtrans', '(c)−(b): Microtransit vs Transit\n(which mode helps car-free people more)',
     'PiYG', 'Microtransit advantage over transit\n(Green = MT better, Purple = transit better)'),
]

for idx, (col, sc_name, title, cmap, cbar_label) in enumerate(diff_configs):
    ax = axes2[idx]
    vals = base[col].values
    vabs = max(abs(np.percentile(vals, 1)), abs(np.percentile(vals, 99)), 1e-9)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    base.plot(column=col, ax=ax, cmap=cmap, norm=norm,
              edgecolor='white', linewidth=0.3)

    pos_vals = vals[vals > 0]
    if len(pos_vals) > 0:
        gainers = base[base[col] > np.percentile(pos_vals, 90)]
        ax.scatter(gainers.geometry.centroid.x, gainers.geometry.centroid.y,
                   s=22, c=GOLD, marker='o', zorder=5, alpha=0.8,
                   label='Top 10% gainers')

    add_stops(ax, res[sc_name]['sel'], ms=100)

    n_pos = (base[col] > 1e-9).sum()
    mean_g = pos_vals.mean() if len(pos_vals) > 0 else 0
    ax.set_title(f'{title}\n'
                 f'{n_pos} hexagons benefit  |  mean gain = {mean_g:.5f}',
                 fontsize=9.5, pad=5)
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig2.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=20)
    cb.set_label(cbar_label, fontsize=8);  cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(y=0.5, color='black', linewidth=0.8, alpha=0.4)

fig2.legend(handles=STD_LEG + [
    Line2D([0],[0], marker='o', color=GOLD, markersize=8,
           linestyle='None', label='Top 10% gain hexagons'),
], loc='lower center', ncol=4, fontsize=9, framealpha=0.9,
   bbox_to_anchor=(0.5, -0.07))
fig2.suptitle('Accessibility Difference Maps  |  Rockingham County, NC\n'
              'What transit and microtransit add over car-only baseline at P=4',
              fontsize=11, y=1.02)
plt.tight_layout()
out2 = OUT_MAPS / 'heatmap_difference_maps.png'
plt.savefig(out2, dpi=150, bbox_inches='tight');  plt.close()
print(f'  Saved: {out2}')

# ── FIGURE 3 — Slide-ready 3-panel ────────────────────────────────────────────
print('Figure 3: Slide-ready 3-panel...')
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6.5))

# Panel 1: Car only baseline
ax = axes3[0]
base.plot(column='abs_a_car_only', ax=ax, cmap='YlOrRd',
          vmin=0, vmax=vmax_abs, edgecolor='white', linewidth=0.3)
add_stops(ax, res['a_car_only']['sel'])
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=Normalize(0, vmax_abs))
sm.set_array([])
cb = fig3.colorbar(sm, ax=ax, shrink=0.65, pad=0.02, aspect=20)
cb.set_label(r'Absolute accessibility $A_i$', fontsize=8)
ax.set_title('(a) Car Only — Baseline\n'
             r'$\tilde{\mathcal{A}}$' + f' = {res["a_car_only"]["At"]:.3f}',
             fontsize=11, pad=6)
ax.set_axis_off()

# Panels 2 & 3: difference maps
for idx, (col, sc_name, sc_label, mode_label, cmap) in enumerate([
    ('diff_tr', 'b_car_transit',    '(b)−(a)',
     'Transit Gain over Car Only', 'RdBu'),
    ('diff_mt', 'c_car_microtrans', '(c)−(a)',
     'Microtransit Gain over Car Only', 'RdBu'),
]):
    ax   = axes3[idx+1]
    vals = base[col].values
    vabs = max(abs(np.percentile(vals, 1)), abs(np.percentile(vals, 99)), 1e-9)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    base.plot(column=col, ax=ax, cmap=cmap, norm=norm,
              edgecolor='white', linewidth=0.3)
    pos_vals = vals[vals > 0]
    if len(pos_vals) > 0:
        gainers = base[base[col] > np.percentile(pos_vals, 90)]
        ax.scatter(gainers.geometry.centroid.x, gainers.geometry.centroid.y,
                   s=25, c=GOLD, marker='o', zorder=5, alpha=0.8)
    add_stops(ax, res[sc_name]['sel'], ms=100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig3.colorbar(sm, ax=ax, shrink=0.65, pad=0.02, aspect=20)
    cb.set_label(f'{mode_label}\n(Blue = gain, Red = loss)', fontsize=8)
    n_pos = (base[col] > 1e-9).sum()
    ax.set_title(f'{sc_label}: {mode_label}\n'
                 f'{n_pos} hexagons benefit from mode coordination',
                 fontsize=11, pad=6)
    ax.set_axis_off()

fig3.legend(handles=STD_LEG + [
    Line2D([0],[0], marker='o', color=GOLD, markersize=9,
           linestyle='None', label='Top 10% gain hexagons'),
], loc='lower center', ncol=3, fontsize=9.5, framealpha=0.9,
   bbox_to_anchor=(0.5, -0.05))
fig3.suptitle('Mobile Clinic Accessibility — Baseline and Mode Coordination Gains\n'
              'Rockingham County, NC  |  P=4 clinic stops',
              fontsize=12, y=1.02)
plt.tight_layout()
out3 = OUT_MAPS / 'heatmap_slide_figure.png'
plt.savefig(out3, dpi=150, bbox_inches='tight');  plt.close()
print(f'  Saved: {out3}')

# ── Summary ────────────────────────────────────────────────────────────────────
print('\nDIFFERENCE MAP SUMMARY')
print('='*55)
for col, label in [('diff_tr','Transit gain (b)-(a)'),
                   ('diff_mt','Microtransit gain (c)-(a)'),
                   ('diff_mt_vs_tr','MT vs Transit (c)-(b)')]:
    v = base[col].values
    pos = v[v > 0]
    print(f'{label}:')
    print(f'  Hexagons with gain > 0: {len(pos)}/{N_HEX}')
    print(f'  Max gain: {v.max():.6f}')
    print(f'  Mean gain (>0): {pos.mean():.6f}' if len(pos) else '  No gainers')
