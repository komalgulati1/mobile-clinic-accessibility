"""
primal_accessibility_viz.py
============================
Standalone visualization for primal accessibility results.
Reads outputs from primal_accessibility.py — run that first.

Inputs (from outputs/):
    primal_accessibility_results_full.csv  — A_tilde + selected stops per mode per P
    mayodan_first_P.csv                    — linear greedy results for comparison

Outputs (to outputs/):
    primal_accessibility_step_plot.png     — step plot all modes
    primal_vs_linear_comparison.png        — allocation vs linear car only
    primal_heatmap_{mode}.png              — per-hex heatmap at P=4,12,26,50
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
SHAPE_W   = 1.79;      SCALE_W  = 4.2
import math
W_MEAN = SCALE_W * math.gamma(1 + 1/SHAPE_W)

PI       = {'ec': 0.85, 'enc': 0.90, 'nec': 0.55, 'nenc': 0.60}
K_SPARSE = 15
KEY_P    = [4, 12, 26, 50]

MODE_COLORS = {
    'car_only':          '#e74c3c',
    'car_for_all':       '#2ecc71',
    'transit_only':      '#3498db',
    'transit_for_all':   '#1abc9c',
    'car_transit':       '#9b59b6',
    'car_microtrans':    '#e67e22',
}
MODE_LABELS = {
    'car_only':          'Car only',
    'car_for_all':       'Car for all',
    'transit_only':      'Transit only',
    'transit_for_all':   'Transit for all',
    'car_transit':       'Car + Transit (SKAT)',
    'car_microtrans':    'Car + Microtransit',
}

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
W_mc = {}
for i in hex_ids:
    W_mc[i] = {}
    for j in J_ids:
        t_c = raw_tt_car.get(i,{}).get(j, np.nan)
        if np.isfinite(t_c):
            W_mc[i][j] = np.exp(ALPHA_MC * ((t_c + W_MEAN) ** BETA_MC))

MODES = {
    'car_only':        {'ec': W_car,     'enc': {},        'nec': W_car,     'nenc': {}},
    'car_for_all':     {'ec': W_car,     'enc': W_car,     'nec': W_car,     'nenc': W_car},
    'car_microtrans':  {'ec': W_car,     'enc': W_mc,      'nec': W_car,     'nenc': W_mc},
}
if W_transit:
    MODES['transit_only']    = {'ec': {},        'enc': W_transit, 'nec': {},        'nenc': W_transit}
    MODES['transit_for_all'] = {'ec': W_transit, 'enc': W_transit, 'nec': W_transit, 'nenc': W_transit}
    MODES['car_transit']     = {'ec': W_car,     'enc': W_transit, 'nec': W_car,     'nenc': W_transit}

# k nearest per hex
nearest_j = {}
for i in hex_ids:
    scored = [(j, W_car.get(i,{}).get(j,0)) for j in J_ids]
    scored.sort(key=lambda x: -x[1])
    nearest_j[i] = [j for j,_ in scored[:K_SPARSE]]

def per_hex_A(sel_hexes, W_by_seg):
    sel_set = set(sel_hexes)
    A_hex = {}
    for i in hex_ids:
        total = 0.0
        for g, Wm in W_by_seg.items():
            if not Wm or n_ig[i][g] <= 0: continue
            avail = [j for j in nearest_j[i] if j in sel_set]
            if not avail: continue
            best_w = max(Wm.get(i,{}).get(j,0) for j in avail)
            total += PI[g] * n_ig[i][g] * best_w
        A_hex[i] = total
    return A_hex

print('  Spatial data loaded.')

# ══════════════════════════════════════════════════════════════════════════════
# 3. Step plot — all modes
#    Fix: car modes are nearly identical (max gap <0.003) so they overlap.
#    Solution: draw order puts car_only on top, use distinct line styles,
#    stagger right-side annotations, add zoom panel for P=1..20.
# ══════════════════════════════════════════════════════════════════════════════
print('[3] Step plot...')

# Draw transit modes first, car cluster last so car_only (red solid) is on top
DRAW_ORDER = ['transit_for_all', 'transit_only', 'car_transit',
              'car_for_all', 'car_microtrans', 'car_only']

MODE_LS = {
    'car_only':        '-',    # solid   -- most prominent
    'car_for_all':     '--',   # dashed
    'car_microtrans':  ':',    # dotted
    'transit_only':    '-',
    'transit_for_all': '-',
    'car_transit':     '-',
}
MODE_LW = {
    'car_only':        3.0,
    'car_for_all':     2.0,
    'car_microtrans':  2.0,
    'transit_only':    2.2,
    'transit_for_all': 2.2,
    'car_transit':     2.2,
}
# Vertical offsets for right-side annotations so they do not stack
# Car cluster all annotate ~0.982, transit_only ~0.958, transit_for_all ~0.926
ANNOT_OFFSET = {
    'car_for_all':     +0.013,
    'car_microtrans':   0.000,
    'car_only':        -0.013,
    'car_transit':      0.000,
    'transit_only':     0.000,
    'transit_for_all':  0.000,
}

fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                         gridspec_kw={'width_ratios': [2.2, 1]})
ax      = axes[0]   # main step plot
ax_zoom = axes[1]   # zoom on P=1..15 to show car mode separation

for mode_name in DRAW_ORDER:
    if mode_name not in modes:
        continue
    sub   = res_df[res_df['mode']==mode_name].sort_values('P')
    color = MODE_COLORS.get(mode_name, '#888')
    label = MODE_LABELS.get(mode_name, mode_name)
    lw    = MODE_LW.get(mode_name, 2.2)
    ls    = MODE_LS.get(mode_name, '-')

    ax.step(sub['P'], sub['A_tilde'], where='post',
            color=color, lw=lw, ls=ls, label=label)
    ax_zoom.step(sub['P'], sub['A_tilde'], where='post',
                 color=color, lw=lw, ls=ls, label=label)

    # Staggered right-side annotation
    final_val = float(sub['A_tilde'].iloc[-1])
    y_ann     = final_val + ANNOT_OFFSET.get(mode_name, 0)
    ax.text(50.8, y_ann, f'{final_val:.4f}',
            fontsize=7.5, va='center', color=color,
            fontweight='bold' if mode_name == 'car_only' else 'normal')

# ── Main plot settings ────────────────────────────────────────────────────────
ax.set_xlabel('Number of clinic stops (P)', fontsize=12)
ax.set_ylabel(r'$\tilde{\mathcal{A}}$ (normalized accessibility)', fontsize=12)
ax.set_title('Primal Accessibility vs P — Allocation BLP\n'
             'Rockingham County, NC  |  Uniform $o_j$  |  '
             'Correct submodular formulation',
             fontsize=11, pad=6)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim(1, 55)
ax.set_ylim(0, 1.05)
ax.annotate('Car modes (solid/dashed/dotted)\ndiffer by <0.003 — see zoom panel',
            xy=(0.015, 0.73), xycoords='axes fraction', fontsize=7.5, color='#555',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

# ── Zoom panel: P=1..15, show car cluster separation ─────────────────────────
ax_zoom.set_xlim(1, 15)
ax_zoom.set_ylim(0.22, 0.90)
ax_zoom.set_xlabel('P', fontsize=11)
ax_zoom.set_ylabel(r'$\tilde{\mathcal{A}}$', fontsize=11)
ax_zoom.set_title('Zoom: P = 1–15\n(car mode separation)',
                  fontsize=10, pad=5)
ax_zoom.grid(alpha=0.3)
ax_zoom.legend(fontsize=8, loc='lower right')

# Annotate P=4 values in zoom for the three car modes
for mode_name, yoffset in [('car_for_all', +0.018),
                             ('car_microtrans', 0.000),
                             ('car_only', -0.018)]:
    if mode_name not in modes: continue
    sub  = res_df[res_df['mode']==mode_name]
    p4   = sub[sub['P']==4]
    if len(p4):
        val = float(p4['A_tilde'].values[0])
        ax_zoom.annotate(
            f'{MODE_LABELS[mode_name]}: {val:.4f}',
            xy=(4, val),
            xytext=(6.5, val + yoffset),
            fontsize=7, color=MODE_COLORS[mode_name],
            arrowprops=dict(arrowstyle='->', color=MODE_COLORS[mode_name], lw=0.8))

plt.tight_layout()
out = f'{OUTPUTS}/primal_accessibility_step_plot.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 4. Comparison: allocation vs linear (car only)
# ══════════════════════════════════════════════════════════════════════════════
print('[4] Comparison plot (allocation vs linear)...')
if os.path.exists(LINEAR_CSV):
    linear_df = pd.read_csv(LINEAR_CSV)
    car_sub   = res_df[res_df['mode']=='car_only'].sort_values('P')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Primal Accessibility: Allocation BLP vs Original Linear Formulation\n'
                 'Car only  |  Uniform o_j  |  Rockingham County, NC', fontsize=11, y=1.02)

    ax = axes[0]
    ax.step(car_sub['P'], car_sub['A_tilde'], where='post',
            color='#e74c3c', lw=2.5, label='Allocation BLP (correct, submodular)')
    ax.step(linear_df['P'], linear_df['A_tilde'], where='post',
            color='#95a5a6', lw=2, ls='--', label='Linear greedy (original, incorrect)')
    ax.set_xlabel('P', fontsize=11)
    ax.set_ylabel(r'$\tilde{\mathcal{A}}$', fontsize=11)
    ax.set_title('A_tilde vs P', fontsize=10, pad=5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(1,51); ax.set_ylim(0)

    ax2 = axes[1]
    alloc_v  = car_sub.set_index('P')['A_tilde']
    linear_v = linear_df.set_index('P')['A_tilde']
    common_P = sorted(set(alloc_v.index) & set(linear_v.index))
    diff     = [alloc_v[p] - linear_v[p] for p in common_P]
    colors_d = ['#e74c3c' if d > 0 else '#3498db' for d in diff]
    ax2.bar(common_P, diff, color=colors_d, edgecolor='white', linewidth=0.3)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xlabel('P', fontsize=11)
    ax2.set_ylabel('Allocation minus Linear', fontsize=10)
    ax2.set_title('Difference: Allocation BLP - Linear\nRed = allocation higher  Blue = linear higher',
                  fontsize=10, pad=5)
    ax2.grid(axis='y', alpha=0.3); ax2.set_xlim(0.3, 51)

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

HEAT_MODES = [m for m in ['car_only','car_for_all','car_microtrans','car_transit',
                           'transit_for_all','transit_only'] if m in modes]

for mode_name in HEAT_MODES:
    W_by_seg = MODES.get(mode_name, {})
    if not W_by_seg: continue
    mode_res = res_df[res_df['mode']==mode_name]

    fig, axes = plt.subplots(1, len(KEY_P), figsize=(6*len(KEY_P), 6))
    fig.suptitle(f'Per-Hex Accessibility Heatmap — {MODE_LABELS.get(mode_name, mode_name)}\n'
                 f'Rockingham County, NC  |  Allocation BLP  |  Uniform o_j',
                 fontsize=11, y=1.02)

    for idx, p_val in enumerate(KEY_P):
        ax = axes[idx]
        row = mode_res[mode_res['P']==p_val]
        if len(row) == 0:
            ax.set_title(f'P={p_val}\n(no data)'); ax.set_axis_off(); continue

        sel_str   = row.iloc[0].get('selected_stops', '')
        sel_hexes = [s for s in str(sel_str).split('|') if s and s != 'nan']

        A_hex    = per_hex_A(sel_hexes, W_by_seg)
        hex_plot = hex_gdf[['hex_id','geometry']].copy()
        hex_plot['A_i'] = hex_plot['hex_id'].map(A_hex).fillna(0)
        vmax = max(hex_plot['A_i'].quantile(0.98), 0.01)

        hex_plot.plot(column='A_i', ax=ax, cmap='RdYlGn',
                      vmin=0, vmax=vmax, edgecolor='white', linewidth=0.2)

        if sel_hexes:
            sc = J_df[J_df['hex_id'].isin(sel_hexes)]
            ax.scatter(sc['lon'], sc['lat'], s=120, c='black',
                       marker='*', zorder=7, label=f'{len(sel_hexes)} stops')

        if hospitals is not None:
            ax.scatter(hospitals['longitude'], hospitals['latitude'],
                       c='blue', marker='+', s=100, linewidths=2, zorder=8)

        sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                                    norm=mcolors.Normalize(0, vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, label='A_i')

        at_val = row.iloc[0]['A_tilde']
        ax.set_title(f'P={p_val}  |  A_tilde={at_val:.3f}', fontsize=10, pad=4)
        ax.legend(fontsize=7, loc='lower left')
        ax.set_axis_off()

        vals = hex_plot['A_i']
        ax.text(0.02, 0.02,
                f'min={vals.min():.2f}\nmax={vals.max():.2f}\nmean={vals.mean():.2f}',
                transform=ax.transAxes, fontsize=7, va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    out = f'{OUTPUTS}/primal_heatmap_{mode_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {out}')

print('\nAll done.')