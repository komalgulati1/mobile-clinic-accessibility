"""
01_accessibility_pipeline.py
============================
P-Median accessibility pipeline for mobile clinic stop selection.

Model: Maximize normalized gravity-based accessibility Ã = A(y)/A_max
       subject to selecting exactly P stops from candidate set J.

Scenarios computed:
    (a) Car only          — baseline
    (b) Car + Transit     — SKAT fixed-route
    (c) Car + Microtransit— RCATS, Monte Carlo over Weibull δ and W
    (d) Car for all       — theoretical upper bound
    (e) Transit only      — theoretical lower bound

Outputs (→ outputs/plots/ and outputs/tables/):
    accessibility_step_plot.png
    accessibility_results.csv
    selected_stops_P4.csv

Usage:
    python scripts/01_accessibility_pipeline.py

References:
    Verma et al. (2025) Scientific Data — impedance parameters
    Yang & Gao (2025) TRA Part A doi:10.1016/j.tra.2025.104504
    Hu & Xu (2025) TRF Part F doi:10.1016/j.trf.2025.103363
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import time, json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pulp import (LpProblem, LpMaximize, LpVariable,
                  lpSum, LpBinary, value, PULP_CBC_CMD)

from config import (
    TT_CAR, TT_TRANSIT, TT_TRANSIT_SINGLE,
    HEX_CENTROIDS, HEX_GRID, HOSP_SUMMARY,
    POIS, ACS_POP, ACS_VEH,
    ALPHA_CAR, BETA_CAR, ALPHA_TRANSIT, BETA_TRANSIT,
    CANDIDATE_NAICS, PI,
    COUNTY_TOTAL, COUNTY_ELDERLY, ZV_SHARE,
    W_SHAPE, W_SCALE, D_SHAPE, D_SCALE, N_MC,
    P_VALS, OUT_PLOTS, OUT_TABS,
)

warnings.filterwarnings('ignore')

# Fall back to single-run transit if averaged file not yet generated
if not TT_TRANSIT.exists():
    print(f'  NOTE: {TT_TRANSIT.name} not found — '
          f'falling back to {TT_TRANSIT_SINGLE.name}')
    print('  Run scripts/00_rerun_transit.py for better transit estimates.')
    _TT_TRANSIT = TT_TRANSIT_SINGLE
else:
    _TT_TRANSIT = TT_TRANSIT

# ── Timing helpers ─────────────────────────────────────────────────────────────
timings = {}
def tick(label):
    timings[label] = {'start': time.time()}
    print(f'\n[START] {label}')
def tock(label):
    elapsed = time.time() - timings[label]['start']
    timings[label]['elapsed'] = elapsed
    print(f'[DONE]  {label}  →  {elapsed:.2f}s')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data
# ══════════════════════════════════════════════════════════════════════════════
tick('Step 1: Load data')

hexes    = pd.read_csv(HEX_CENTROIDS)
tt_car   = pd.read_csv(TT_CAR)
tt_trans = pd.read_csv(_TT_TRANSIT)
pois     = pd.read_csv(POIS, low_memory=False)
hosp_sum = pd.read_csv(HOSP_SUMMARY)

with open(ACS_POP) as f: acs_pop = json.load(f)
with open(ACS_VEH) as f: acs_veh = json.load(f)

hex_ids = hexes['hex_id'].tolist()
N_HEX   = len(hex_ids)
print(f'  Hexagons:        {N_HEX}')
print(f'  Car OD pairs:    {len(tt_car):,}')
print(f'  Transit OD pairs:{len(tt_trans):,}')
print(f'  POI records:     {len(pois):,}')

tock('Step 1: Load data')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Compute o_j (POI visit weights per candidate stop hexagon)
# ══════════════════════════════════════════════════════════════════════════════
tick('Step 2: Compute o_j (stop attractiveness weights)')

pois['naics_code'] = pd.to_numeric(pois['naics_code'], errors='coerce')
cand_pois = pois[pois['naics_code'].isin(CANDIDATE_NAICS)].copy()
print(f'  Candidate POIs after NAICS filter: {len(cand_pois)}')

hex_gdf = gpd.read_file(HEX_GRID).to_crs('EPSG:4326')
hex_col = next(
    (c for c in hex_gdf.columns if 'hex' in c.lower() or c == 'h3_index'),
    hex_gdf.columns[0])

poi_gdf = gpd.GeoDataFrame(
    cand_pois,
    geometry=gpd.points_from_xy(cand_pois['longitude'], cand_pois['latitude']),
    crs='EPSG:4326')
if 'index_right' in poi_gdf.columns:
    poi_gdf = poi_gdf.drop(columns=['index_right'])

hex_right = hex_gdf[[hex_col, 'geometry']].copy().reset_index(drop=True)
joined    = gpd.sjoin(poi_gdf, hex_right, how='left', predicate='within')
joined['total_visit_2025'] = pd.to_numeric(
    joined['total_visit_2025'], errors='coerce').fillna(0)

hv = joined.groupby(hex_col)['total_visit_2025'].sum().reset_index()
hv.columns = ['hex_id', 'total_visits']
hv = hv[hv['total_visits'] > 0]
max_visits = hv['total_visits'].max()
hv['o_j'] = hv['total_visits'] / max_visits

J_df  = hv[hv['hex_id'].isin(hex_ids)].copy()
J_ids = J_df['hex_id'].tolist()
o_j   = dict(zip(J_df['hex_id'], J_df['o_j']))

print(f'  Candidate stop hexagons |J|: {len(J_ids)}')
print(f'  o_j range: [{J_df["o_j"].min():.4f}, {J_df["o_j"].max():.4f}]')

tock('Step 2: Compute o_j (stop attractiveness weights)')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Compute n_ig (four population segments per hexagon)
# ══════════════════════════════════════════════════════════════════════════════
tick('Step 3: Compute n_ig (population segments)')

def parse_census_json(data, cols):
    """Parse Census API list-of-lists format (header row first)."""
    headers = data[0]
    rows = []
    for rec in data[1:]:
        d = dict(zip(headers, rec))
        result = {k: float(d.get(v, 0) or 0) for k, v in cols.items()}
        result['geoid'] = (str(d.get('state','')) + str(d.get('county','')) +
                           str(d.get('tract','')) + str(d.get('block group','')))
        rows.append(result)
    return rows

elderly_cols = [f'B01001_{n:03d}E' for n in
                [20,21,22,23,24,25,44,45,46,47,48,49]]
pop_records = parse_census_json(
    acs_pop, {'total': 'B01003_001E',
               **{f'eld_{c}': c for c in elderly_cols}})
for r in pop_records:
    r['elderly'] = sum(r.pop(f'eld_{c}', 0) for c in elderly_cols)

veh_records = parse_census_json(
    acs_veh, {'hh_total': 'B08201_001E', 'zero_veh': 'B08201_002E'})

# Use ACS-parsed totals; fall back to config constants if parsing fails
_county_total   = max(sum(r['total']   for r in pop_records), COUNTY_TOTAL)
_county_elderly = max(sum(r['elderly'] for r in pop_records), COUNTY_ELDERLY)
_county_zveh    = sum(r['zero_veh']  for r in veh_records)
_county_hh      = max(sum(r['hh_total'] for r in veh_records), 1)
_zv_share       = max(_county_zveh / _county_hh, ZV_SHARE)

n_per_hex  = _county_total   / N_HEX
e_per_hex  = _county_elderly / N_HEX
ne_per_hex = n_per_hex - e_per_hex

n_ig = {
    hid: {
        'ec':   e_per_hex  * (1 - _zv_share),
        'enc':  e_per_hex  * _zv_share,
        'nec':  ne_per_hex * (1 - _zv_share),
        'nenc': ne_per_hex * _zv_share,
    }
    for hid in hex_ids
}

print(f'  County total: {_county_total:.0f}  |  '
      f'Elderly: {_county_elderly:.0f}  |  ZV share: {_zv_share:.4f}')
print(f'  n per hex: {n_per_hex:.1f}  (area-proportional approximation)')

tock('Step 3: Compute n_ig (population segments)')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Impedance matrices (car and transit)
# ══════════════════════════════════════════════════════════════════════════════
tick('Step 4: Build impedance matrices')

def impedance(t, alpha, beta):
    """Power-exponential impedance: exp(α·tᵝ). Returns 0 for unreachable."""
    t = np.asarray(t, dtype=float)
    w = np.zeros_like(t)
    m = np.isfinite(t) & (t > 0)
    w[m] = np.exp(alpha * (t[m] ** beta))
    return w

# Car
car_sub = tt_car[tt_car['to_hex'].isin(J_ids)].copy()
car_sub['w'] = impedance(car_sub['travel_time_min'].values, ALPHA_CAR, BETA_CAR)
W_car  = {}
raw_tt = {}
for row in car_sub.itertuples(index=False):
    W_car.setdefault(row.from_hex, {})[row.to_hex] = row.w
    raw_tt.setdefault(row.from_hex, {})[row.to_hex] = (
        float(row.travel_time_min) if pd.notna(row.travel_time_min) else np.nan)

# Transit
tr_sub = tt_trans[tt_trans['to_hex'].isin(J_ids)].copy()
tr_sub['tt'] = tr_sub['travel_time_min'].replace(0, np.nan)
tr_sub['w']  = impedance(tr_sub['tt'].values, ALPHA_TRANSIT, BETA_TRANSIT)
W_transit = {}
for row in tr_sub.itertuples(index=False):
    W_transit.setdefault(row.from_hex, {})[row.to_hex] = row.w

W_zero = {i: {j: 0.0 for j in J_ids} for i in hex_ids}

print(f'  Car pairs (→J): {len(car_sub):,}  |  '
      f'Transit pairs with w>0: {(tr_sub["w"]>0).sum():,}')
print(f'  Transit reachable: '
      f'{tr_sub["tt"].notna().sum():,} / {len(tr_sub):,} '
      f'({tr_sub["tt"].notna().mean()*100:.1f}%)')

tock('Step 4: Build impedance matrices')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Compute A_max (theoretical upper bound)
# ══════════════════════════════════════════════════════════════════════════════
tick('Step 5: Compute A_max')

A_max = sum(
    PI[g] * n_ig[i][g] *
    sum(o_j.get(j, 0) * W_car.get(i, {}).get(j, 0) for j in J_ids)
    for i in hex_ids for g in PI
)
print(f'  A_max (car-for-all, all stops open) = {A_max:.4f}')

tock('Step 5: Compute A_max')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — BLP solver and coefficient builder
# ══════════════════════════════════════════════════════════════════════════════
def build_coeff(W_by_seg):
    """
    Pre-compute linear objective coefficient c_j for each candidate stop j.
    c_j = Σᵢ Σ_g π_g · n_ig · o_j · w_ij^m
    """
    c = {}
    for j in J_ids:
        oj = o_j.get(j, 0)
        if oj == 0:
            c[j] = 0.0
            continue
        total = sum(
            PI[g] * n_ig[i][g] * oj * Wm.get(i, {}).get(j, 0)
            for i in hex_ids
            for g, Wm in W_by_seg.items()
        )
        c[j] = total
    return c

def solve_blp(c_j, P, name):
    """Solve the cardinality-constrained BLP: max Σⱼ cⱼ yⱼ s.t. Σⱼ yⱼ = P."""
    t0   = time.time()
    prob = LpProblem(name, LpMaximize)
    y    = {j: LpVariable(f'y_{j}', cat=LpBinary) for j in J_ids}
    prob += lpSum(c_j[j] * y[j] for j in J_ids)
    prob += lpSum(y[j] for j in J_ids) == P
    prob.solve(PULP_CBC_CMD(msg=0))
    sel = [j for j in J_ids if value(y[j]) > 0.5]
    At  = value(prob.objective) / A_max if A_max > 0 else 0.0
    print(f'    P={P}: Ã={At:.4f}  '
          f'selected={sel[:2]}{"..." if len(sel)>2 else ""}  '
          f'({time.time()-t0:.2f}s)')
    return {'P': P, 'objective': value(prob.objective),
            'A_tilde': At, 'selected_stops': sel,
            'solve_time_s': time.time()-t0}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Solve all scenarios for P = 1..4
# ══════════════════════════════════════════════════════════════════════════════
tick('Step 7: Solve all scenarios')

SCENARIO_DEFS = {
    'car_only':      {'ec': W_car,     'enc': W_zero,    'nec': W_car,    'nenc': W_zero},
    'car_for_all':   {'ec': W_car,     'enc': W_car,     'nec': W_car,    'nenc': W_car},
    'transit_only':  {'ec': W_transit, 'enc': W_transit, 'nec': W_transit,'nenc': W_transit},
    'car_transit':   {'ec': W_car,     'enc': W_transit, 'nec': W_car,    'nenc': W_transit},
}

all_results = {}
for sc, Wb in SCENARIO_DEFS.items():
    print(f'\n  === {sc} ===')
    coeff = build_coeff(Wb)
    all_results[sc] = [solve_blp(coeff, P, f'{sc}_P{P}') for P in P_VALS]

tock('Step 7: Solve all scenarios')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Car + Microtransit (Monte Carlo)
# ══════════════════════════════════════════════════════════════════════════════
tick('Step 8: Car + Microtransit (Monte Carlo)')

np.random.seed(42)
delta_samp = np.clip(np.random.weibull(D_SHAPE, N_MC) * D_SCALE, 1.0, 2.0)
W_samp     = np.random.weibull(W_SHAPE, N_MC) * W_SCALE

print(f'  δ: mean={delta_samp.mean():.3f} ± {delta_samp.std():.3f}')
print(f'  W: mean={W_samp.mean():.3f} ± {W_samp.std():.3f} min')
print(f'  Running {N_MC} MC draws...')

t0 = time.time()
mc_At = {P: [] for P in P_VALS}

for mc_idx in range(N_MC):
    d, W = delta_samp[mc_idx], W_samp[mc_idx]

    c_mt = {}
    for j in J_ids:
        oj = o_j.get(j, 0)
        if oj == 0:
            c_mt[j] = 0.0
            continue
        total = 0.0
        for i in hex_ids:
            seg = n_ig[i]
            # Car-owning: car impedance
            w_c = W_car.get(i, {}).get(j, 0)
            total += (PI['ec']*seg['ec'] + PI['nec']*seg['nec']) * oj * w_c
            # Car-free: microtransit t_mt = δ·t_car + W
            t_c = raw_tt.get(i, {}).get(j, np.nan)
            if np.isfinite(t_c):
                w_mt = np.exp(ALPHA_CAR * ((d*t_c + W) ** BETA_CAR))
                total += (PI['enc']*seg['enc'] + PI['nenc']*seg['nenc']) * oj * w_mt
        c_mt[j] = total

    for P in P_VALS:
        prob = LpProblem(f'mc_{mc_idx}_P{P}', LpMaximize)
        y    = {j: LpVariable(f'y_{j}', cat=LpBinary) for j in J_ids}
        prob += lpSum(c_mt[j]*y[j] for j in J_ids)
        prob += lpSum(y[j] for j in J_ids) == P
        prob.solve(PULP_CBC_CMD(msg=0))
        mc_At[P].append(value(prob.objective) / A_max)

    if (mc_idx+1) % 50 == 0:
        print(f'  {mc_idx+1}/{N_MC} ({time.time()-t0:.1f}s)')

mc_stats = {
    P: {'mean': np.mean(mc_At[P]),
        'p5':   np.percentile(mc_At[P], 5),
        'p95':  np.percentile(mc_At[P], 95)}
    for P in P_VALS
}
for P in P_VALS:
    s = mc_stats[P]
    print(f'  P={P}: mean={s["mean"]:.4f}  '
          f'[p5={s["p5"]:.4f}, p95={s["p95"]:.4f}]')

tock('Step 8: Car + Microtransit (Monte Carlo)')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Five-scenario step plot
# ══════════════════════════════════════════════════════════════════════════════
tick('Step 9: Step plot')

A_car   = [r['A_tilde'] for r in all_results['car_only']]
A_all   = [r['A_tilde'] for r in all_results['car_for_all']]
A_tr    = [r['A_tilde'] for r in all_results['transit_only']]
A_ct    = [r['A_tilde'] for r in all_results['car_transit']]
A_cmt   = [mc_stats[P]['mean'] for P in P_VALS]
A_p5    = [mc_stats[P]['p5']   for P in P_VALS]
A_p95   = [mc_stats[P]['p95']  for P in P_VALS]

fig, ax = plt.subplots(figsize=(10, 6.5))

def step_line(ax, P_vals, A_vals, color, label, ls='-', lw=2, zo=3):
    x = [P_vals[0]-0.5] + [p+0.5 for p in P_vals]
    ax.step(x, [0]+A_vals, where='post', color=color,
            linestyle=ls, linewidth=lw, label=label, zorder=zo)
    ax.scatter(P_vals, A_vals, color=color, s=55, zorder=zo+1)

# MC uncertainty band
xb = [P_VALS[0]-0.5] + [p+0.5 for p in P_VALS]
ax.fill_between(xb, [0]+A_p5, [0]+A_p95,
                step='post', alpha=0.18, color='#8b5e3c')

step_line(ax, P_VALS, A_all,  '#2e7d32', 'Car for all (upper bound)',   ls='--', lw=2)
step_line(ax, P_VALS, A_ct,   '#0d6eab', 'Car + Transit (SKAT)',        ls='-.', lw=2.2)
step_line(ax, P_VALS, A_cmt,  '#8b5e3c', 'Car + Microtransit (mean)',   ls='-',  lw=2.2)
step_line(ax, P_VALS, A_car,  '#1a3a5c', 'Car only (baseline)',         ls='-',  lw=2.5)
step_line(ax, P_VALS, A_tr,   '#8b1a1a', 'Transit only (lower bound)',  ls=':',  lw=1.8)

mc_patch = mpatches.Patch(color='#8b5e3c', alpha=0.3,
    label=f'Car + Microtransit (5th–95th pct, n={N_MC})')

for p, a in zip(P_VALS, A_car):
    ax.annotate(f'{a:.3f}', (p, a), xytext=(p+0.06, a-0.020),
                fontsize=8.5, color='#1a3a5c', fontweight='bold')
for p, a in zip(P_VALS, A_ct):
    ax.annotate(f'{a:.3f}', (p, a), xytext=(p+0.06, a+0.007),
                fontsize=8, color='#0d6eab')
for p, a in zip(P_VALS, A_cmt):
    ax.annotate(f'{a:.3f}', (p, a), xytext=(p+0.06, a+0.018),
                fontsize=8, color='#8b5e3c')

ax.set_xlabel('Number of clinic stops (P)', fontsize=12)
ax.set_ylabel(r'Normalized accessibility $\tilde{\mathcal{A}}$' + '\n'
              '(fraction of theoretical maximum)', fontsize=11)
ax.set_title('Mobile Clinic Stop Selection — Accessibility vs. Number of Stops\n'
             'Rockingham County, NC  |  Five Mode Scenarios', fontsize=12, pad=10)
ax.set_xlim(0.4, 4.8);  ax.set_ylim(-0.01, 0.62)
ax.set_xticks(P_VALS);  ax.grid(axis='y', alpha=0.3, linestyle='--')

handles, labels = ax.get_legend_handles_labels()
handles.insert(3, mc_patch); labels.insert(3, mc_patch.get_label())
ax.legend(handles, labels, loc='upper left', fontsize=9.5, framealpha=0.92)
ax.text(0.985, 0.04,
        r'$\tilde{\mathcal{A}}=\mathcal{A}(\mathbf{y})/\mathcal{A}^{\max}$'
        '\n' r'$\mathcal{A}^{\max}$: car-for-all, all stops open',
        transform=ax.transAxes, fontsize=8, ha='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

plt.tight_layout()
out_plot = OUT_PLOTS / 'accessibility_step_plot.png'
plt.savefig(out_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {out_plot}')

tock('Step 9: Step plot')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Save results tables
# ══════════════════════════════════════════════════════════════════════════════
tick('Step 10: Save results tables')

rows = []
for sc, res_list in all_results.items():
    for r in res_list:
        rows.append({'scenario': sc,
                     'P': r['P'],
                     'A_tilde': r['A_tilde'],
                     'objective': r['objective'],
                     'solve_time_s': r['solve_time_s']})
for P in P_VALS:
    s = mc_stats[P]
    rows.append({'scenario': 'car_microtransit_mean', 'P': P,
                 'A_tilde': s['mean'], 'A_tilde_p5': s['p5'],
                 'A_tilde_p95': s['p95']})

results_df = pd.DataFrame(rows)
out_csv = OUT_TABS / 'accessibility_results.csv'
results_df.to_csv(out_csv, index=False)
print(f'  Results:  {out_csv}')

# P=4 car-only selected stops
best_stops = all_results['car_only'][-1]['selected_stops']
stops_df = pd.DataFrame({'hex_id': best_stops})
stops_df = stops_df.merge(
    pd.DataFrame({'hex_id': J_ids, 'o_j': [o_j[j] for j in J_ids]}),
    on='hex_id').merge(hexes, on='hex_id', how='left')
out_stops = OUT_TABS / 'selected_stops_P4.csv'
stops_df.to_csv(out_stops, index=False)
print(f'  Stops:    {out_stops}')

tock('Step 10: Save results tables')

# ── Timing summary ─────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TIMING SUMMARY')
print('='*60)
total = 0
for step, t in timings.items():
    e = t.get('elapsed', 0); total += e
    print(f'  {step:<48} {e:>7.2f}s')
print(f'  {"TOTAL":<48} {total:>7.2f}s')
print('='*60)

print('\nRESULTS SUMMARY')
print('='*60)
print(results_df[results_df['scenario'].isin(
    ['car_only','car_transit','car_for_all','transit_only'])
    ].to_string(index=False))
print('\nNext: python scripts/02_generate_heatmaps.py')
