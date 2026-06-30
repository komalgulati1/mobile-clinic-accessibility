"""
sensitivity_analysis_acs.py
============================
ACS Sampling Uncertainty Sensitivity Analysis — Rockingham County, NC
Komal Gulati, CR2C2 / NC A&T

Addresses the zone-shape ACS error critique (doi:10.1016/j.trc.2024.104759):
H3 hexagons do not align with ACS block group boundaries, so area-proportional
interpolation inherits block-group-level sampling errors. This script propagates
those errors through the primal accessibility BLP using three approaches:

1. MOE Bracket: solve BLP at n_ig - MOE, n_ig (baseline), n_ig + MOE
   Gives a simple ±1 MOE accessibility band.

2. Monte Carlo: draw R replicates of n_ig from N(estimate, SE) per hex,
   solve BLP for each replicate, report mean ± 1.96*std band.

3. Summary: report how much A_tilde changes across the uncertainty band
   at key P values and which stops change under perturbation.

Runs for car_only mode (dominant mode) at P in {1, 4, 8, 12, 26, 50}.
Monte Carlo uses R=200 replicates for tractability (BLP is fast at single P).

Inputs (from outputs/):
    hexagon_elderly_SE.csv      — elderly_est, SE_elderly_vrt, MOE_pct per hex
    hexagon_zveh_share.csv      — zv_share, SE_zv_share per hex
    rockingham_hex_centroids_r7.csv
    rockingham_hex_r7.gpkg
    rockingham_POIs.csv
    tt_car_r7.csv
    acs_rockingham_with_moe.json
    acs_vehicles_tract_with_moe.json

Outputs (to outputs/):
    sensitivity_moe_bracket.csv         — A_tilde at -MOE, baseline, +MOE
    sensitivity_mc_results.csv          — A_tilde per replicate per P
    sensitivity_mc_summary.csv          — mean, std, CI95 per P
    sensitivity_stop_stability.csv      — stop selection overlap across replicates
    sensitivity_moe_band_plot.png       — step plot with ±MOE shading
    sensitivity_mc_band_plot.png        — step plot with 95% CI shading
    sensitivity_moe_pct_plot.png        — MOE% distribution across hexes
"""

import os, json, warnings, time, math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pulp import (LpProblem, LpMaximize, LpVariable, lpSum,
                  LpBinary, value, PULP_CBC_CMD)

warnings.filterwarnings('ignore')
np.random.seed(42)

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
HEX_ELDERLY_SE = f'{OUTPUTS}/hexagon_elderly_SE.csv'
HEX_ZVEH_SHARE = f'{OUTPUTS}/hexagon_zveh_share.csv'

CANDIDATE_NAICS = [813110, 611110, 624410, 621111,
                   922110, 922120, 922130, 922140, 922150, 922160, 922190]

ALPHA_CAR = -0.020097;  BETA_CAR = 1.361630
PI        = {'ec': 0.85, 'enc': 0.90, 'nec': 0.55, 'nenc': 0.60}
K_SPARSE  = 15
MOE_Z     = 1.645   # ACS publishes MOE at 90% CI
CI95_Z    = 1.960   # for reporting 95% CI

# Sensitivity settings
P_SENS    = [1, 4, 8, 12, 26, 50]   # P values to analyse
R_MC      = 200                       # Monte Carlo replicates
MODE      = 'car_only'                # mode to analyse (dominant mode)

def tick(label): print(f'\n[START] {label}'); return time.time()
def tock(t0, label): print(f'[DONE]  {label}  ({time.time()-t0:.1f}s)')

def impedance(t, a, b):
    t = np.asarray(t, dtype=float)
    w = np.zeros_like(t)
    m = np.isfinite(t) & (t > 0)
    w[m] = np.exp(a * (t[m] ** b))
    return w

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load spatial and POI data (same as primal_accessibility.py)
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

# County-level totals
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
COUNTY_ZV   = sum(float(dict(zip(headers_veh,r)).get('B08201_002E',0) or 0)
                  for r in acs_veh[1:])
COUNTY_HH   = max(sum(float(dict(zip(headers_veh,r)).get('B08201_001E',0) or 0)
                      for r in acs_veh[1:]), 1)
COUNTY_ZV_SHARE = COUNTY_ZV / COUNTY_HH
print(f'  County total pop: {COUNTY_TOTAL:.0f}  elderly: {COUNTY_ELDERLY:.0f}  zv_share: {COUNTY_ZV_SHARE:.4f}')
tock(t0, 'Load data')

# ══════════════════════════════════════════════════════════════════════════════
# 2. Load ACS uncertainty files
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Load ACS SE data')
se_df = pd.read_csv(HEX_ELDERLY_SE)
zv_df = pd.read_csv(HEX_ZVEH_SHARE)

hex_eld_est  = dict(zip(se_df['hex_id'], se_df['elderly_est']))
hex_total_pop= dict(zip(se_df['hex_id'], se_df['total_pop']))
hex_se_eld   = dict(zip(se_df['hex_id'], se_df['SE_elderly_vrt']))
hex_moe_pct  = dict(zip(se_df['hex_id'], se_df['MOE_pct']))
hex_zv       = dict(zip(zv_df['hex_id'], zv_df['zv_share']))
hex_se_zv    = dict(zip(zv_df['hex_id'], zv_df['SE_zv_share']))

print(f'  Loaded SE data for {len(hex_eld_est)} hexes')
print(f'  MOE_pct: mean={se_df["MOE_pct"].mean():.1f}%  max={se_df["MOE_pct"].max():.1f}%')
print(f'  Hexes with MOE > 50%: {(se_df["MOE_pct"]>50).sum()}')
print(f'  SE_zv_share: mean={zv_df["SE_zv_share"].mean():.4f}  max={zv_df["SE_zv_share"].max():.4f}')
tock(t0, 'Load ACS SE data')

# ══════════════════════════════════════════════════════════════════════════════
# 3. Build candidate set J and impedance (same as primal)
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Build J and impedance')
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
hv.columns = ['hex_id','total_visits']
hv = hv[hv['total_visits'] > 0]
J_df  = hv[hv['hex_id'].isin(hex_ids)].copy().reset_index(drop=True)
J_df  = J_df.merge(hexes[['hex_id','lon','lat']], on='hex_id', how='left')
J_ids = J_df['hex_id'].tolist()
print(f'  |J| = {len(J_ids)} candidate stops')

car_df = tt_car[tt_car['to_hex'].isin(J_ids)].copy()
car_df['w'] = impedance(car_df['travel_time_min'].values, ALPHA_CAR, BETA_CAR)
W_car = {}
for row in car_df.itertuples(index=False):
    W_car.setdefault(row.from_hex, {})[row.to_hex] = row.w

# W_by_seg for car_only mode
W_by_seg_car = {'ec': W_car, 'enc': {}, 'nec': W_car, 'nenc': {}}

# k nearest per hex
nearest_j = {}
for i in hex_ids:
    scored = [(j, W_car.get(i,{}).get(j,0)) for j in J_ids]
    scored.sort(key=lambda x: -x[1])
    nearest_j[i] = [j for j,_ in scored[:K_SPARSE]]

tock(t0, 'Build J and impedance')

# ══════════════════════════════════════════════════════════════════════════════
# 4. Helper: build n_ig from perturbed estimates
# ══════════════════════════════════════════════════════════════════════════════
def build_n_ig(eld_pert, zv_pert):
    """
    Build n_ig from perturbed elderly estimates and zv_share.
    Clips to valid ranges: elderly in [0, total_pop], zv in [0,1].
    """
    n_per_hex = COUNTY_TOTAL / N_HEX
    n_ig = {}
    for hid in hex_ids:
        n_h  = hex_total_pop.get(hid, n_per_hex)
        e_h  = float(np.clip(eld_pert.get(hid, COUNTY_ELDERLY/N_HEX), 0, n_h))
        zv_h = float(np.clip(zv_pert.get(hid, COUNTY_ZV_SHARE), 0, 1))
        nn_h = max(n_h - e_h, 0)
        n_ig[hid] = {
            'ec':   e_h  * (1 - zv_h),
            'enc':  e_h  * zv_h,
            'nec':  nn_h * (1 - zv_h),
            'nenc': nn_h * zv_h,
        }
    return n_ig

# ══════════════════════════════════════════════════════════════════════════════
# 5. Helper: compute A_max and solve BLP at given P with given n_ig
# ══════════════════════════════════════════════════════════════════════════════
def compute_A_max(n_ig, W_by_seg):
    total = 0.0
    for i in hex_ids:
        for g, Wm in W_by_seg.items():
            if not Wm: continue
            best = max((Wm.get(i,{}).get(j,0)) for j in nearest_j[i])
            total += PI[g] * n_ig[i][g] * best
    return total

def solve_blp_at_P(P, n_ig, W_by_seg, A_max_val, silent=True):
    prob = LpProblem(f'sens_P{P}', LpMaximize)
    y = {j: LpVariable(f'y_{j}', cat=LpBinary) for j in J_ids}
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

    obj_terms = []
    for g, zi_dict in z.items():
        Wm = W_by_seg[g]
        for i, zij in zi_dict.items():
            for j, zvar in zij.items():
                coeff = PI[g] * n_ig[i][g] * Wm.get(i,{}).get(j,0)
                if coeff > 0:
                    obj_terms.append(coeff * zvar)
    prob += lpSum(obj_terms)
    prob += lpSum(y[j] for j in J_ids) == P
    for g, zi_dict in z.items():
        for i, zij in zi_dict.items():
            if zij:
                prob += lpSum(zij.values()) <= 1
                for j, zvar in zij.items():
                    prob += zvar <= y[j]

    prob.solve(PULP_CBC_CMD(msg=0))
    sel     = [j for j in J_ids if value(y[j]) is not None and value(y[j]) > 0.5]
    raw_obj = value(prob.objective) or 0.0
    A_tilde = raw_obj / A_max_val if A_max_val > 0 else 0.0
    return A_tilde, set(sel)

# ══════════════════════════════════════════════════════════════════════════════
# 6. Baseline n_ig
# ══════════════════════════════════════════════════════════════════════════════
n_ig_base = build_n_ig(hex_eld_est, hex_zv)
A_max_base = compute_A_max(n_ig_base, W_by_seg_car)
print(f'\n  Baseline A_max (car_only): {A_max_base:.2f}')

# ══════════════════════════════════════════════════════════════════════════════
# 7. Approach 1: MOE Bracket
#    Solve at n_ig - MOE (lower), n_ig (baseline), n_ig + MOE (upper)
#    MOE = MOE_pct/100 * elderly_est for elderly
#    MOE_zv = MOE_Z * SE_zv_share for zv_share
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('MOE Bracket Analysis')

# Build perturbed estimates
eld_lower = {hid: max(0, hex_eld_est[hid] * (1 - hex_moe_pct[hid]/100))
             for hid in hex_ids}
eld_upper = {hid: hex_eld_est[hid] * (1 + hex_moe_pct[hid]/100)
             for hid in hex_ids}

zv_lower  = {hid: max(0, hex_zv[hid] - MOE_Z * hex_se_zv.get(hid, 0))
             for hid in hex_ids}
zv_upper  = {hid: min(1, hex_zv[hid] + MOE_Z * hex_se_zv.get(hid, 0))
             for hid in hex_ids}

n_ig_lower = build_n_ig(eld_lower, zv_upper)   # lower elderly + higher zv -> smaller ec
n_ig_upper = build_n_ig(eld_upper, zv_lower)   # upper elderly + lower zv -> larger ec

A_max_lower = compute_A_max(n_ig_lower, W_by_seg_car)
A_max_upper = compute_A_max(n_ig_upper, W_by_seg_car)

moe_rows = []
print(f'\n  {"P":>3}  {"A_lower":>8}  {"A_base":>8}  {"A_upper":>8}  {"band_pp":>8}')
for P in P_SENS:
    t_p = time.time()
    At_l, sel_l = solve_blp_at_P(P, n_ig_lower, W_by_seg_car, A_max_lower)
    At_b, sel_b = solve_blp_at_P(P, n_ig_base,  W_by_seg_car, A_max_base)
    At_u, sel_u = solve_blp_at_P(P, n_ig_upper, W_by_seg_car, A_max_upper)

    band = (At_u - At_l) * 100
    print(f'  {P:>3}  {At_l:>8.4f}  {At_b:>8.4f}  {At_u:>8.4f}  {band:>7.2f}pp  ({time.time()-t_p:.1f}s)')
    moe_rows.append({
        'P': P, 'A_tilde_lower': At_l,
        'A_tilde_base': At_b, 'A_tilde_upper': At_u,
        'band_pp': band,
        'sel_lower': '|'.join(sel_l),
        'sel_base':  '|'.join(sel_b),
        'sel_upper': '|'.join(sel_u),
        'overlap_lb': len(sel_l & sel_b),
        'overlap_ub': len(sel_u & sel_b),
    })

moe_df = pd.DataFrame(moe_rows)
moe_df.to_csv(f'{OUTPUTS}/sensitivity_moe_bracket.csv', index=False)
print(f'  Saved: sensitivity_moe_bracket.csv')
tock(t0, 'MOE Bracket Analysis')

# ══════════════════════════════════════════════════════════════════════════════
# 8. Approach 2: Monte Carlo
#    Draw R replicates of (elderly_est, zv_share) from N(est, SE) per hex
#    Solve BLP at each replicate for each P in P_SENS
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick(f'Monte Carlo ({R_MC} replicates)')

# Pre-draw all replicates
print(f'  Pre-drawing {R_MC} replicates...')
eld_draws = {}   # hex_id -> array of R draws
zv_draws  = {}
for hid in hex_ids:
    eld_draws[hid] = np.random.normal(
        hex_eld_est.get(hid, COUNTY_ELDERLY/N_HEX),
        hex_se_eld.get(hid, 0),
        size=R_MC)
    zv_draws[hid]  = np.random.normal(
        hex_zv.get(hid, COUNTY_ZV_SHARE),
        hex_se_zv.get(hid, 0),
        size=R_MC)

mc_rows = []
for r in range(R_MC):
    eld_r = {hid: eld_draws[hid][r] for hid in hex_ids}
    zv_r  = {hid: zv_draws[hid][r]  for hid in hex_ids}
    n_ig_r   = build_n_ig(eld_r, zv_r)
    A_max_r  = compute_A_max(n_ig_r, W_by_seg_car)

    for P in P_SENS:
        At, sel = solve_blp_at_P(P, n_ig_r, W_by_seg_car, A_max_r)
        mc_rows.append({'replicate': r, 'P': P, 'A_tilde': At,
                        'selected_stops': '|'.join(sel)})

    if (r+1) % 20 == 0:
        print(f'  Replicate {r+1}/{R_MC} done')

mc_df = pd.DataFrame(mc_rows)
mc_df.to_csv(f'{OUTPUTS}/sensitivity_mc_results.csv', index=False)
print(f'  Saved: sensitivity_mc_results.csv')
tock(t0, 'Monte Carlo')

# ══════════════════════════════════════════════════════════════════════════════
# 9. MC Summary statistics
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('MC Summary')
mc_summary = mc_df.groupby('P')['A_tilde'].agg(
    mean='mean', std='std', min='min', max='max',
    p5=lambda x: x.quantile(0.05),
    p95=lambda x: x.quantile(0.95)
).reset_index()
mc_summary['ci95_lo'] = mc_summary['mean'] - CI95_Z * mc_summary['std']
mc_summary['ci95_hi'] = mc_summary['mean'] + CI95_Z * mc_summary['std']
mc_summary['band_pp'] = (mc_summary['p95'] - mc_summary['p5']) * 100
mc_summary.to_csv(f'{OUTPUTS}/sensitivity_mc_summary.csv', index=False)

print(f'\n  MC Summary (car_only):')
print(f'  {"P":>3}  {"mean":>7}  {"std":>6}  {"p5":>7}  {"p95":>7}  {"band_pp":>8}')
for _, row in mc_summary.iterrows():
    mn = row['mean']; sd = row['std']
    print(f'  {int(row.P):>3}  {mn:>7.4f}  {sd:>6.4f}  '
          f'{row.p5:>7.4f}  {row.p95:>7.4f}  {row.band_pp:>7.2f}pp')

# Stop selection stability
print(f'\n  Stop selection stability (fraction of replicates each stop is selected):')
for P in P_SENS:
    sub = mc_df[mc_df['P']==P]
    all_stops = {}
    for _, r in sub.iterrows():
        for s in r['selected_stops'].split('|'):
            if s: all_stops[s] = all_stops.get(s, 0) + 1
    total = len(sub)
    stable = {s: c/total for s,c in all_stops.items() if c/total >= 0.8}
    unstable = {s: c/total for s,c in all_stops.items() if c/total < 0.8}
    print(f'  P={P}: {len(stable)} stable stops (>=80%),  {len(unstable)} unstable (<80%)')

mc_summary.to_csv(f'{OUTPUTS}/sensitivity_mc_summary.csv', index=False)
print(f'  Saved: sensitivity_mc_summary.csv')
tock(t0, 'MC Summary')

# ══════════════════════════════════════════════════════════════════════════════
# 10. Plot 1: MOE band plot
# ══════════════════════════════════════════════════════════════════════════════
t0 = tick('Generating plots')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('ACS Sampling Uncertainty Sensitivity — Car Only Mode\n'
             'Rockingham County, NC  |  Allocation BLP',
             fontsize=12, y=1.02)

# Left: MOE bracket
ax = axes[0]
ax.fill_between(moe_df['P'], moe_df['A_tilde_lower'], moe_df['A_tilde_upper'],
                alpha=0.25, color='#e74c3c', label='±1 MOE band')
ax.plot(moe_df['P'], moe_df['A_tilde_base'],
        color='#e74c3c', lw=2.5, label='Baseline')
ax.plot(moe_df['P'], moe_df['A_tilde_lower'],
        color='#e74c3c', lw=1.2, ls='--', alpha=0.7, label='Lower (−MOE)')
ax.plot(moe_df['P'], moe_df['A_tilde_upper'],
        color='#e74c3c', lw=1.2, ls=':', alpha=0.7, label='Upper (+MOE)')

for _, row in moe_df.iterrows():
    ax.annotate(f'±{row["band_pp"]:.1f}pp',
                xy=(row['P'], (row['A_tilde_lower']+row['A_tilde_upper'])/2),
                fontsize=7, ha='left', color='#c0392b',
                xytext=(row['P']+0.5, (row['A_tilde_lower']+row['A_tilde_upper'])/2))

ax.set_xlabel('Number of clinic stops (P)', fontsize=11)
ax.set_ylabel(r'$\tilde{\mathcal{A}}$ (normalised accessibility)', fontsize=11)
ax.set_title('Approach 1: ±1 MOE bracket\n'
             '(perturbed elderly estimate + zero-vehicle share)',
             fontsize=10, pad=5)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(0, max(P_SENS)+2)
ax.set_ylim(0, 1.05)

# Right: MC 90% band
ax2 = axes[1]
p_all = mc_summary['P'].values
ax2.fill_between(p_all, mc_summary['p5'], mc_summary['p95'],
                 alpha=0.25, color='#2980b9', label='5th–95th percentile')
ax2.fill_between(p_all, mc_summary['ci95_lo'].clip(0), mc_summary['ci95_hi'],
                 alpha=0.15, color='#1abc9c', label='95% CI (mean ± 1.96σ)')
ax2.plot(p_all, mc_summary['mean'],
         color='#2980b9', lw=2.5, label=f'MC mean (R={R_MC})')

for _, row in mc_summary.iterrows():
    ax2.annotate(f'±{row["band_pp"]/2:.1f}pp',
                 xy=(row['P'], row['mean']),
                 fontsize=7, ha='left', color='#1a5276',
                 xytext=(row['P']+0.5, row['mean']))

ax2.set_xlabel('Number of clinic stops (P)', fontsize=11)
ax2.set_ylabel(r'$\tilde{\mathcal{A}}$ (normalised accessibility)', fontsize=11)
ax2.set_title(f'Approach 2: Monte Carlo (R = {R_MC} replicates)\n'
              r'$n_{ig} \sim \mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$ per hex',
              fontsize=10, pad=5)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, max(P_SENS)+2)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(f'{OUTPUTS}/sensitivity_band_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: sensitivity_band_plot.png')

# ══════════════════════════════════════════════════════════════════════════════
# 11. Plot 2: MOE% distribution across hexes
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ACS Sampling Uncertainty per Hexagon — Rockingham County, NC',
             fontsize=12, y=1.02)

# MOE% histogram
ax = axes[0]
ax.hist(se_df['MOE_pct'].dropna(), bins=30, color='#e74c3c',
        edgecolor='white', linewidth=0.4, alpha=0.85)
ax.axvline(se_df['MOE_pct'].mean(), color='black', lw=1.5, ls='--',
           label=f'Mean = {se_df["MOE_pct"].mean():.1f}%')
ax.axvline(30, color='#f39c12', lw=1.2, ls=':',
           label='30% reference')
ax.axvline(50, color='#c0392b', lw=1.2, ls=':',
           label='50% reference')
ax.set_xlabel('Margin of error (% of estimate) — elderly population', fontsize=11)
ax.set_ylabel('Number of hexagons', fontsize=11)
ax.set_title('Distribution of ACS MOE% for elderly estimate\n'
             'after VRT downscaling to H3 resolution 7', fontsize=10, pad=5)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# SE_zv histogram
ax2 = axes[1]
ax2.hist(zv_df['SE_zv_share'].dropna(), bins=30, color='#2980b9',
         edgecolor='white', linewidth=0.4, alpha=0.85)
ax2.axvline(zv_df['SE_zv_share'].mean(), color='black', lw=1.5, ls='--',
            label=f'Mean SE = {zv_df["SE_zv_share"].mean():.4f}')
ax2.set_xlabel('Standard error of zero-vehicle share', fontsize=11)
ax2.set_ylabel('Number of hexagons', fontsize=11)
ax2.set_title('Distribution of SE for zero-vehicle share\n'
              'after VRT downscaling to H3 resolution 7', fontsize=10, pad=5)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUTS}/sensitivity_moe_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: sensitivity_moe_distribution.png')

tock(t0, 'Generating plots')

# ══════════════════════════════════════════════════════════════════════════════
# 12. Summary
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('SENSITIVITY ANALYSIS SUMMARY')
print('='*65)
print(f'\n  Mode: car_only  |  R = {R_MC} MC replicates')
print(f'\n  {"P":>3}  {"Baseline":>9}  {"MOE band":>9}  {"MC p5–p95":>12}  {"MC std":>8}')
for P in P_SENS:
    moe_row = moe_df[moe_df['P']==P].iloc[0]
    mc_row  = mc_summary[mc_summary['P']==P].iloc[0]
    base    = moe_row['A_tilde_base']
    moe_b   = moe_row['band_pp']
    mc_b    = mc_row['band_pp']
    mc_s    = mc_row['std']*100
    print(f'  {P:>3}  {base:>9.4f}  {moe_b:>8.2f}pp  '
          f'{mc_b:>11.2f}pp  {mc_s:>7.2f}pp')

print('\nDone.')
