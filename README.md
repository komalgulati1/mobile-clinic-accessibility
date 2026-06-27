# Mobile Clinic Accessibility — P-Median Optimization Framework

**Coordinating Microtransit and Mobile Clinics for Rural Healthcare Access**  
Rockingham County, NC  

Komal Gulati · CR2C2, NC A&T State University  
Advisor: Dr. Venktesh Pandey  
Funded by USDOT / CR2C2

---

## Overview

This repository implements a **P-Median accessibility maximization** framework
to identify optimal mobile clinic stop locations in rural Rockingham County, NC.

The model selects P clinic stops from a candidate set J to maximize
population-weighted gravity-based accessibility across five mode scenarios:

| Scenario | Description |
|---|---|
| **(a) Car only** | Baseline — car-owning segments only |
| **(b) Car + Transit** | Car owners use car; car-free use SKAT fixed-route |
| **(c) Car + Microtransit** | Car owners use car; car-free use RCATS microtransit |
| **(d) Car for all** | Theoretical upper bound |
| **(e) Transit only** | Theoretical lower bound |

### Key Results (Rockingham County, NC)

| P | Car Only (Ã) | Car + Transit (Ã) | Car + Microtransit (Ã) | Car for All (Ã) |
|---|---|---|---|---|
| 1 | 0.168 | 0.168 | 0.171 | 0.180 |
| 2 | 0.298 | 0.298 | 0.303 | 0.321 |
| 3 | 0.413 | 0.414 | 0.420 | 0.445 |
| 4 | 0.468 | 0.469 | 0.476 | 0.504 |

Ã = normalized accessibility (fraction of theoretical maximum).
Transit adds negligible benefit due to near-zero SKAT coverage
(only 5.1% of OD pairs reachable; mean travel time 98.5 min).
Microtransit (RCATS) adds a consistent ~0.7 pp increment at P=4.

---

## Repository Structure

```
mobile-clinic-accessibility/
│
├── config.py                       ← Central config — paths + model parameters
│
├── scripts/
│   ├── 00_rerun_transit.py         ← (Optional) Recompute transit OD matrix
│   ├── 01_accessibility_pipeline.py← Main model — solve P-Median for all scenarios
│   ├── 02_generate_heatmaps.py     ← Choropleth heatmaps for all 5 scenarios
│   └── 03_generate_difference_maps.py ← Difference maps + absolute accessibility
│
├── data/
│   ├── README.md                   ← How to obtain the data
│   ├── raw/                        ← Source data (not tracked by git)
│   └── processed/                  ← Derived inputs (not tracked by git)
│
├── outputs/
│   ├── README.md                   ← What each output file contains
│   ├── maps/                       ← PNG heatmaps (generated)
│   ├── plots/                      ← Step plots (generated)
│   └── tables/                     ← CSV results (generated)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/mobile-clinic-accessibility.git
cd mobile-clinic-accessibility
pip install -r requirements.txt
```

For transit routing (optional):
```bash
pip install r5py        # also requires Java 11+
```

### 2. Set up data

Place your data files according to `data/README.md`.
The minimum required files to run the main pipeline are:

```
data/raw/
    rockingham_POIs.csv
    acs_rockingham.json
    acs_vehicles_tract.json

data/processed/
    rockingham_hex_r7.gpkg
    rockingham_hex_centroids_r7.csv
    tt_car_r7.csv
    tt_transit_r7_avg.csv        ← or tt_transit_r7.csv (single-run fallback)
    hospital_access_summary.csv
    rockingham_hospitals.csv
```

### 3. Run the pipeline

```bash
# (Optional) Recompute averaged transit travel times — needs r5py + GTFS
python scripts/00_rerun_transit.py

# Main model — solves P-Median for all scenarios and P=1..4
python scripts/01_accessibility_pipeline.py

# Choropleth heatmaps for all 5 scenarios
python scripts/02_generate_heatmaps.py

# Difference maps and absolute accessibility plots
python scripts/03_generate_difference_maps.py
```

All outputs are saved to `outputs/` automatically.

### 4. Adapting to a different study area

To apply this framework to a different county or region:

1. **Update `config.py`** — the only file you need to edit:
   - Point `DATA_RAW` and `DATA_PRO` to your data folders
   - Update `COUNTY_TOTAL`, `COUNTY_ELDERLY`, `ZV_SHARE` from ACS
   - Update `CANDIDATE_NAICS` if your clinic stop typology differs
   - Update `PI` usage probabilities if you have local estimates

2. **Replace the data files** with your study area equivalents:
   - H3 hex grid at resolution 7 (or adjust to your preferred resolution)
   - Car and transit OD travel time matrices (from r5py, OTP, or similar)
   - POI visit data with `total_visit_2025` column (or equivalent)
   - ACS population and vehicle availability data

3. The model code requires **no other changes**.

---

## Model Formulation

### Objective

$$\tilde{\mathcal{A}}(\mathbf{y}) = \frac{1}{\mathcal{A}^{\max}} \sum_{i \in I} \sum_{g} \lambda_g^m \cdot \pi_g \cdot n_{ig} \sum_{j \in J} o_j \cdot w_{ij}^m \cdot y_j$$

subject to:
$$\sum_{j \in J} y_j = P, \quad y_j \in \{0, 1\}$$

### Variables and parameters

| Symbol | Description |
|---|---|
| $y_j \in \{0,1\}$ | 1 if stop $j$ is open |
| $P$ | Number of stops to select (1–4) |
| $J$ | Candidate stop hexagons (POI visit weight > 0) |
| $I$ | All hexagons (demand zones) |
| $\pi_g$ | Usage probability for segment $g$ (ec=0.85, enc=0.90, nec=0.55, nenc=0.60) |
| $n_{ig}$ | Population of segment $g$ in hexagon $i$ |
| $o_j$ | Stop attractiveness (normalized POI visit count) |
| $w_{ij}^m$ | Gravity impedance under mode $m$: $\exp(\alpha^m \cdot (t_{ij}^m)^{\beta^m})$ |
| $\lambda_g^m$ | Mode availability indicator for segment $g$ under scenario $m$ |
| $\mathcal{A}^{\max}$ | Upper bound: car-for-all with all stops open |

### Microtransit travel time

$$t_{ij}^{mt} = \delta \cdot t_{ij}^{car} + W$$

where:
- $\delta \sim \text{Weibull}(k=1.069, \lambda=1.3)$, clipped to $[1, 2]$ — detour ratio (Hu & Xu 2025)
- $W \sim \text{Weibull}(k=2.084, \lambda=19.901 \text{ min})$ — waiting time (Yang & Gao 2025)

Uncertainty is quantified via $N=200$ Monte Carlo draws.

---

## References

- Verma, S. et al. (2025). *Scientific Data*. Gravity impedance parameters.
- Yang, Z. & Gao, S. (2025). *Transportation Research Part A*, 104504.
  Microtransit wait time distribution.
- Hu, X. & Xu, M. (2025). *Transportation Research Part F*, 103363.
  Microtransit detour ratio distribution.
- Malone, N.C. et al. (2020). *Int J Equity Health* 19, 101.
  Mobile clinic literature.
- Hakimi, S.L. (1964). *Operations Research* 12(3), 450–459.
  P-Median problem.

---

## Contact

Komal Gulati · kgulati@ncat.edu  
CR2C2 — Center for Regional and Rural Connected Communities  
NC A&T State University, Greensboro, NC
