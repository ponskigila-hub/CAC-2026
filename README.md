# FoodShock — Cross-Country ASEAN Food Price Forecasting & Early Shock Detection System

![SDG 2](https://img.shields.io/badge/SDG-2%20Zero%20Hunger-green)
![Competition](https://img.shields.io/badge/MLSC%20Cyan%202026-EDM%20Laboratory-blue)
![Python](https://img.shields.io/badge/Python-3.12-blue)

**Machine Learning Solution Challenge (MLSC) – Cyan Academic Competition 2026**  
**Organizer:** Enterprise Data Management (EDM) Laboratory  
**Target SDG:** **SDG 2 — Zero Hunger**

FoodShock forecasts monthly staple-food prices across ASEAN markets and issues early warnings for potential price shocks, supporting price-stability monitoring under SDG 2.

---

## Shock Threshold Justification (+5% MoM)

1. **Policy foundation:** Aligned with the WFP **ALPS (Alert for Price Spikes)** framework and ASEAN Food Security Information System (**AFSIS**) monitoring practice, where a month-over-month staple-food increase above **+5%** signals acute affordability stress (roughly equivalent to >60% annualised food inflation if sustained).
2. **Empirical foundation:** On the ASEAN panel return distribution (`target_pct_change_t1`), **P90 ≈ +4.44%** and **P95 ≈ +7.20%** (IQR ≈ [−1.89%, +1.77%]). The **+5.0%** rule isolates the upper ~8.5–9% tail, and is combined with adaptive series-specific monitoring (μᵢ + 1.5σᵢ) and a **calibrated Stage-2 probability threshold** (F1-optimal mean threshold ≈ 0.40; operational mean threshold ≈ 0.63).

---


## SDG Target 2.c & Operational Users

FoodShock maps to **SDG Target 2.c** — ensure proper functioning of food commodity markets and facilitate timely access to market information to help limit extreme food-price volatility.

| Alert | Meaning | Intended users | Illustrative action |
|-------|---------|----------------|---------------------|
| **NORMAL** | No material upward pressure | Routine monitoring desks | Continue scheduled surveillance |
| **WASPADA / ELEVATED RISK** | Moderate rise or elevated P(Shock) | AFSIS secretariat; national food-price units | Intensify monitoring; prepare market briefs |
| **FOOD SHOCK ALERT** | P(Shock) ≥ train-validated threshold **or** predicted rise > +5% MoM | National food agencies (e.g. stabilisation bodies), trade ministries | Trigger rapid assessment; consider targeted market operations / communication |

The **+5% MoM** rule is FoodShock’s **empirical upper-tail threshold** (panel P90 ≈ +4.44%, P95 ≈ +7.20%), inspired by WFP ALPS / AFSIS spike-monitoring practice. Official ALPS is based on seasonal-trend deviations; FoodShock does not claim to re-implement the full ALPS algorithm.

## Architectural Rationale & Methodological Alignment

| Stage | What we do | Why it fits the problem |
|-------|------------|-------------------------|
| **1. Audit ≥ 60 months & unit standardisation** | Keep series with ≥ 60 months; convert to USD; KG (rice/meat) / L (oil) | Short series cannot support seasonal lags; mixed currencies/units create artificial scale bias |
| **2. Calendar-safe FE (zero leakage)** | Full monthly re-index **without** forward-fill of raw prices; lags/rolling from `shift(1)` | Forward-filling empty months invents 0% changes and leaks future levels into rolling stats |
| **3. Macro: implied FX & ASEAN spillover** | `fx_pct_1m/3m`, `fx_vol_6m` from price/usdprice; regional commodity momentum + country gap | Local food inflation often tracks FX depreciation and regional supply shocks, not only local lags |
| **4. Residual target** \(\Delta y_{t+1}=y_{t+1}-y_t\) from **current** \(y_t\) (`usdprice`) | Train on changes; reconstruct \(\hat y_{t+1}=y_t+\widehat{\Delta y}\) | Rice (~USD 0.8) vs beef (~USD 6–7) differ ~9×; level targets bias trees toward expensive goods |
| **5. 6-fold expanding-window walk-forward + DM test** | Cutoffs 2019-06…2024-06; 12-month test each; micro & macro metrics; Diebold–Mariano vs Persistence | All **7 countries** (incl. IDN, THA) receive OOS evaluation where data exist; significance is tested, not assumed |
| **6. Two-stage risk layer** | Point residual + Quantile **P10–P90** (PICP ≈ **80.2%** (conformalized, all folds)) + balanced **Stage-2 classifier** (Operational Precision ≈ **0.169**, ROC-AUC ≈ **0.627**, thr ≈ **0.67**) | Point forecasts regress to the mean on rare shocks; probability + upside interval operationalise ALPS/AFSIS alerts |
| **7. 5-page Streamlit app** | Home, EDA, Preprocessing, Training, Prediction (what-if + interactive what-if) | Turns the pipeline into a usable early-warning tool for analysts |

---

## Repository Structure (40 entries)

```
.
├── README.md                          # Project docs, methodology, results, quick start
├── requirements.txt                   # Reproducible Python dependencies
├── .gitignore                         # Excludes venv, caches, raw >90MB extracts
├── app.py                             # Streamlit entry: 5-page navigation + sidebar stats
├── data_loader.py                     # Cached CSV loaders for app pages
├── models.py                          # Two-stage V2: residual point, P10/P90, shock classifier
├── utils.py                           # sMAPE, residual reconstruction, alert rules, FEATURE_COLS
├── styles.py                          # Dark analytics theme, KPI cards, status badges
├── .streamlit/
│   └── config.toml                    # Streamlit theme (emerald SDG-2 accent)
├── src/
│   ├── foodshock_pipeline.py          # Audit ≥60m, unit std, calendar-safe FE (batch)
│   ├── run_residual_modeling.py       # V1 residual baseline / SHAP helper
│   └── run_upgraded_v2.py             # V2: FX/spillover, 6-fold WF, Optuna, DM, quantiles, classifier
├── notebooks/
│   └── FoodShock_MLSC2026_Solution.ipynb  # Full narrative + embedded V2 outputs
├── views/
│   ├── __init__.py
│   ├── home.py                        # KPI, architecture flow, WF benchmark, Stage-2 metrics
│   ├── eda.py                         # Interactive filters & trends
│   ├── preprocessing.py               # Audit transparency & integrity rules
│   ├── training.py                    # Official tables + live train residual vs direct
│   └── prediction.py                  # What-if & interactive what-if with P10–P90 and P(Shock)
├── data/
│   ├── foodshock_asean_model_ready.csv    # Clean panel 27,950 × 38 (0 NaN/Inf)
│   ├── fpma_wfp_country_coverage.csv
│   ├── fpma_wfp_commodity_coverage.csv
│   ├── model_evaluation_results.csv       # WF micro/macro, SS_RMSE, DM tests
│   ├── model_eval_by_commodity.csv
│   ├── model_eval_by_country.csv
│   ├── model_eval_by_pricetype.csv        # Retail n=22,506 · Wholesale n=1,385 (OOS)
│   ├── model_eval_shock_classification.csv
│   └── shock_threshold_justification.txt
├── figures/
│   ├── eda_trends.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── error_by_commodity.png
│   ├── shap_summary.png
│   └── shock_residual_dist.png
```

> **Raw data note:** The original WFP/HDX extract (`asean_wfp_2015_2025.csv`, ~98 MB) is **not** shipped (GitHub size limits). The model-ready panel is `data/foodshock_asean_model_ready.csv`. Source: [WFP Food Prices – HDX](https://data.humdata.org/).

---

## Key Results (V4 Expanding-Window Walk-Forward)

Validation: 6 expanding folds (2019-06 … 2024-06). **All 7 ASEAN countries** appear in OOS windows where data exist. Encoding and threshold selection are **fold-internal** (month-block validation).

| Model | MAE (micro) | RMSE (micro) | sMAPE | SS_RMSE | DM p-value (panel HAC) |
|-------|-------------|--------------|-------|---------|------------------------|
| **CatBoost WF Residual** | 0.1015 | **0.1931** | 3.18% | **+0.0099** | 0.352 |
| Persistence (y_t) | **0.0986** | 0.1950 | 3.04% | 0.000 | — |
| XGBoost WF Residual | 0.1076 | 0.1991 | 3.35% | −0.021 | 0.642 |
| LightGBM WF Residual | 0.1083 | 0.2008 | 3.51% | −0.030 | 0.592 |

**Honest econometric reading:** Monthly staple prices are near-martingale. CatBoost reduces **RMSE** by ~1% vs Persistence (tail-error skill), but **MAE is slightly higher** and the **panel Diebold–Mariano test is not significant at 5%** (p ≈ 0.35). Practical value is therefore concentrated in calibrated **P10–P90 intervals** and **Stage-2 risk ranking**, not in a large mean-error gap.

### Conformalized Quantile Intervals (all 6 folds)

| Overall OOS PICP (P10–P90) | Mean interval width (USD) |
|----------------------------|---------------------------|
| **0.802** (~80% target) | **0.292** |

### Stage-2 Upward Shock Classifier vs Momentum

Label: `target_pct_change_t1 > +0.05` (base rate ≈ 9%). Thresholds chosen on **month-block** validation inside each train fold.

**Operational rule:** maximize Recall subject to Precision ≥ 0.20 on the validation block; if no threshold meets the floor, fall back to maximizing $F_{0.5}$. In V5: **5/6 folds** used the precision floor, **1/6** used $F_{0.5}$ fallback. Mean operational threshold ≈ **0.63**.

| Method | Precision | Recall | F1 | ROC-AUC | PR-AUC | **FDR** | **FPR** | Mean thr |
|--------|-----------|--------|-----|---------|--------|---------|---------|----------|
| Stage-2 F1-optimal | 0.121 | 0.641 | 0.204 | 0.627 | 0.135 | 0.879 | 0.457 | 0.40 |
| **Stage-2 Operational** | **0.163** | 0.192 | 0.176 | **0.627** | **0.135** | **0.837** | **0.097** | **0.63** |
| Momentum `pct_change_1m>5%` | 0.136 | 0.133 | 0.135 | 0.494 | 0.102 | 0.864 | 0.083 | 0.05 |

**FDR vs FPR:** FDR = $1-$Precision = FP/(TP+FP) is the share of *issued alerts* that are false. FPR = FP/(FP+TN) is the share of *non-shock months* that incorrectly trigger an alert. Operational mode cuts FPR to ≈ **9.7%** (vs F1-mode ≈ 46%).



## Quick Start

### 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Solution Notebook
```bash
jupyter notebook notebooks/FoodShock_MLSC2026_Solution.ipynb
```

### 3. Batch Pipeline (optional re-run)
```bash
python src/foodshock_pipeline.py      # requires local raw WFP extract if rebuilding from scratch
python src/run_upgraded_v2.py         # V2 walk-forward + two-stage artefacts
```

### 4. Interactive Web Application (Streamlit)
```bash
streamlit run app.py
```

**Suggested exploration flow**
1. **Home** — KPIs & official walk-forward benchmark  
2. **EDA** — interactive filters & price trends  
3. **Preprocessing** — coverage ≥ 60 months proof  
4. **Model Training** — live Residual vs Direct vs Persistence  
5. **Prediction & Alert** — what-if + interactive what-if + P10–P90 + P(Shock)  

---

## Limitations & Future Work

Monthly staple prices behave like a near-martingale: Persistence remains competitive on MAE. The system’s value is concentrated in **RMSE skill**, **panel Diebold–Mariano tests that are **not** significant at 5% (CatBoost p ≈ 0.35)

Current limitations include sparse coverage for some country–commodity cells (e.g., THA, MMR), the absence of harmonised weather/energy covariates across all seven countries, and still-moderate precision on the shock classifier. Future work: richer external drivers, conformal or quantile ensembles, and cost-sensitive thresholding for operational AFSIS dashboards.

---

*FoodShock · MLSC 2026 · SDG 2 — Zero Hunger*

## Test-sample coverage (transparent)

Walk-forward OOS rows are dominated by countries with longer WFP reporting histories (Philippines ≈ 58.7% of OOS rows). Thailand contributes few OOS rows (series ends ~March 2020; n ≈ 18 in WF tests). **MAE_macro** and per-country tables are reported so micro-averages are not mistaken for equal-country performance.

**Data source:** [WFP Global Food Prices (HDX)](https://data.humdata.org/) extract for ASEAN markets, 2015–2025.

**One-command reproduction (no 98 MB raw file required if `data/foodshock_asean_model_ready.csv` is present):**
```bash
python src/run_upgraded_v2.py
```

## Data provenance & FX features

- **Source:** WFP Global Food Prices via HDX (ASEAN extract 2015–2025); model-ready panel **27,950 × 39**, months spanning early 2016–late 2025 after lag construction.
- **Implied FX:** When the ~98 MB raw extract is present, `build_implied_fx()` recomputes `price/usdprice` medians by country-month. **Otherwise** the pipeline uses **precomputed** `fx_pct_1m/3m` and `fx_vol_6m` already stored in `foodshock_asean_model_ready.csv` (still derived from WFP local/USD prices — not a live external FX API).

```bash
python src/run_upgraded_v2.py   # full V4 recompute of CSVs + figures
```

## Operational decision flow (SDG Target 2.c)

1. **Detect** — FoodShock issues NORMAL / WASPADA / FOOD SHOCK ALERT from conformal P90 band + Stage-2 $P(\mathrm{Shock})$ (operational threshold) + predicted MoM rise.
2. **Route** — AFSIS Secretariat consolidates cross-country signals; national food agencies (stabilisation bodies / NFA-type operators) and trade ministries receive country–commodity alerts.
3. **Verify** — Cross-check warehouse stocks, border prices, and FX moves before acting.
4. **Act** — Targeted market operations, temporary distribution adjustments, or public communication — calibrated to alert severity, not to every residual wiggle.

