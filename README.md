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
2. **Empirical foundation:** On the ASEAN panel return distribution (`target_pct_change_t1`), **P90 ≈ +4.44%** and **P95 ≈ +7.20%** (IQR ≈ [−1.89%, +1.77%]). The **+5.0%** rule isolates the upper ~8.5–9% tail, and is combined with adaptive series-specific monitoring (μᵢ + 1.5σᵢ) and a **calibrated Stage-2 probability threshold** (optimal F1 threshold ≈ 0.35).

---

## Architectural Rationale & Methodological Alignment

| Stage | What we do | Why it fits the problem |
|-------|------------|-------------------------|
| **1. Audit ≥ 60 months & unit standardisation** | Keep series with ≥ 60 months; convert to USD; KG (rice/meat) / L (oil) | Short series cannot support seasonal lags; mixed currencies/units create artificial scale bias |
| **2. Calendar-safe FE (zero leakage)** | Full monthly re-index **without** forward-fill of raw prices; lags/rolling from `shift(1)` | Forward-filling empty months invents 0% changes and leaks future levels into rolling stats |
| **3. Macro: implied FX & ASEAN spillover** | `fx_pct_1m/3m`, `fx_vol_6m` from price/usdprice; regional commodity momentum + country gap | Local food inflation often tracks FX depreciation and regional supply shocks, not only local lags |
| **4. Residual target** \(\Delta y_{t+1}=y_{t+1}-y_t\) from **current** \(y_t\) (`usdprice`) | Train on changes; reconstruct \(\hat y_{t+1}=y_t+\widehat{\Delta y}\) | Rice (~USD 0.8) vs beef (~USD 6–7) differ ~9×; level targets bias trees toward expensive goods |
| **5. 6-fold expanding-window walk-forward + DM test** | Cutoffs 2019-06…2024-06; 12-month test each; micro & macro metrics; Diebold–Mariano vs Persistence | All **7 countries** (incl. IDN, THA) receive OOS evaluation where data exist; significance is tested, not assumed |
| **6. Two-stage risk layer** | Point residual + Quantile **P10–P90** (PICP ≈ **80.3%**) + balanced **Stage-2 classifier** (Recall ≈ **56.4%**, F1 ≈ **0.309**, thr ≈ 0.35) | Point forecasts regress to the mean on rare shocks; probability + upside interval operationalise ALPS/AFSIS alerts |
| **7. 5-page Streamlit app** | Home, EDA, Preprocessing, Training, Prediction (what-if + bulk CSV) | Turns the pipeline into a usable early-warning tool for analysts |

---

## Repository Structure (42 entries)

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
│   └── prediction.py                  # What-if & bulk CSV with P10–P90 and P(Shock)
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
└── user_analysis_example/
    └── contoh_bulk_test.csv           # Sample bulk prediction CSV
```

> **Raw data note:** The original WFP/HDX extract (`asean_wfp_2015_2025.csv`, ~98 MB) is **not** shipped (GitHub size limits). The model-ready panel is `data/foodshock_asean_model_ready.csv`. Source: [WFP Food Prices – HDX](https://data.humdata.org/).

---

## Key Results (Expanding-Window Walk-Forward, 6 annual folds)

Validation: expanding train → next 12 months at cutoffs 2019-06 … 2024-06. **All 7 ASEAN countries** appear in out-of-sample windows (including IDN & THA on folds where data exist). Metrics are **micro-averaged**; macro-MAE averages countries with equal weight.

| Model | MAE (micro) | RMSE (micro) | sMAPE | SS_RMSE vs Pers. | DM p-value |
|-------|-------------|--------------|-------|------------------|------------|
| **CatBoost WF Residual** | 0.0984 | **0.1931** | 3.06% | **+0.010** | **< 0.001** |
| **XGBoost WF Residual** | 0.0995 | **0.1930** | 3.09% | **+0.010** | **0.040** |
| LightGBM WF Residual | 0.0997 | 0.1937 | 3.14% | +0.007 | < 0.001 |
| Persistence (y_t) | 0.0986 | 0.1950 | 3.04% | 0.000 | — |
| HistGradientBoosting WF | 0.1015 | 0.1952 | 3.18% | ≈0 | 0.80 |
| RandomForest WF | 0.1017 | 0.1971 | 3.14% | −0.010 | — |

**Interpretation:** Monthly food prices are near-martingale; Persistence is strong on MAE. Value-add is shown by **lower RMSE**, positive **skill scores**, **Diebold–Mariano significance** for boosted residual models, **quantile PICP ≈ 80%**, and the Stage-2 classifier.

### Stage-2 Shock Classifier (Walk-Forward OOF)

| Precision | Recall | F1 | ROC-AUC | PR-AUC | Base rate | Opt. threshold |
|-----------|--------|-----|---------|--------|-----------|----------------|
| 0.213 | **0.564** | **0.309** | 0.605 | 0.228 | 0.162 | 0.35 |

### Stratified OOS counts (walk-forward)

- **Retail** n = 22,506 · **Wholesale** n = 1,385  
- Countries with OOS rows: IDN, KHM, LAO, MMR, PHL, THA, TLS  

---

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
5. **Prediction & Alert** — what-if + bulk CSV + P10–P90 + P(Shock)  

---

## Limitations & Future Work

Monthly staple prices behave like a near-martingale: Persistence remains competitive on MAE. The system’s value is concentrated in **RMSE skill**, **Diebold–Mariano significance**, **P10–P90 coverage (PICP ≈ 80%)**, and **Stage-2 recall on rare shocks** rather than large mean-error gaps.

Current limitations include sparse coverage for some country–commodity cells (e.g., THA, MMR), the absence of harmonised weather/energy covariates across all seven countries, and still-moderate precision on the shock classifier. Future work: richer external drivers, conformal or quantile ensembles, and cost-sensitive thresholding for operational AFSIS dashboards.

---

*FoodShock · MLSC 2026 · SDG 2 — Zero Hunger*
