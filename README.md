# FoodShock — Cross-Country ASEAN Food Price Forecasting & Early Shock Detection System

![SDG 2](https://img.shields.io/badge/SDG-2%20Zero%20Hunger-green)
![Competition](https://img.shields.io/badge/MLSC%20Cyan%202026-EDM%20Laboratory-blue)
![Python](https://img.shields.io/badge/Python-3.12-blue)

**Machine Learning Solution Challenge (MLSC) – Cyan Academic Competition 2026**  
**Organizer:** Enterprise Data Management (EDM) Laboratory  
**Target SDG:** **SDG 2 — Zero Hunger**

FoodShock forecasts monthly food prices across ASEAN markets and issues early warnings for potential price shocks, supporting price stability monitoring and policy readiness under SDG 2.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── FoodShock_MLSC2026_Solution.ipynb   # Full solution notebook (tables & figures embedded)
├── src/
│   ├── foodshock_pipeline.py               # Audit ≥60 months, standardisation, feature engineering
│   └── run_residual_modeling.py            # Residual forecasting, walk-forward eval, SHAP
├── app.py                                  # Streamlit entry point (interactive web application)
├── data_loader.py
├── models.py
├── styles.py
├── utils.py
├── views/                                  # Streamlit pages
│   ├── home.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── training.py
│   └── prediction.py
├── data/                                   # Clean artefacts only (no raw >90 MB file)
│   ├── foodshock_asean_model_ready.csv
│   ├── fpma_wfp_country_coverage.csv
│   ├── fpma_wfp_commodity_coverage.csv
│   ├── model_evaluation_results.csv
│   ├── model_eval_by_commodity.csv
│   ├── model_eval_by_country.csv
│   ├── model_eval_by_pricetype.csv
│   └── model_eval_shock_classification.csv
├── figures/
│   ├── eda_trends.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── error_by_commodity.png
│   ├── shap_summary.png
│   └── shock_residual_dist.png
└── user_analysis_example/
    └── contoh_bulk_test.csv
```

> **Note on raw data:** The original WFP/HDX extract (`asean_wfp_2015_2025.csv`, ~98 MB) is intentionally **not** included so the repository stays within GitHub file-size limits. The cleaned, model-ready panel is available at `data/foodshock_asean_model_ready.csv`. Raw source: [WFP Food Prices – HDX](https://data.humdata.org/).

---


## Shock Threshold Justification (+5% MoM)

1. **Policy foundation:** Aligned with the WFP **ALPS (Alert for Price Spikes)** framework and ASEAN Food Security Information System (**AFSIS**) monitoring practice, where a month-over-month staple-food increase above **+5%** signals acute affordability stress (roughly equivalent to >60% annualised food inflation if sustained).
2. **Empirical foundation:** On the ASEAN panel return distribution (`target_pct_change_t1`), **P90 ≈ +4.44%** and **P95 ≈ +7.20%** (IQR ≈ [−1.89%, +1.77%]). The **+5.0%** rule therefore isolates the upper ~8.5–9% tail, and is combined with adaptive series-specific monitoring (μᵢ + 1.5σᵢ) and a **calibrated Stage-2 probability threshold** (optimal F1 threshold ≈ 0.35).

## Methodology Snapshot

| Component | Detail |
|-----------|--------|
| Scope | Series with ≥ 60 months; 7 ASEAN countries; 5 commodity groups (Rice, Chicken Meat, Beef Meat, Pork Meat, Cooking Oil) |
| Units | USD; KG (rice/meat) / L (cooking oil); explicit Retail vs Wholesale |
| Leakage control | Full monthly calendar re-index **without** forward-fill of raw prices; lags & rolling from `shift(1)` |
| Macro features | Implied FX (`fx_pct_1m/3m`, `fx_vol_6m`); ASEAN commodity momentum & country-vs-region gap |
| Target | Residual from **current** price $y_t$ (`usdprice`): $\Delta y_{t+1}=y_{t+1}-y_t$ → reconstruct $\hat{y}_{t+1}=y_t+\widehat{\Delta y}_{t+1}$ |
| Validation | Expanding-window walk-forward (6 annual folds, 2019–2025) |
| Shock rule | Predicted % change > +5% (or < −5%) → **FOOD SHOCK ALERT** |

**Best result (LightGBM Tuned Residual):** RMSE **0.2445** · R² **0.9890** (outperforms Persistence on RMSE & R²).

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
# or
jupyter lab notebooks/FoodShock_MLSC2026_Solution.ipynb
```

### 3. Batch Pipeline (optional re-run)
```bash
# Feature engineering & audit (expects raw CSV if re-running from scratch)
python src/foodshock_pipeline.py

# Residual modelling, evaluation, SHAP
python src/run_residual_modeling.py
```

### 4. Interactive Web Application (Streamlit)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

**Suggested exploration flow**
1. **Home** — metrics & official benchmark table  
2. **EDA** — interactive filters & price trends  
3. **Preprocessing & Audit** — coverage ≥ 60 months proof  
4. **Model Training** — live Residual vs Direct vs Persistence  
5. **Prediction & Shock Alert** — what-if case + bulk CSV + alert badges  

---

## Key Results (Expanding-Window Walk-Forward, 6 annual folds)

Validation: expanding train → next 12 months test at cutoffs 2019-06 … 2024-06. **All 7 ASEAN countries** appear in out-of-sample windows (including IDN & THA in earlier folds). Metrics are **micro-averaged** over observations; macro-MAE averages countries with equal weight.

| Model | MAE (micro) | RMSE (micro) | sMAPE | SS_RMSE vs Pers. | DM p-value |
|-------|-------------|--------------|-------|------------------|------------|
| **CatBoost WF Residual** | 0.0984 | **0.1931** | 3.06% | **+0.010** | **< 0.001** |
| **XGBoost WF Residual** | 0.0995 | **0.1930** | 3.09% | **+0.010** | **0.040** |
| LightGBM WF Residual | 0.0997 | 0.1937 | 3.14% | +0.007 | < 0.001 |
| Persistence (y_t) | 0.0986 | 0.1950 | 3.04% | 0.000 | — |
| HistGradientBoosting WF | 0.1015 | 0.1952 | 3.18% | ≈0 | 0.80 |
| RandomForest WF | 0.1017 | 0.1971 | 3.14% | −0.010 | — |

**Interpretation:** Monthly food prices are near-martingale; Persistence is a strong baseline on MAE. Value-add is evidenced by **lower RMSE**, positive **skill scores**, and **Diebold–Mariano significance** for gradient-boosted residual models, plus **quantile PICP ≈ 80%** (P10–P90) and a dedicated shock classifier.

### Stage-2 Shock Classifier (Walk-Forward OOF)

| Precision | Recall | F1 | ROC-AUC | PR-AUC | Base rate | Opt. threshold |
|-----------|--------|-----|---------|--------|-----------|----------------|
| 0.213 | **0.564** | **0.309** | 0.605 | 0.228 | 0.162 | 0.35 |

Recall and F1 substantially exceed a non-informative baseline; alerts use calibrated probabilities rather than a hard 5% rule on the point forecast alone.



## Limitations & Future Work

Point residual regression is conservative on rare tail shocks (>5%). Planned upgrades: quantile regression (P90/P95), dedicated shock classifier, and external drivers (FX, energy, weather, trade flows).

---

*FoodShock · MLSC 2026 · SDG 2 — Zero Hunger*
