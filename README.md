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

## Methodology Snapshot

| Component | Detail |
|-----------|--------|
| Scope | Series with ≥ 60 months; 7 ASEAN countries; 5 commodity groups (Rice, Chicken Meat, Beef Meat, Pork Meat, Cooking Oil) |
| Units | USD; KG (rice/meat) / L (cooking oil); explicit Retail vs Wholesale |
| Leakage control | Full monthly calendar re-index **without** forward-fill of raw prices; lags & rolling from `shift(1)` |
| Target | Residual: $\Delta y_{t+1}=y_{t+1}-y_t$ → reconstruct $\hat{y}_{t+1}=y_t+\widehat{\Delta y}_{t+1}$ |
| Validation | Time-based hold-out (last 18 months) |
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

## Key Results (Official Benchmark)

| Model | MAE | RMSE | sMAPE | R² |
|-------|-----|------|-------|-----|
| Persistence (lag_1) | 0.1479 | 0.2472 | 3.81% | 0.9887 |
| **LightGBM_Tuned_Residual** | **0.1509** | **0.2445** | 3.91% | **0.9890** |
| RandomForest (residual) | 0.1581 | 0.2491 | 4.08% | 0.9886 |
| XGBoost / CatBoost / HistGB / LGBM | 0.161–0.164 | 0.255–0.258 | ≈4.2% | ≈0.988 |

Stratified metrics (by commodity, country, Retail vs Wholesale) and shock classification scores are in `data/model_eval_*.csv`.

---

## Limitations & Future Work

Point residual regression is conservative on rare tail shocks (>5%). Planned upgrades: quantile regression (P90/P95), dedicated shock classifier, and external drivers (FX, energy, weather, trade flows).

---

*FoodShock · MLSC 2026 · SDG 2 — Zero Hunger*
