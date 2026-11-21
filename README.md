# FoodShock — Cross-Country ASEAN Food Price Forecasting & Early Shock Detection System

![SDG 2](https://img.shields.io/badge/SDG-2%20Zero%20Hunger-green)
![Competition](https://img.shields.io/badge/MLSC%20Cyan%202026-EDM%20Laboratory-blue)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**Machine Learning Solution Challenge (MLSC) – Cyan Academic Competition 2026**  
**Organizer:** Enterprise Data Management (EDM) Laboratory  
**Cabang:** Machine Learning Solution Challenge (EDM Lab)  
**Target SDG:** **SDG 2 — Zero Hunger** (stabilitas pasokan & keterjangkauan harga pangan pokok di Asia Tenggara)

### Identitas Tim
| Peran | Nama | Institusi / Kontak |
|-------|------|--------------------|
| **Tim** | **PenaklukFCMobile** | Universitas Bina Nusantara |
| **Ketua** | Nicholas Driyadis Tjoe | 089602767594 |
| **Anggota** | Christian Caesar | Universitas Bina Nusantara |
| **Anggota** | Gerald Amadeo Passover Sutanto | Universitas Bina Nusantara |

---

## Executive Summary & Problem Statement

Stabilitas harga pangan pokok (beras, daging, minyak goreng) merupakan fondasi ketahanan pangan rumah tangga di ASEAN. Fluktuasi harga yang tajam mengurangi daya beli, meningkatkan risiko gizi buruk, dan dapat memicu instabilitas sosial. Data WFP/HDX 2015–2025 menunjukkan heterogenitas harga yang tinggi antar negara dan antar jenis pasar (Retail vs Wholesale).

**FoodShock** menjawab tantangan tersebut dengan:
1. Sistem *forecasting* panel lintas negara yang akurat untuk horizon t+1.
2. Deteksi dini *Food Price Shock* berbasis residual forecasting + skor klasifikasi.
3. Interpretabilitas (SHAP) yang mendukung rekomendasi kebijakan konkret bagi AFSIS dan kementerian nasional.

---

## Repository Structure

```
.
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Standard Python/DS ignore rules
├── FoodShock_MLSC2026_Solution.ipynb      # Main solution notebook (narrative + code + outputs)
├── foodshock_pipeline.py                  # Full end-to-end pipeline (audit → features → models)
├── run_residual_modeling.py               # Focused residual-forecasting modeling script
├── asean_wfp_2015_2025.csv                # Raw WFP/HDX ASEAN food price data (2015–2025)
├── foodshock_asean_model_ready.csv        # Clean, feature-engineered panel (model-ready)
├── fpma_wfp_country_coverage.csv          # Audit: country-level coverage (≥60 months)
├── fpma_wfp_commodity_coverage.csv        # Audit: commodity-group coverage
├── model_evaluation_results.csv           # Overall model comparison (MAE, RMSE, sMAPE, R²)
├── model_eval_by_commodity.csv            # Stratified metrics by commodity group
├── model_eval_by_country.csv              # Stratified metrics by country
├── model_eval_by_pricetype.csv            # Stratified metrics Retail vs Wholesale
├── model_eval_shock_classification.csv    # Precision / Recall / F1 / ROC-AUC for shock detection
└── figures/
    ├── eda_trends.png
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    ├── shap_summary.png
    ├── error_by_commodity.png
    └── shock_residual_dist.png
```

---

## Methodology & Data Integrity

### Data Audit & Scope
- Series retained only if **≥ 60 months** of observations.
- **7 ASEAN countries:** IDN, KHM, LAO, PHL, THA, TLS, MMR.
- **5 standardized commodity groups** (supported by ≥ 3 countries): Rice, Chicken Meat, Beef Meat, Pork Meat, Cooking Oil.
- Prices in **USD**; units standardized (KG for rice/meat, L for cooking oil).
- Explicit separation of `pricetype` (Retail / Wholesale) and `priceflag`.

### Zero Data Leakage
- Full monthly calendar re-indexing **without forward-filling raw prices**.
- All lag and rolling features computed from `shift(1)` → features at time *t* use only information up to *t−1*.
- Time-based (walk-forward) split: last 18 months held out as final test set.

### Residual Forecasting Formulation
To remove level bias between low-price commodities (Rice ≈ USD 0.77) and high-price commodities (Beef ≈ USD 6.7):

$$
\Delta y_{t+1} = y_{t+1} - y_t
$$

$$
\hat{y}_{t+1} = y_t + \widehat{\Delta y}_{t+1}
$$

Models are trained on the residual target and predictions are reconstructed to absolute price scale for evaluation.

### Models Compared
- Baselines: Persistence (lag_1), Seasonal Naive (lag_12)
- Tree ensembles: Random Forest, XGBoost, LightGBM, CatBoost, HistGradientBoosting
- Tuned: LightGBM + Optuna (TimeSeriesSplit)

---

## Key Experimental Results

### Overall Model Comparison (Absolute Price after Residual Reconstruction)

| Model                        | MAE    | RMSE   | sMAPE  | R²     |
|-----------------------------|--------|--------|--------|--------|
| Persistence (lag_1)         | 0.1479 | 0.2472 | 3.81%  | 0.9887 |
| **LightGBM_Tuned_Residual** | **0.1509** | **0.2445** | 3.91% | **0.9890** |
| RandomForest                | 0.1581 | 0.2491 | 4.08%  | 0.9886 |
| XGBoost                     | 0.1615 | 0.2561 | 4.14%  | 0.9879 |
| HistGradientBoosting        | 0.1627 | 0.2556 | 4.20%  | 0.9880 |
| CatBoost                    | 0.1639 | 0.2562 | 4.22%  | 0.9879 |
| LightGBM                    | 0.1641 | 0.2580 | 4.20%  | 0.9877 |
| Seasonal Naive (lag_12)     | 0.3142 | 0.4756 | 9.15%  | 0.9583 |

**Key takeaway:** Residual forecasting brings ML models within striking distance of the strong Persistence baseline on MAE, while **outperforming it on RMSE and R²**.

### Stratified Performance (LightGBM_Tuned_Residual)

**By Commodity Group** (selected): Rice and Cooking Oil show the lowest absolute errors; meat products are more volatile.

**By Price Type:**
| Price Type | MAE    | RMSE   | sMAPE  | n    |
|------------|--------|--------|--------|------|
| Retail     | 0.1527 | 0.2468 | 3.88%  | 6812 |
| Wholesale  | 0.0789 | 0.1237 | 5.18%  | 171  |

### SHAP Key Insights
- After residual modeling, momentum (`pct_change_*`), volatility (`rolling_std_6`, `cv_6`), and seasonality (`month_sin` / `month_cos`) gain prominence alongside `lag_1`.
- Country encoding captures structural heterogeneity across ASEAN markets.

### Food Price Shock Classification
- Definition: |%Δ| > 5% **or** residual > 1.5 × historical training std.
- Metrics reported in `model_eval_shock_classification.csv` (Precision, Recall, F1, ROC-AUC).

---

## Limitations & Future Work

Regression-based residual models are inherently conservative (*regression-to-the-mean*). Consequently, detection of extreme tail shocks (>5%) currently yields modest Precision (≈25%) and low Recall. This is an empirical limitation of one-step-ahead point forecasting on rare events.

**Planned improvements:**
- Quantile regression (LightGBM / CatBoost quantile loss) to target P90 / P95 residual directly.
- Dedicated two-stage or binary classifier for the shock label.
- Incorporation of external drivers (exchange rates, energy prices, weather anomalies, trade flows).

---

## How to Reproduce

### 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Full Pipeline (from raw data)
```bash
python foodshock_pipeline.py
```

### 3. Residual Modeling Only (uses existing model-ready CSV)
```bash
python run_residual_modeling.py
```

### 4. Explore Results
Open `FoodShock_MLSC2026_Solution.ipynb` in Jupyter Lab / VS Code / GitHub.

---

## Citation & Acknowledgement

Dataset: World Food Programme (WFP) / Humanitarian Data Exchange (HDX) – Food Prices.  
Competition: Machine Learning Solution Challenge (MLSC) – Cyan Academic Competition 2026, Enterprise Data Management (EDM) Laboratory.

**Team:** PenaklukFCMobile — Universitas Bina Nusantara  
**Ketua:** Nicholas Driyadis Tjoe (089602767594)  
**Anggota:** Christian Caesar · Gerald Amadeo Passover Sutanto

---

*Built for SDG 2 — Zero Hunger. Designed for transparency, reproducibility, and policy relevance.*
