# FoodShock V9
**Cross-Country ASEAN Food Price Forecasting & Early Shock Detection**  
Machine Learning Solution Challenge 2026 · Cyan Academic Competition · Preliminary round  
SDG 2 (Zero Hunger), target 2.c: market information and extreme food-price volatility

## 1. Problem and intended use

A sudden increase in staple-food prices can reduce household purchasing power and complicate market-stabilization decisions. FoodShock is a historical market-screening prototype for food-price monitoring teams, such as national food agencies and regional information coordinators. It forecasts a market/commodity's next monthly USD price and prioritizes possible upward price shocks for investigation. It is not a live food-security alert service, a causal food-insecurity model, or a substitute for local-price and affordability analysis.

A shock is a next-month price increase strictly greater than 5%, with a 1e-12 numerical tolerance to prevent floating-point misclassification at equality. This is FoodShock's empirical upper-tail rule: the supplied panel's P90 monthly return is about +4.44%, and P95 is +7.20%. The full-panel quantiles are descriptive, not a claim that the threshold was optimized prospectively. The +5% definition is fixed throughout evaluation; it is not the official WFP ALPS formula. Intended action: normal = routine monitoring, elevated = review, alert = prioritize independent investigation. The warning tier is illustrative; binary metrics evaluate the alert tier only.

## 2. Submission quick start

The ZIP includes the dataset, canonical source, executed notebook, actual V9 evaluation tables and figures, eight frozen CatBoost/quantile/classifier bundles, and integration tests. No network or raw download is required for benchmark reproduction after installing dependencies.

```bash
# From the extracted project directory (Python 3.13.5 was used to build the artifacts)
python -m venv cac2026-venv
# Windows PowerShell: cac2026-venv\Scripts\Activate.ps1
# macOS/Linux: source cac2026-venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

The official benchmark is already generated; rerunning it is optional. To reproduce everything using the canonical panel:

```bash
python src/run_v9.py --trials 25
python src/build_report.py
python -m unittest discover -s tests -v
```

Use the exact dependency versions in `requirements.txt` for artifact compatibility. The tested Python/library versions and file hashes are in `data/v9_manifest.json`. Restart Streamlit after regenerating artifacts. The optional fast `--trials 1` is for smoke experimentation only and overwrites results if run in the submission directory; use a copy of the project. The packaged benchmark and notebook were produced with 25 trials, not the smoke configuration. Never load pickle/joblib artifacts obtained from an untrusted source.

## 3. Data and feature engineering

The packaged model-ready dataset contains 27,950 rows and 39 columns from February 2016–November 2025, 582 market series, seven countries (Indonesia, Cambodia, Laos, Philippines, Myanmar, Thailand, Timor-Leste), and five staple commodity groups (rice, chicken, beef, pork, cooking oil). Prices are USD per KG (rice/meat) or per L (oil). Retail and wholesale are separate. Monthly series were retained with at least 60 months of source coverage and reindexed to a monthly calendar without forward-filling raw prices. The original raw WFP snapshot downloaded from [Kaggle](https://www.kaggle.com/datasets/abhishekgupta56447/global-food-prices-database-wfp) is not bundled; see [DATA_SOURCES.md](docs/DATA_SOURCES.md) for attribution, reproducible model-ready path, optional raw rebuild, and unresolved provenance details.

The preprocessing ensures `country_vs_asean_mom` equals `pct_change_1m − asean_comm_mom_1m`. Observed prices and supplied shock labels are preserved (numeric values checked within floating-point tolerance). No synthetic rows are added to competition training or testing. The integration tests use one synthetic fixture solely to test optional raw preprocessing.

The 23 numeric features and three train-fitted categorical encodings are declared in `core/utils.py`. Target columns are never in the feature list. At origin t, the observed USD price is available, and lag/rolling/momentum features use t−1 or earlier; FX proxies are inherited from country-month WFP local/USD price ratios and assumed observable at t. ASEAN momentum is lagged in the supplied panel. The monthly rolling statistics, seasonal encoding, and scenario feature formulas share `core/features.py`. Raw rebuild uses the same formulas and computes regional momentum before target filtering with calendar joins.

Important limitation: the supplied panel was retrospectively selected based on series length and has only labeled rows; raw availability/reporting delays and FX construction cannot be independently audited without the original extract. This evaluation therefore controls model/label temporal leakage but is not a guarantee of fully prospective data collection or real-time feature availability. Philippines accounts for ~58% of observations; Thailand/Myanmar have sparse OOS coverage. Report micro, macro and subgroup results rather than treating countries equally represented.

## 4. Shared V9 methodology

### Temporal evaluation

Seven expanding forecast-origin test windows start in June 2019, 2020, 2021, 2022, 2023, 2024, and 2025. Each covers 12 months or the remaining available data (last window ends November 2025). Outer training observations must have `origin + 1 calendar month < test cutoff`. The final ~20% of outer-training months form a calibration block, with the same strict target-date purge between fit and calibration. Tuning uses a further purged chronological validation split within the fit block only. Encoders fit on training data only; unseen categories map to −1 and remain in OOS evaluation. No row-wise random fallback is permitted.

### Stage 1 — residual forecast

`delta = price(t+1) − price(t)` and `predicted_price = max(0, price(t) + predicted_delta)`.

Compare CatBoost, LightGBM, XGBoost, HistGradientBoosting and Random Forest with the persistence forecast `price(t)`. CatBoost/LightGBM/XGBoost each receive 25 Optuna TPE trials per fold (seed 42): 525 trials across seven folds, plus 25 for the frozen 18-month holdout model. HistGradientBoosting and Random Forest use documented fixed parameters. All algorithms have the same fit/calibration/test boundaries. The point model is fitted on the fit block and **not refitted after calibration**. CatBoost is the pre-specified system architecture, not relabeled as the best overall model if persistence wins.

### Interval layer

Two LightGBM quantile residual regressors (alpha .1 and .9) are fitted on the same fit block, then frozen. The calibration score is `max(lower−actual, actual−upper)` on residuals. The 80% finite-sample order statistic, floored at zero, expands the ordered quantile bounds; prices are clipped at zero. Calibration and inference use the same model and encoder. Report empirical prediction interval coverage (PICP) and mean width. Panel/time dependence violates the simple IID exchangeability assumption; there is no unconditional coverage guarantee. Direct-mode custom models have no validated interval and the app displays none.

### Stage 2 — upward shock screening

A regularized, balanced LightGBM classifier produces a **risk score**, not a calibrated event probability. The deployed binary hybrid decision is `score >= fold_threshold OR predicted_change > 0.05` (with the numerical tolerance above). Both the point model and classifier remain frozen during threshold selection on calibration data. The threshold grid is .05–.95; maximize recall subject to calibration precision >=20%, otherwise maximize F0.5. The precision constraint is a validation objective, not a promise on OOS. The warning tier is separate. The app reads the exact saved model-specific threshold, never the mean threshold across folds.

Compare hybrid decision, score-only, forecast-only, lagged momentum, validation-thresholded historical group risk, and logistic regression. Report precision, recall, F1, FPR, FDR, lift, confusion matrix counts, alerts/month, and month-resampling CIs. PR-AUC and ROC-AUC are attached only to continuous ranking scores; they are not assigned to the hybrid binary rule. For custom holdout metrics, `PR_AUC`/`ROC_AUC` refer to Stage-2 score ranking.

### Evaluation and interpretation

Report MAE, RMSE, sMAPE, price R², residual R², equal-country macro MAE, skill versus persistence, month-level HAC Diebold–Mariano statistic/p-value and whole-month bootstrap skill intervals. DM uses equally weighted month losses; pooled RMSE weights each market row, so their directions can differ. Month bootstrap preserves within-month panel dependence but does not model serial dependence between resampled months. Ablation uses identical fold splits and each fold's full-model CatBoost parameters while adding feature groups (lag/current, rolling/seasonal, FX, regional). It is a controlled comparison, not an OOS-selected model. SHAP explains the exact frozen official holdout CatBoost on a deterministic holdout sample, not a separately retrained proxy.

See [RESULTS.md](docs/RESULTS.md) and `notebooks/FoodShock_MLSC2026_Solution.ipynb` for the actual generated numbers, interpretation, limitations, and reproducible inference example. In V9, persistence slightly beats CatBoost on pooled RMSE; logistic regression also has slightly higher F1 than the hybrid. Neither the system's benefit nor the +5% alerts are guaranteed. The stronger contribution is the auditable uncertainty/risk decision-support workflow, not a claim of universal prediction superiority.

## 5. Streamlit application modes

- Home: dynamically reads the V9 official benchmark and hybrid evaluation (with the Stage-2 PR-AUC separately labeled).
- EDA / Preprocessing: reads the same packaged model-ready dataset, displays coverage and explains temporal assumptions.
- Model Training: seven-fold results, ablation, exact-model SHAP and official holdout metadata; custom training uses the same engine and a separate 18-month holdout.
- Prediction / Historical out-of-sample: select country, commodity, market, series and origin month; load that fold's frozen model, infer on the corresponding feature row, and verify against the stored benchmark before displaying the actual next-month value. Graph uses target-month alignment.
- Official holdout model what-if: uses the frozen CatBoost model for the June 2024–November 2025 holdout, with its own calibration and threshold. This is not a model retrained through the last observed month for live 2026 forecasts.
- Custom model what-if: uses the session's complete model bundle, its own threshold, and only category combinations seen during fit. Direct mode does not invent intervals. Basic what-if changes the current price only; advanced what-if constructs a coherent synthetic past-price history and recalculates all dependent lag/rolling/momentum/seasonal/regional features. FX is held fixed. Synthetic defaults are explicitly labeled.

The same `core/models.py`, `core/utils.py`, `core/features.py`, and `core/model_store.py` are used by evaluation and the application. Hash checks guard the packaged data, evaluation CSVs, and loaded model files. The UI is intentionally decision support, not a live API. Full browser execution was not available in the build environment because Streamlit could not be installed; core model/inference parity and integration tests were executed successfully.

## 6. Files for reviewers

| File | Purpose |
|---|---|
| `src/run_v9.py` | Canonical full benchmark, ablation, SHAP and frozen model generation |
| `src/prepare_data.py` | Correct supplied panel or optionally reconstruct from original raw WFP extract |
| `core/models.py`, `core/utils.py`, `core/features.py` | Shared temporal engine, model inference, alert and feature formulas |
| `core/model_store.py` | Dataset, evaluation and model artifact verification |
| `data/foodshock_asean_model_ready.csv` | Required packaged model-ready dataset |
| `data/v9_manifest.json` | Exact versions, dataset/model/result hashes, official fit and calibration metadata |
| `data/v9_oos_predictions.csv` | Actual per-row predictions for 5 ML algorithms across seven folds |
| `data/v9_official_holdout_predictions.csv` | Actual rows from separate 18-month official model holdout |
| `data/fold_models/`, `data/v9_official_bundle.joblib` | Frozen fold/holdout model artifacts |
| `data/model_evaluation_results.csv`, `data/model_eval_*` | Regression, subgroup, and classification evaluation |
| `data/v9_fold_metrics.csv`, `data/hyperparameter_tuning_summary.csv` | Fold dates, fit/calibration/test counts, validation threshold, tuning metadata |
| `data/v9_feature_ablation.csv`, `data/v9_shap_importance.csv` | Feature contribution and exact-model interpretability |
| `notebooks/FoodShock_MLSC2026_Solution.ipynb` | Executed technical narrative and reproducible demonstration |
| `app.py`, `application/` | Streamlit application |
| `tests/test_v9.py`, `docs/VALIDATION_REPORT.md` | Executed integration tests and known limitations |
| `docs/DATA_SOURCES.md` | Dataset provenance |

## 7. Preliminary-round scope

The supplied guidebook requires source code and dataset for preliminary judging; presentation and live demo are final-round items. This package focuses on the five preliminary criteria (problem formulation, data quality, ML implementation, evaluation, innovation/impact). It does not claim guaranteed qualification, verified real-world impact, or official panitia scoring. Confirm the exact raw WFP resource/license and applicable competition rules before submitting. 
