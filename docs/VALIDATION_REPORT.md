# V9 validation report

Executed against the submitted data and model artifacts with the numerical versions in `data/v9_manifest.json`.

- Full benchmark completed: 7 folds × 5 ML algorithms, 525 Optuna trials for 3 tuned models, plus 25 official holdout trials. OOS sample: 26,236 unique market-month forecast origins. Generated all evaluation tables and plots from the run.
- All 9 automated integration tests passed. Tests cover dataset schema, finite features, duplicate keys, exact label policy, target-date purges, 56 historical fold-model prediction replays (8 per fold), 12 official holdout replays, aggregation/policy parity across all 26,236 CatBoost OOS rows, boundary behavior, coherent scenarios, real custom residual and direct training, and synthetic raw-schema preprocessing fixture.
- Real prediction replays compare point forecast, lower/upper interval, Stage-2 score, threshold and binary alert against the stored outputs; tolerances are 1e-8 for numeric values. Checksums guard the exact dataset, evaluation CSVs and frozen model artifacts.
- Source syntax compilation passed. The V9 notebook was executed successfully: 10/10 code cells completed with no error output.

## Not verified

The declared source is the Kaggle Global Food Prices Database (WFP); the original raw extract is absent, so its snapshot date, license and raw-to-model-ready reproduction cannot be confirmed. `src/prepare_data.py --raw` was tested against a synthetic fixture, not a substitute source for the competition data.

Streamlit was not installed in the available execution environment and a dependency installation attempt could not obtain it. Consequently, real browser/page interaction and clean-venv installation were not executed here. The application source, data/model integration, and core custom training/inference paths were checked. Before submission, run `python -m streamlit run app.py` locally and exercise Home, EDA, Preprocessing, Model Training, Historical OOS, Official What-if, and Custom What-if. Check that all pages render and that historical replay reports a match. The provided frozen artifacts avoid needing to retrain to view the app.

See RESULTS.md for benchmark limitations.
