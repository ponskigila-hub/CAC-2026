# Results

All figures are derived from the CSVs in `results/`. Validation and retrospective evaluation use four origins each. Predictions of the same target from different origins are separate forecasting cases, not independent events. 2023–2025 was previously inspected in this project, so this is not a newly untouched test set.

## Price forecasting

| Model | Validation MAPE | Retrospective MAPE |
|---|---:|---:|
| Persistence | 15.515% | 12.635% |
| Seasonal Naive | 16.879% | 16.599% |
| Ridge | 19.141% | 16.145% |
| Histogram Gradient Boosting | 16.571% | 14.001% |
| HGB blend 25% | 15.310% | 12.279% |
| HGB blend 50% | 15.440% | 12.390% |
| HGB blend 75% | 15.863% | 12.984% |
| Automatic (by horizon) | 14.652% | 13.684% |

319,728 validation pairs; 224,336 retrospective price pairs. The validation-optimal automatic policy selects HGB blend 25% at 1–3 and 13–36 months, Seasonal Naive at 4–6 and 7–12 months. This policy does not beat Persistence on aggregate retrospective MAPE. The fixed HGB blend 25% is the lowest-MAPE single candidate on validation, and is also the retrospective winner, reducing MAPE relative to Persistence by 2.81%. This is a modest gain, not a universal advantage. It is exposed as a selectable alternative. At horizon 36, unblended HGB achieves 12.648% MAPE vs 16.418% Persistence; the HGB blend 25% achieves 15.270%. The 36-month evaluation only has 1,638 pairs, from the earliest retrospective origin.

## Shock classification

| Model | AP | ROC-AUC | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| HGB (selected) | 0.3191 | 0.7012 | 38.33% | 22.22% | 28.13% | 7.64% |

217,627 retrospective pairs with known labels. TP 8,510; FP 13,694; FN 29,791; TN 165,632. The fraction of false alarms among alerts is 61.67%, distinct from FPR. The HGB policy has validation FPR 9.90% and recall 28.95%. Test FPR by group is 12.62% (1–3), 7.91% (4–6), 6.80% (7–12), 5.83% (13–36). The short horizon still exceeds the validation FPR target; most actual shocks are missed. The score is not a calibrated probability.

## Fair comparison with previous Forecasting Edition

For the identical Dec 2022 origin (104,875 price pairs, 102,621 known shock labels), old vs new HGB MAPE is 16.863% vs 16.776%; fixed HGB blend 25% gives 13.529% vs Persistence 13.864%. Automatic gives 16.202%. The old HGB shock policy had FPR 8.51%, precision 40.57%, recall 25.24%; new horizon-aware HGB yields FPR 10.26%, precision 39.06%, recall 28.58%. Thus false alarm performance did not improve at the matched origin. Pooled new FPR of 7.64% uses a different evaluation population and must not be represented as a direct improvement. Old scores are historical comparison context only; see current `rolling_origin.csv` and `rolling_risk.csv` for comparable new rows.

## Diagnostics and deployment

The full 17-feature configuration has higher AP than both reduced configurations at each validation origin under the HGB standard configuration. MAPE is mixed; feature importance is not causal. The last validation sample shows season cosine and three-month volatility as the leading AP permutation-importance features. Sampling sensitivity is mixed (30k cap MAPE 14.06%, AP 0.4202; 90k cap MAPE 13.64%, AP 0.3948). Retrospective empirical interval coverage for Automatic is 83.48%, 82.26%, 81.96%, 88.77% across groups. Deployment model refit uses data through 2025 and adapts thresholds and empirical quantiles from OOF history; no independent future test or calibration guarantee is claimed.

## Example forecast

Pasar Mundu, Rice (medium quality), KG, Jawa Barat, target January 2028 (25 months from December 2025), Automatic mode: Rp15,528/kg, monthly change +0.11%, risk score 0.0602, threshold 0.3726, non-shock. Empirical historical-error range Rp13,702–21,342/kg. Not an observed future price. See `example_jan2028.csv`.
