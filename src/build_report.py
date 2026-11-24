"""Regenerate RESULTS.md from the actual benchmark CSVs and manifest."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'

def run():
    m=json.loads((DATA/'v9_manifest.json').read_text()); r=pd.read_csv(DATA/'model_evaluation_results.csv'); s=pd.read_csv(DATA/'model_eval_shock_classification.csv'); a=pd.read_csv(DATA/'v9_feature_ablation.csv'); c=pd.read_csv(DATA/'model_eval_by_country.csv'); p=pd.read_csv(DATA/'foodshock_asean_model_ready.csv')
    cat=r.set_index('model').loc['CatBoost']; pers=r.set_index('model').loc['Persistence (y_t)']; hy=s.set_index('method').loc['Hybrid_Operational_Alert']; st=s.set_index('method').loc['Stage2_Only']; lr=s.set_index('method').loc['LogisticRegression']; hold=m['official_holdout_metrics']
    def table(df,cols): return df[cols].round(5).to_markdown(index=False)
    text=f'''# V9 benchmark results (generated from saved outputs)

This report describes the actual run on the packaged model-ready panel. Dataset SHA-256: `{m['dataset_sha256']}`. Seed {m['seed']}; {m['trials_per_tuned_model']} Optuna trials per tuned algorithm per fold. Seven non-overlapping walk-forward test windows cover {m['oos_rows']:,} distinct market-month forecast origins. The separate official holdout is reported separately and is not added to the seven-fold sample.

## Regression benchmark

{table(r,['model','n','MAE_micro','RMSE_micro','sMAPE_micro','R2_micro','R2_delta','MAE_macro','SS_RMSE','DM_pvalue','SS_RMSE_CI95'])}

CatBoost RMSE = **{cat.RMSE_micro:.5f} USD** vs persistence **{pers.RMSE_micro:.5f} USD**; pooled skill = **{cat.SS_RMSE:+.2%}**. A positive skill would mean improvement; the sign here is {'positive' if cat.SS_RMSE>0 else 'non-positive'}. The month-bootstrap skill CI is {cat.SS_RMSE_CI95}; month-level HAC DM p={cat.DM_pvalue:.3f}. High price-level R² ({cat.R2_micro:.4f}) must not be interpreted as percentage accuracy: residual R² is {cat.R2_delta:.4f}. The point model does not establish statistically reliable superiority to persistence on this evaluation. The DM loss uses equal-month averages rather than pooled row weights.

![Model comparison](../figures/model_comparison.png)

## Upward shock detection (hybrid alert vs ranking baselines)

{table(s,['method','Precision','Recall','F1','FPR','PR_AUC','ROC_AUC','TP','FP','FN','TN','Lift','Alerts_Per_Month'])}

Hybrid: precision **{hy.Precision:.2%}**, recall **{hy.Recall:.2%}**, FPR **{hy.FPR:.2%}**, base shock rate {hy.Base_Rate:.2%}, lift {hy.Lift:.2f}×, about {hy.Alerts_Per_Month:.1f} alerts per observed forecast month. True alerts: {int(hy.TP):,}; false alerts: {int(hy.FP):,}; missed shocks: {int(hy.FN):,}. Month-bootstrap precision CI {hy.Precision_CI95}, recall CI {hy.Recall_CI95}. Stage-2 ranking PR-AUC {st.PR_AUC:.3f}, ROC-AUC {st.ROC_AUC:.3f}; these are score metrics, not hybrid binary AUC. Logistic regression achieves F1 {lr.F1:.4f} vs hybrid {hy.F1:.4f}, so a simpler baseline remains competitive. Use alerts only to prioritize investigation, not automate interventions.

Empirical interval coverage **{hy.Overall_PICP_P10_P90:.2%}** (target 80%); mean width {hy.Mean_Interval_Width:.4f} USD. Time dependence means coverage is not guaranteed in future markets.

![Alert comparison](../figures/shock_classification.png)

## Controlled feature ablation

{table(a,['feature_set','n_numeric_features','MAE_micro','RMSE_micro','SS_RMSE'])}

All ML ablations include the three encoded category columns and share the full model's fold-specific tuned parameters. FX and regional features can be inspected independently; the table does not support claiming that every added feature improves pooled RMSE. Do not use these OOS results to select a new model and then report the same sample as an independent final test.

![Ablation](../figures/feature_ablation.png)

## Country coverage and errors (CatBoost)

{table(c,['countryiso3','n','MAE_micro','RMSE_micro','Persistence_RMSE','SS_RMSE'])}

Philippines dominates the panel; sparse OOS markets have highly uncertain subgroup scores. Equal-country macro MAE is reported in the regression table.

## Official frozen holdout model (separate from aggregated OOS)

- Algorithm: {m['official_model']}; fit through {m['official_fit_end']}; calibration {m['official_cal_start']} to {m['official_cal_end']}; holdout begins {m['official_holdout_cutoff']}.
- Params: `{m['official_params']}`; threshold = {m['official_threshold']:.3f}, qcal = {m['official_qcal']:.6f} USD.
- Fit/calibration/holdout counts: {hold['n_train']:,} / {hold['n_calibration']:,} / {hold['n_test']:,}.
- RMSE {hold['RMSE']:.5f} vs persistence {hold['Persistence_RMSE']:.5f}; interval coverage {hold['PICP_P10_P90']:.2%}; hybrid precision {hold['Precision']:.2%}, recall {hold['Recall']:.2%}, FPR {hold['FPR']:.2%}. Very low holdout recall illustrates threshold drift and limits operational use.
- SHAP uses this exact fitted point model, with 1,200 held-out observations (or all when fewer than 1,200).

![Official SHAP importance](../figures/shap_summary.png)

## Limitations and next work

The original raw snapshot from the declared Kaggle dataset, download date and license are not independently verified. Source series selection is retrospective and features assume timely publication. USD prices are not local household affordability measures. There is no causal or user-impact validation. The model is not a live forecasting service. A prospective later dataset, rolling calibration, validation-based persistence fallback, improved precision/recall and real stakeholder evaluation would be appropriate future work. No OOS results have been changed to imply those improvements already exist.
'''
    (ROOT/'docs'/'RESULTS.md').write_text(text,encoding='utf-8')
if __name__=='__main__':run()
