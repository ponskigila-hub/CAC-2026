#!/usr/bin/env python3
"""Residual forecasting modeling only (uses existing model-ready CSV)"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix)
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import shap
import optuna
from optuna.samplers import TPESampler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
OUT_DIR = Path(".")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
COMMODITY_GROUPS = ["Rice", "Chicken Meat", "Beef Meat", "Pork Meat", "Cooking Oil"]

def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

print("Loading model-ready data...")
df = pd.read_csv("foodshock_asean_model_ready.csv")
df["date"] = pd.to_datetime(df["date"])
print("Shape:", df.shape)

if "series_id" not in df.columns:
    df["series_id"] = (
        df["countryiso3"].astype(str) + "|" +
        df["market"].astype(str) + "|" +
        df["commodity_group"].astype(str) + "|" +
        df["pricetype"].astype(str) + "|" +
        df["priceflag"].astype(str) + "|" +
        df["unit"].astype(str)
    )
df = df.sort_values(["date", "series_id"]).reset_index(drop=True)

# Residual target
df["residual_target"] = df["target_price_usd_t1"] - df["lag_1"]

feature_cols = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "rolling_std_6",
    "rolling_min_6", "rolling_max_6", "price_to_mean6", "cv_6",
    "pct_change_1m", "pct_change_3m", "pct_change_12m",
    "month_sin", "month_cos",
]
cat_features = ["countryiso3", "commodity_group", "pricetype", "priceflag"]

for col in cat_features:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

X_num = feature_cols + [c + "_enc" for c in cat_features]

max_date = df["date"].max()
test_start = max_date - pd.DateOffset(months=17)
train_mask = df["date"] < test_start
test_mask = df["date"] >= test_start

X_train = df.loc[train_mask, X_num]
y_train_res = df.loc[train_mask, "residual_target"]
X_test = df.loc[test_mask, X_num]
y_test_price = df.loc[test_mask, "target_price_usd_t1"]
meta_test = df.loc[test_mask, [
    "countryiso3", "commodity_group", "pricetype", "priceflag",
    "series_id", "date", "lag_1", "target_price_usd_t1", "target_pct_change_t1"
]].copy()

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Train period: {df.loc[train_mask,'date'].min().date()} → {df.loc[train_mask,'date'].max().date()}")
print(f"Test period : {df.loc[test_mask,'date'].min().date()} → {df.loc[test_mask,'date'].max().date()}")
print(f"Train-only countries: {sorted(set(df.loc[train_mask,'countryiso3']) - set(df.loc[test_mask,'countryiso3']))}")
print(f"Test countries: {sorted(df.loc[test_mask,'countryiso3'].unique())}")

results = []

# Baselines
y_pred_pers = meta_test["lag_1"].values
results.append({
    "model": "Persistence (lag_1)",
    "MAE": mean_absolute_error(y_test_price, y_pred_pers),
    "RMSE": np.sqrt(mean_squared_error(y_test_price, y_pred_pers)),
    "sMAPE": smape(y_test_price, y_pred_pers),
    "R2": r2_score(y_test_price, y_pred_pers),
})

y_pred_seas = df.loc[test_mask, "lag_12"].fillna(df.loc[test_mask, "lag_1"]).values
results.append({
    "model": "Seasonal Naive (lag_12)",
    "MAE": mean_absolute_error(y_test_price, y_pred_seas),
    "RMSE": np.sqrt(mean_squared_error(y_test_price, y_pred_seas)),
    "sMAPE": smape(y_test_price, y_pred_seas),
    "R2": r2_score(y_test_price, y_pred_seas),
})

models = {
    "RandomForest": RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.85, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=500, max_depth=7, learning_rate=0.04, subsample=0.85, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    "CatBoost": CatBoostRegressor(iterations=500, depth=6, learning_rate=0.04, random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=400, max_depth=7, learning_rate=0.04, random_state=RANDOM_STATE),
}

preds_price = {"Persistence (lag_1)": y_pred_pers, "Seasonal Naive (lag_12)": y_pred_seas}

for name, model in models.items():
    print(f"Training {name} on residual ...")
    model.fit(X_train, y_train_res)
    res_hat = model.predict(X_test)
    y_hat = meta_test["lag_1"].values + res_hat
    preds_price[name] = y_hat
    results.append({
        "model": name,
        "MAE": mean_absolute_error(y_test_price, y_hat),
        "RMSE": np.sqrt(mean_squared_error(y_test_price, y_hat)),
        "sMAPE": smape(y_test_price, y_hat),
        "R2": r2_score(y_test_price, y_hat),
    })
    print(f"  → MAE={results[-1]['MAE']:.4f}  sMAPE={results[-1]['sMAPE']:.2f}%")

results_df = pd.DataFrame(results).sort_values("MAE")
print("\n=== Comparison after Residual Reconstruction ===")
print(results_df.to_string(index=False))

# Optuna on LightGBM residual
print("\n--- Optuna (LightGBM residual) ---")
tscv = TimeSeriesSplit(n_splits=4)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 700),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    maes = []
    train_idx_all = np.where(train_mask)[0]
    for tr_rel, va_rel in tscv.split(X_train):
        model.fit(X_train.iloc[tr_rel], y_train_res.iloc[tr_rel])
        pred_res = model.predict(X_train.iloc[va_rel])
        lag = df.iloc[train_idx_all[va_rel]]["lag_1"].values
        y_true_abs = df.iloc[train_idx_all[va_rel]]["target_price_usd_t1"].values
        maes.append(mean_absolute_error(y_true_abs, lag + pred_res))
    return np.mean(maes)

study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=25, show_progress_bar=False)
print("Best params:", study.best_params)
print("Best CV MAE:", study.best_value)

best_lgb = lgb.LGBMRegressor(**study.best_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
best_lgb.fit(X_train, y_train_res)
res_tuned = best_lgb.predict(X_test)
y_pred_tuned = meta_test["lag_1"].values + res_tuned
tuned = {
    "model": "LightGBM_Tuned_Residual",
    "MAE": mean_absolute_error(y_test_price, y_pred_tuned),
    "RMSE": np.sqrt(mean_squared_error(y_test_price, y_pred_tuned)),
    "sMAPE": smape(y_test_price, y_pred_tuned),
    "R2": r2_score(y_test_price, y_pred_tuned),
}
print("Tuned:", tuned)
results_df = pd.concat([results_df, pd.DataFrame([tuned])], ignore_index=True).sort_values("MAE")
results_df.to_csv("model_evaluation_results.csv", index=False)

# Select best ML
ml_df = results_df[~results_df["model"].str.contains("Persistence|Seasonal")]
best_name = ml_df.iloc[0]["model"]
if best_name == "LightGBM_Tuned_Residual":
    final_model = best_lgb
    y_pred_final = y_pred_tuned
else:
    final_model = models[best_name]
    y_pred_final = preds_price[best_name]

meta_test["y_true"] = y_test_price.values
meta_test["y_pred"] = y_pred_final
meta_test["residual_pred"] = y_pred_final - meta_test["lag_1"].values

def eval_group(g):
    return pd.Series({
        "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
        "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
        "sMAPE": smape(g["y_true"], g["y_pred"]),
        "R2": r2_score(g["y_true"], g["y_pred"]) if len(g) > 2 else np.nan,
        "n": len(g),
    })

by_com = meta_test.groupby("commodity_group").apply(eval_group).reset_index()
by_cty = meta_test.groupby("countryiso3").apply(eval_group).reset_index()
by_pt = meta_test.groupby("pricetype").apply(eval_group).reset_index()
print("\nBy Commodity:\n", by_com.to_string(index=False))
print("\nBy Country:\n", by_cty.to_string(index=False))
print("\nBy PriceType:\n", by_pt.to_string(index=False))
by_com.to_csv("model_eval_by_commodity.csv", index=False)
by_cty.to_csv("model_eval_by_country.csv", index=False)
by_pt.to_csv("model_eval_by_pricetype.csv", index=False)

# Shock classification
hist_std = df.loc[train_mask, "residual_target"].std()
shock_threshold_pct = 0.05
meta_test["actual_shock"] = (
    (meta_test["target_pct_change_t1"].abs() > shock_threshold_pct) |
    ((meta_test["y_true"] - meta_test["lag_1"]).abs() > (1.5 * hist_std))
).astype(int)
meta_test["pred_pct"] = (meta_test["y_pred"] - meta_test["lag_1"]) / meta_test["lag_1"].clip(lower=1e-6)
meta_test["pred_shock"] = (
    (meta_test["pred_pct"].abs() > shock_threshold_pct) |
    (meta_test["residual_pred"].abs() > (1.5 * hist_std))
).astype(int)

y_true_s = meta_test["actual_shock"]
y_pred_s = meta_test["pred_shock"]
try:
    auc = roc_auc_score(y_true_s, meta_test["residual_pred"].abs())
except Exception:
    auc = np.nan
shock_metrics = {
    "Precision": precision_score(y_true_s, y_pred_s, zero_division=0),
    "Recall": recall_score(y_true_s, y_pred_s, zero_division=0),
    "F1": f1_score(y_true_s, y_pred_s, zero_division=0),
    "ROC_AUC": auc,
    "Support_Shock": int(y_true_s.sum()),
    "Support_Total": len(y_true_s),
    "Shock_Rate": float(y_true_s.mean()),
}
print("\nShock Classification:", shock_metrics)
print("CM:\n", confusion_matrix(y_true_s, y_pred_s))
pd.DataFrame([shock_metrics]).to_csv("model_eval_shock_classification.csv", index=False)

# SHAP
print("SHAP...")
sample_idx = np.random.choice(len(X_test), size=min(1000, len(X_test)), replace=False)
X_shap = X_test.iloc[sample_idx]
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_shap)
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()

# Figures
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=results_df.sort_values("MAE"), x="MAE", y="model", ax=ax, palette="viridis")
ax.set_title("Model Comparison — MAE (Residual → Absolute Reconstruction)")
ax.set_xlabel("MAE (USD)")
plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison.png", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test_price, y_pred_final, alpha=0.25, s=12)
lims = [min(y_test_price.min(), y_pred_final.min()), max(y_test_price.max(), y_pred_final.max())]
ax.plot(lims, lims, "r--", lw=2)
ax.set_xlabel("Actual Price (USD)")
ax.set_ylabel("Predicted Price (USD)")
ax.set_title(f"Actual vs Predicted — {best_name}")
plt.tight_layout()
plt.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=200)
plt.close()

trend = model_ready if False else df  # use df
trend = df.groupby(["date", "commodity_group"])["usdprice"].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
for grp in COMMODITY_GROUPS:
    sub = trend[trend["commodity_group"] == grp]
    if len(sub):
        ax.plot(sub["date"], sub["usdprice"], label=grp, lw=1.5)
ax.set_title("Average Food Prices (USD) by Commodity Group")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_trends.png", dpi=200)
plt.close()

meta_test["abs_error"] = np.abs(meta_test["y_true"] - meta_test["y_pred"])
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=meta_test, x="commodity_group", y="abs_error", ax=ax)
ax.set_title("Absolute Error by Commodity Group")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(FIG_DIR / "error_by_commodity.png", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=meta_test, x="residual_pred", hue="actual_shock", bins=50, ax=ax, kde=True)
ax.axvline(1.5*hist_std, color="red", ls="--")
ax.axvline(-1.5*hist_std, color="red", ls="--")
ax.set_title("Predicted Residual by Actual Shock")
plt.tight_layout()
plt.savefig(FIG_DIR / "shock_residual_dist.png", dpi=200)
plt.close()

print("\n=== FINAL SORTED RESULTS ===")
print(results_df.to_string(index=False))
print("\nDone.")
