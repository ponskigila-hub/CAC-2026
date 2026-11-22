#!/usr/bin/env python3
"""
FoodShock — Cross-Country ASEAN Food Price Forecasting & Early Shock Detection System
Machine Learning Solution Challenge (MLSC) - Cyan Academic Competition 2026
Principal Data Scientist Pipeline
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import shap
import optuna
from optuna.samplers import TPESampler

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_RAW = "asean_wfp_2015_2025.csv"
OUT_DIR = Path(".")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

MIN_MONTHS = 60
SELECTED_COUNTRIES = ["IDN", "KHM", "LAO", "PHL", "THA", "TLS", "MMR"]
COMMODITY_GROUPS = ["Rice", "Chicken Meat", "Beef Meat", "Pork Meat", "Cooking Oil"]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def map_commodity_group(commodity: str, unit: str) -> str | None:
    c = str(commodity).lower()
    u = str(unit).upper().strip()
    if "rice" in c and u == "KG":
        return "Rice"
    if "chicken" in c and u == "KG":
        return "Chicken Meat"
    if "beef" in c and u == "KG":
        return "Beef Meat"
    if "pork" in c and u == "KG":
        return "Pork Meat"
    if any(x in c for x in ["oil (vegetable", "oil (palm", "oil (soybean", "oil (cooking", "oil (groundnut"]) and u in ["L", "KG", "5 L"]:
        return "Cooking Oil"
    return None


def standardize_oil_price(row):
    """Convert Cooking Oil to price per Litre. Approximate density 0.92 kg/L for KG."""
    u = str(row["unit"]).upper()
    price = row["usdprice"]
    if u == "L":
        return price
    if u == "5 L":
        return price / 5.0
    if u == "KG":
        # approximate density vegetable oil ~0.92 kg/L
        return price * 0.92
    return price


# ------------------------------------------------------------
# Stage 1: Audit & Standardization
# ------------------------------------------------------------
def stage1_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("=" * 60)
    print("STAGE 1: Audit Cakupan & Standardisasi Data")
    print("=" * 60)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")

    # Filter ASEAN focus countries
    df = df[df["countryiso3"].isin(SELECTED_COUNTRIES)].copy()

    # Map commodity groups
    df["commodity_group"] = df.apply(
        lambda r: map_commodity_group(r["commodity"], r["unit"]), axis=1
    )
    df = df[df["commodity_group"].notna()].copy()

    # Standardize Cooking Oil to per Litre
    oil_mask = df["commodity_group"] == "Cooking Oil"
    df.loc[oil_mask, "usdprice"] = df.loc[oil_mask].apply(standardize_oil_price, axis=1)
    df.loc[oil_mask, "unit"] = "L"

    # Keep only consistent units now
    df = df[df["unit"].isin(["KG", "L"])].copy()

    # Series key for coverage
    df["series_id"] = (
        df["countryiso3"] + "|" +
        df["market"].astype(str) + "|" +
        df["commodity_group"] + "|" +
        df["pricetype"] + "|" +
        df["priceflag"] + "|" +
        df["unit"]
    )

    # Coverage audit
    coverage = (
        df.groupby(
            ["countryiso3", "market", "commodity_group", "pricetype", "priceflag", "unit"]
        )
        .agg(
            n_months=("year_month", "nunique"),
            n_obs=("usdprice", "count"),
            min_date=("date", "min"),
            max_date=("date", "max"),
            mean_usdprice=("usdprice", "mean"),
            std_usdprice=("usdprice", "std"),
        )
        .reset_index()
    )
    coverage["series_id"] = (
        coverage["countryiso3"] + "|" +
        coverage["market"].astype(str) + "|" +
        coverage["commodity_group"] + "|" +
        coverage["pricetype"] + "|" +
        coverage["priceflag"] + "|" +
        coverage["unit"]
    )

    long_series = coverage[coverage["n_months"] >= MIN_MONTHS].copy()
    print(f"\nSeries with >= {MIN_MONTHS} months: {len(long_series)}")
    print(long_series.groupby("countryiso3").size())
    print("\nBy commodity_group:")
    print(long_series.groupby("commodity_group").size())

    # Country coverage table
    country_cov = (
        long_series.groupby("countryiso3")
        .agg(
            n_series=("series_id", "count"),
            n_commodity_groups=("commodity_group", "nunique"),
            avg_months=("n_months", "mean"),
            max_months=("n_months", "max"),
            min_date=("min_date", "min"),
            max_date=("max_date", "max"),
        )
        .reset_index()
        .sort_values("n_series", ascending=False)
    )
    country_cov.to_csv(OUT_DIR / "fpma_wfp_country_coverage.csv", index=False)
    print("\nCountry coverage saved → fpma_wfp_country_coverage.csv")
    print(country_cov)

    # Commodity coverage
    com_cov = (
        long_series.groupby("commodity_group")
        .agg(
            n_series=("series_id", "count"),
            n_countries=("countryiso3", "nunique"),
            countries=("countryiso3", lambda x: ",".join(sorted(x.unique()))),
            avg_months=("n_months", "mean"),
            avg_price_usd=("mean_usdprice", "mean"),
        )
        .reset_index()
        .sort_values("n_countries", ascending=False)
    )
    com_cov.to_csv(OUT_DIR / "fpma_wfp_commodity_coverage.csv", index=False)
    print("\nCommodity coverage saved → fpma_wfp_commodity_coverage.csv")
    print(com_cov)

    # Filter data to long series only
    valid_series = set(long_series["series_id"])
    df_long = df[df["series_id"].isin(valid_series)].copy()
    print(f"\nFiltered rows for long series: {len(df_long)}")

    return df_long, country_cov, com_cov


# ------------------------------------------------------------
# Stage 2: Feature Engineering (Calendar-Safe, Zero Leakage)
# ------------------------------------------------------------
def stage2_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STAGE 2: Feature Engineering & Model-Ready Dataset")
    print("=" * 60)

    # Aggregate to monthly mean per series (handle multiple observations / aggregate flags)
    monthly = (
        df.groupby(["series_id", "year_month", "countryiso3", "market",
                    "commodity_group", "pricetype", "priceflag", "unit"])
        .agg(usdprice=("usdprice", "mean"))
        .reset_index()
    )
    monthly["date"] = monthly["year_month"].dt.to_timestamp()

    # Full calendar re-index per series WITHOUT forward-fill of price
    all_frames = []
    for sid, g in monthly.groupby("series_id"):
        g = g.sort_values("year_month").set_index("year_month")
        # Full monthly range
        full_idx = pd.period_range(g.index.min(), g.index.max(), freq="M")
        g = g.reindex(full_idx)
        g["series_id"] = sid
        # Carry static columns
        for col in ["countryiso3", "market", "commodity_group", "pricetype", "priceflag", "unit"]:
            g[col] = g[col].ffill().bfill()
        g["date"] = g.index.to_timestamp()
        all_frames.append(g.reset_index().rename(columns={"index": "year_month"}))

    panel = pd.concat(all_frames, ignore_index=True)
    panel = panel.sort_values(["series_id", "year_month"]).reset_index(drop=True)
    print(f"Panel after calendar reindex (with NaNs where missing): {panel.shape}")

    # Feature engineering — STRICTLY shift(1) so no leakage into current row
    def add_features(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("year_month").copy()
        price = g["usdprice"]

        # Lags of price (calendar-safe)
        for lag in [1, 2, 3, 6, 12]:
            g[f"lag_{lag}"] = price.shift(lag)

        # Rolling stats computed on shifted series (available at t using data up to t-1)
        shifted = price.shift(1)
        g["rolling_mean_3"] = shifted.rolling(3, min_periods=2).mean()
        g["rolling_mean_6"] = shifted.rolling(6, min_periods=3).mean()
        g["rolling_std_6"] = shifted.rolling(6, min_periods=3).std()
        g["rolling_min_6"] = shifted.rolling(6, min_periods=3).min()
        g["rolling_max_6"] = shifted.rolling(6, min_periods=3).max()
        g["price_to_mean6"] = shifted / g["rolling_mean_6"]
        g["cv_6"] = g["rolling_std_6"] / g["rolling_mean_6"]

        # Momentum / pct change (using past)
        g["pct_change_1m"] = shifted.pct_change(1, fill_method=None)
        g["pct_change_3m"] = shifted.pct_change(3, fill_method=None)
        g["pct_change_12m"] = shifted.pct_change(12, fill_method=None)

        # Seasonal
        month = g["year_month"].dt.month
        g["month_sin"] = np.sin(2 * np.pi * month / 12)
        g["month_cos"] = np.cos(2 * np.pi * month / 12)
        g["month"] = month

        # Targets: next month price and pct change (for shock detection)
        g["target_price_usd_t1"] = price.shift(-1)
        g["target_pct_change_t1"] = (price.shift(-1) - price) / price

        return g

    panel["_sid"] = panel["series_id"].values
    def _apply_feats(g):
        g = add_features(g)
        return g
    panel = panel.groupby("_sid", group_keys=False).apply(_apply_feats)
    panel = panel.reset_index(drop=True)
    if "series_id" not in panel.columns:
        panel["series_id"] = panel["_sid"]
    panel = panel.drop(columns=["_sid"], errors="ignore")

    # Drop rows where features or target are missing (beginning/end of series)
    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_std_6",
        "rolling_min_6", "rolling_max_6", "price_to_mean6", "cv_6",
        "pct_change_1m", "pct_change_3m", "pct_change_12m",
        "month_sin", "month_cos",
    ]
    model_ready = panel.dropna(subset=feature_cols + ["target_price_usd_t1"]).copy()

    # Encode categoricals for modeling convenience
    model_ready["countryiso3"] = model_ready["countryiso3"].astype("category")
    model_ready["commodity_group"] = model_ready["commodity_group"].astype("category")
    model_ready["pricetype"] = model_ready["pricetype"].astype("category")
    model_ready["priceflag"] = model_ready["priceflag"].astype("category")

    print(f"Model-ready shape after dropna: {model_ready.shape}")
    print(f"Columns present: {model_ready.columns.tolist()}")
    print(f"Countries: {model_ready['countryiso3'].unique().tolist()}")
    print(f"Commodity groups: {model_ready['commodity_group'].unique().tolist()}")
    print(f"Date range: {model_ready['date'].min()} → {model_ready['date'].max()}")

    # Save - only existing columns
    out_cols = [
        "series_id", "year_month", "date", "countryiso3", "market",
        "commodity_group", "pricetype", "priceflag", "unit", "usdprice",
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_std_6",
        "rolling_min_6", "rolling_max_6", "price_to_mean6", "cv_6",
        "pct_change_1m", "pct_change_3m", "pct_change_12m",
        "month_sin", "month_cos", "month",
        "target_price_usd_t1", "target_pct_change_t1",
    ]
    existing = [c for c in out_cols if c in model_ready.columns]
    model_ready[existing].to_csv(OUT_DIR / "foodshock_asean_model_ready.csv", index=False)
    print(f"Saved → foodshock_asean_model_ready.csv with columns: {existing}")

    return model_ready


# ------------------------------------------------------------
# Stage 3: Modeling, Tuning, Evaluation, SHAP
# ------------------------------------------------------------
def stage3_modeling(model_ready: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STAGE 3: Residual / Log-Return Forecasting + Walk-Forward Evaluation")
    print("=" * 60)

    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_std_6",
        "rolling_min_6", "rolling_max_6", "price_to_mean6", "cv_6",
        "pct_change_1m", "pct_change_3m", "pct_change_12m",
        "month_sin", "month_cos",
    ]
    cat_features = ["countryiso3", "commodity_group", "pricetype", "priceflag"]

    df = model_ready.copy()
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

    # ----- Residual target: Δy = y_{t+1} - y_t  (using lag_1 as y_t) -----
    # This removes level bias between Rice (~0.77) and Beef (~6.7)
    df["residual_target"] = df["target_price_usd_t1"] - df["lag_1"]
    # Also prepare log-return version for classification / secondary check
    df["log_return_target"] = np.log(df["target_price_usd_t1"] / df["lag_1"].clip(lower=1e-6))

    from sklearn.preprocessing import LabelEncoder
    le_dict = {}
    for col in cat_features:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    X_num = feature_cols + [c + "_enc" for c in cat_features]

    # Time-based split: last 18 months as final hold-out test
    max_date = df["date"].max()
    test_start = max_date - pd.DateOffset(months=17)
    train_mask = df["date"] < test_start
    test_mask = df["date"] >= test_start

    X_train = df.loc[train_mask, X_num]
    y_train_res = df.loc[train_mask, "residual_target"]
    X_test = df.loc[test_mask, X_num]
    y_test_price = df.loc[test_mask, "target_price_usd_t1"]   # absolute for final metrics
    y_test_res = df.loc[test_mask, "residual_target"]
    meta_test = df.loc[test_mask, [
        "countryiso3", "commodity_group", "pricetype", "priceflag",
        "series_id", "date", "lag_1", "target_price_usd_t1", "target_pct_change_t1"
    ]].copy()

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train period: {df.loc[train_mask, 'date'].min().date()} → {df.loc[train_mask, 'date'].max().date()}")
    print(f"Test period : {df.loc[test_mask, 'date'].min().date()} → {df.loc[test_mask, 'date'].max().date()}")
    print(f"Countries in TRAIN only (no test coverage): "
          f"{sorted(set(df.loc[train_mask,'countryiso3']) - set(df.loc[test_mask,'countryiso3']))}")
    print(f"Countries in TEST: {sorted(df.loc[test_mask,'countryiso3'].unique())}")

    # ----- Baselines on absolute price -----
    results = []

    # Persistence: ŷ = lag_1  (equivalent to residual=0)
    y_pred_pers = meta_test["lag_1"].values
    results.append({
        "model": "Persistence (lag_1)",
        "MAE": mean_absolute_error(y_test_price, y_pred_pers),
        "RMSE": np.sqrt(mean_squared_error(y_test_price, y_pred_pers)),
        "sMAPE": smape(y_test_price, y_pred_pers),
        "R2": r2_score(y_test_price, y_pred_pers),
    })

    # Seasonal Naive
    y_pred_seas = df.loc[test_mask, "lag_12"].fillna(df.loc[test_mask, "lag_1"]).values
    results
    results.append({
        "model": "Seasonal Naive (lag_12)",
        "MAE": mean_absolute_error(y_test_price, y_pred_seas),
        "RMSE": np.sqrt(mean_squared_error(y_test_price, y_pred_seas)),
        "sMAPE": smape(y_test_price, y_pred_seas),
        "R2": r2_score(y_test_price, y_pred_seas),
    })

    # ----- ML Models trained on RESIDUAL -----
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=250, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.04,
            random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=400, max_depth=7, learning_rate=0.04,
            random_state=RANDOM_STATE
        ),
    }

    preds_price = {"Persistence (lag_1)": y_pred_pers, "Seasonal Naive (lag_12)": y_pred_seas}
    preds_res = {}

    for name, model in models.items():
        print(f"Training {name} on residual target ...")
        model.fit(X_train, y_train_res)
        res_hat = model.predict(X_test)
        # Reconstruct absolute price: ŷ_{t+1} = lag_1 + residual_hat
        y_hat = meta_test["lag_1"].values + res_hat
        preds_price[name] = y_hat
        preds_res[name] = res_hat
        results.append({
            "model": name,
            "MAE": mean_absolute_error(y_test_price, y_hat),
            "RMSE": np.sqrt(mean_squared_error(y_test_price, y_hat)),
            "sMAPE": smape(y_test_price, y_hat),
            "R2": r2_score(y_test_price, y_hat),
        })
        print(f"  {name}: MAE={results[-1]['MAE']:.4f}, sMAPE={results[-1]['sMAPE']:.2f}%")

    results_df = pd.DataFrame(results).sort_values("MAE")
    print("\n=== Overall Model Comparison (Absolute Price after Residual Reconstruction) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(OUT_DIR / "model_evaluation_results.csv", index=False)

    # Best ML model
    ml_results = results_df[~results_df["model"].str.contains("Persistence|Seasonal")]
    best_name = ml_results.iloc[0]["model"]
    print(f"\nBest ML model: {best_name}")
    best_model = models[best_name]
    y_pred_best = preds_price[best_name]

    # ----- Hyperparameter tuning on LightGBM residual -----
    print("\n--- Optuna Tuning (LightGBM residual) ---")
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
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        maes = []
        for tr_idx, va_idx in tscv.split(X_train):
            model.fit(X_train.iloc[tr_idx], y_train_res.iloc[tr_idx])
            pred_res = model.predict(X_train.iloc[va_idx])
            # evaluate on absolute scale inside CV
            lag = df.loc[train_mask].iloc[va_idx]["lag_1"].values
            y_true_abs = df.loc[train_mask].iloc[va_idx]["target_price_usd_t1"].values
            maes.append(mean_absolute_error(y_true_abs, lag + pred_res))
        return np.mean(maes)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print("Best Optuna params:", study.best_params)
    print("Best CV MAE (abs):", study.best_value)

    best_lgb = lgb.LGBMRegressor(**study.best_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    best_lgb.fit(X_train, y_train_res)
    res_tuned = best_lgb.predict(X_test)
    y_pred_tuned = meta_test["lag_1"].values + res_tuned
    tuned_metrics = {
        "model": "LightGBM_Tuned_Residual",
        "MAE": mean_absolute_error(y_test_price, y_pred_tuned),
        "RMSE": np.sqrt(mean_squared_error(y_test_price, y_pred_tuned)),
        "sMAPE": smape(y_test_price, y_pred_tuned),
        "R2": r2_score(y_test_price, y_pred_tuned),
    }
    print("Tuned LightGBM Residual:", tuned_metrics)
    results_df = pd.concat([results_df, pd.DataFrame([tuned_metrics])], ignore_index=True)
    results_df = results_df.sort_values("MAE")
    results_df.to_csv(OUT_DIR / "model_evaluation_results.csv", index=False)

    # Use best overall (prefer tuned if better)
    if tuned_metrics["MAE"] < results_df[results_df["model"] == best_name]["MAE"].values[0]:
        final_model = best_lgb
        y_pred_final = y_pred_tuned
        best_name = "LightGBM_Tuned_Residual"
    else:
        final_model = models[best_name]
        y_pred_final = y_pred_best

    meta_test["y_true"] = y_test_price.values
    meta_test["y_pred"] = y_pred_final
    meta_test["residual_pred"] = y_pred_final - meta_test["lag_1"].values

    # ----- Stratified evaluation -----
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

    print("\n=== Performance by Commodity Group ===")
    print(by_com.to_string(index=False))
    print("\n=== Performance by Country ===")
    print(by_cty.to_string(index=False))
    print("\n=== Performance by Price Type (Retail vs Wholesale) ===")
    print(by_pt.to_string(index=False))

    by_com.to_csv(OUT_DIR / "model_eval_by_commodity.csv", index=False)
    by_cty.to_csv(OUT_DIR / "model_eval_by_country.csv", index=False)
    by_pt.to_csv(OUT_DIR / "model_eval_by_pricetype.csv", index=False)

    # ----- Food Price Shock Classification -----
    print("\n=== Food Price Shock Early-Warning Classification ===")
    # Define shock: |pct_change_t1| > 5%  OR  residual larger than 1.5 historical std
    hist_std = df.loc[train_mask, "residual_target"].std()
    shock_threshold_pct = 0.05
    meta_test["actual_shock"] = (
        (meta_test["target_pct_change_t1"].abs() > shock_threshold_pct) |
        (meta_test["y_true"] - meta_test["lag_1"]).abs() > (1.5 * hist_std)
    ).astype(int)

    # Predicted shock from model residual
    meta_test["pred_pct"] = (meta_test["y_pred"] - meta_test["lag_1"]) / meta_test["lag_1"].clip(lower=1e-6)
    meta_test["pred_shock"] = (
        (meta_test["pred_pct"].abs() > shock_threshold_pct) |
        (meta_test["residual_pred"].abs() > (1.5 * hist_std))
    ).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    y_true_s = meta_test["actual_shock"]
    y_pred_s = meta_test["pred_shock"]
    # For AUC use continuous residual magnitude
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
        "Shock_Rate": y_true_s.mean(),
    }
    print(shock_metrics)
    print("Confusion Matrix:\n", confusion_matrix(y_true_s, y_pred_s))

    pd.DataFrame([shock_metrics]).to_csv(OUT_DIR / "model_eval_shock_classification.csv", index=False)

    # Also evaluate Persistence as shock detector (always predict no-change → never shock)
    pers_shock = np.zeros(len(y_true_s), dtype=int)
    print("Persistence F1 (always no-shock):", f1_score(y_true_s, pers_shock, zero_division=0))

    # ----- SHAP -----
    print("\nComputing SHAP values (sample)...")
    sample_idx = np.random.choice(len(X_test), size=min(1000, len(X_test)), replace=False)
    X_shap = X_test.iloc[sample_idx]
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_shap)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved figures/shap_summary.png")

    imp = pd.DataFrame({
        "feature": X_num,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop features:")
    print(imp.head(12).to_string(index=False))

    # ----- Visualizations -----
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = results_df.sort_values("MAE")
    sns.barplot(data=plot_df, x="MAE", y="model", ax=ax, palette="viridis")
    ax.set_title("Model Comparison — MAE on Absolute Price (Residual Reconstruction)")
    ax.set_xlabel("Mean Absolute Error (USD)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test_price, y_pred_final, alpha=0.25, s=12)
    lims = [min(y_test_price.min(), y_pred_final.min()),
            max(y_test_price.max(), y_pred_final.max())]
    ax.plot(lims, lims, "r--", lw=2)
    ax.set_xlabel("Actual Price (USD)")
    ax.set_ylabel("Predicted Price (USD)")
    ax.set_title(f"Actual vs Predicted — {best_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=200)
    plt.close()

    # EDA trends (reuse if exists, else recreate)
    trend = (
        model_ready.groupby(["date", "commodity_group"])["usdprice"]
        .mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    for grp in COMMODITY_GROUPS:
        sub = trend[trend["commodity_group"] == grp]
        if len(sub) > 0:
            ax.plot(sub["date"], sub["usdprice"], label=grp, lw=1.5)
    ax.set_title("Average Food Prices (USD) by Commodity Group — ASEAN Long Series")
    ax.set_ylabel("USD Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_trends.png", dpi=200)
    plt.close()

    meta_test["abs_error"] = np.abs(meta_test["y_true"] - meta_test["y_pred"])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=meta_test, x="commodity_group", y="abs_error", ax=ax)
    ax.set_title("Absolute Error Distribution by Commodity Group")
    ax.set_ylabel("|Error| (USD)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "error_by_commodity.png", dpi=200)
    plt.close()

    # Shock related plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=meta_test, x="residual_pred", hue="actual_shock", bins=50, ax=ax, kde=True)
    ax.set_title("Predicted Residual Distribution by Actual Shock Label")
    ax.axvline(1.5*hist_std, color="red", ls="--", label="±1.5σ")
    ax.axvline(-1.5*hist_std, color="red", ls="--")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shock_residual_dist.png", dpi=200)
    plt.close()

    print("All figures saved to figures/")
    print("\n=== FINAL RESULTS (sorted by MAE) ===")
    print(results_df.to_string(index=False))

    return final_model, results_df, meta_test, imp



def main():
    print("Loading raw data...")
    df = pd.read_csv(DATA_RAW, low_memory=False)
    print(f"Raw shape: {df.shape}")

    df_long, country_cov, com_cov = stage1_audit(df)
    model_ready = stage2_feature_engineering(df_long)
    final_model, results_df, meta_test, imp = stage3_modeling(model_ready)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(results_df.to_string(index=False))
    return final_model, results_df


if __name__ == "__main__":
    main()
