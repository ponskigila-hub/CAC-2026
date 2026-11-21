"""Model builders and trainers for residual vs direct forecasting."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from utils import FEATURE_COLS, CAT_COLS, smape, reconstruct_price

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


def available_algorithms() -> list[str]:
    algs = ["Random Forest", "HistGradientBoosting"]
    if HAS_LGBM:
        algs.insert(0, "LightGBM")
    if HAS_XGB:
        algs.append("XGBoost")
    if HAS_CAT:
        algs.append("CatBoost")
    return algs


def _build_model(name: str, n_estimators: int, learning_rate: float, max_depth: int, random_state: int = 42):
    if name == "LightGBM" and HAS_LGBM:
        return lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=random_state, n_jobs=-1, verbose=-1,
        )
    if name == "XGBoost" and HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=random_state, n_jobs=-1, verbosity=0,
        )
    if name == "CatBoost" and HAS_CAT:
        return CatBoostRegressor(
            iterations=n_estimators, learning_rate=learning_rate,
            depth=max_depth, random_seed=random_state, verbose=0, allow_writing_files=False,
        )
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=random_state,
        )
    # Default / Random Forest
    return RandomForestRegressor(
        n_estimators=min(n_estimators, 300), max_depth=max_depth,
        min_samples_leaf=5, n_jobs=-1, random_state=random_state,
    )


def prepare_xy(
    df: pd.DataFrame,
    target_mode: str = "residual",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Prepare feature matrix and targets.
    target_mode: 'residual' -> y = target_price - lag_1
                 'direct'   -> y = target_price
    Returns X, y_train_target, y_abs_price, encoders
    """
    data = df.copy()
    for col in FEATURE_COLS + ["target_price_usd_t1", "lag_1"]:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    encoders = {}
    for col in CAT_COLS:
        if col in data.columns:
            le = LabelEncoder()
            data[col + "_enc"] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

    feature_names = FEATURE_COLS + [c + "_enc" for c in CAT_COLS if c in data.columns]
    X = data[feature_names].copy()

    y_abs = data["target_price_usd_t1"]
    if target_mode == "residual":
        y = data["target_price_usd_t1"] - data["lag_1"]
    else:
        y = data["target_price_usd_t1"]

    return X, y, y_abs, encoders


def time_based_split(df: pd.DataFrame, test_months: int = 18):
    """Return boolean masks for train / test using last N months as hold-out."""
    if "date" not in df.columns:
        raise ValueError("date column required for time-based split")
    max_date = df["date"].max()
    test_start = max_date - pd.DateOffset(months=test_months - 1)
    train_mask = df["date"] < test_start
    test_mask = df["date"] >= test_start
    return train_mask, test_mask


def train_and_evaluate(
    df: pd.DataFrame,
    algorithm: str = "LightGBM",
    target_mode: str = "residual",
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    test_months: int = 18,
) -> dict[str, Any]:
    """
    Train model on time-based split and return metrics + predictions.
    Always evaluates on absolute price scale for fair comparison with Persistence.
    """
    data = df.dropna(subset=FEATURE_COLS + ["target_price_usd_t1", "lag_1"]).copy()
    if "date" in data.columns:
        data = data.sort_values("date").reset_index(drop=True)

    train_mask, test_mask = time_based_split(data, test_months=test_months)
    if train_mask.sum() < 50 or test_mask.sum() < 10:
        raise ValueError("Insufficient data after time-based split. Relax filters.")

    X, y, y_abs, encoders = prepare_xy(data, target_mode=target_mode)

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test_abs = y_abs.loc[test_mask].values
    lag_test = data.loc[test_mask, "lag_1"].values

    model = _build_model(algorithm, n_estimators, learning_rate, max_depth)
    model.fit(X_train, y_train)
    raw_pred = model.predict(X_test)

    if target_mode == "residual":
        y_pred_abs = reconstruct_price(lag_test, raw_pred)
    else:
        y_pred_abs = raw_pred

    # Persistence baseline
    y_pers = lag_test

    metrics = {
        "model": f"{algorithm} ({'Residual' if target_mode == 'residual' else 'Direct'})",
        "MAE": mean_absolute_error(y_test_abs, y_pred_abs),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_abs, y_pred_abs))),
        "sMAPE": smape(y_test_abs, y_pred_abs),
        "R2": r2_score(y_test_abs, y_pred_abs),
        "Persistence_MAE": mean_absolute_error(y_test_abs, y_pers),
        "Persistence_RMSE": float(np.sqrt(mean_squared_error(y_test_abs, y_pers))),
        "Persistence_sMAPE": smape(y_test_abs, y_pers),
        "Persistence_R2": r2_score(y_test_abs, y_pers),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }

    return {
        "model": model,
        "encoders": encoders,
        "feature_names": list(X.columns),
        "target_mode": target_mode,
        "metrics": metrics,
        "y_true": y_test_abs,
        "y_pred": np.asarray(y_pred_abs),
        "y_pers": y_pers,
        "dates": data.loc[test_mask, "date"].values if "date" in data.columns else None,
    }


def predict_single(
    model,
    encoders: dict,
    feature_names: list[str],
    target_mode: str,
    row: dict,
) -> float:
    """Predict absolute price for a single feature row (dict)."""
    import pandas as pd

    feat = {}
    for c in FEATURE_COLS:
        feat[c] = float(row.get(c, 0.0))

    for col in CAT_COLS:
        enc_name = col + "_enc"
        if enc_name in feature_names and col in encoders:
            le = encoders[col]
            val = str(row.get(col, "Unknown"))
            if val in le.classes_:
                feat[enc_name] = int(le.transform([val])[0])
            else:
                feat[enc_name] = 0
        elif enc_name in feature_names:
            feat[enc_name] = 0

    X = pd.DataFrame([{k: feat.get(k, 0) for k in feature_names}])
    raw = model.predict(X)[0]

    if target_mode == "residual":
        return float(reconstruct_price(row.get("lag_1", row.get("usdprice", 0)), raw))
    return float(raw)
