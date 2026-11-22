"""Model builders — residual point, quantile P10/P90, Stage-2 shock classifier."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from utils import FEATURE_COLS, CAT_COLS, smape, reconstruct_price, ensure_feature_row

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
    from catboost import CatBoostRegressor, CatBoostClassifier
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


def _build_regressor(name: str, n_estimators: int, learning_rate: float, max_depth: int, random_state: int = 42):
    if name == "LightGBM" and HAS_LGBM:
        return lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
    if name == "XGBoost" and HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            random_state=random_state, n_jobs=-1, verbosity=0,
        )
    if name == "CatBoost" and HAS_CAT:
        return CatBoostRegressor(
            iterations=n_estimators, learning_rate=learning_rate, depth=max_depth,
            random_seed=random_state, verbose=0, allow_writing_files=False,
        )
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            random_state=random_state,
        )
    return RandomForestRegressor(
        n_estimators=min(n_estimators, 300), max_depth=max_depth,
        min_samples_leaf=5, n_jobs=-1, random_state=random_state,
    )


def prepare_xy(df: pd.DataFrame, target_mode: str = "residual"):
    data = df.copy()
    for col in FEATURE_COLS + ["target_price_usd_t1"]:
        if col not in data.columns:
            if col in FEATURE_COLS:
                data[col] = 0.0 if ("fx" in col or "mom" in col or col.startswith("pct_")) else data.get("usdprice", 1.0)
            else:
                raise ValueError(f"Missing required column: {col}")

    encoders = {}
    for col in CAT_COLS:
        if col in data.columns:
            le = LabelEncoder()
            data[col + "_enc"] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
        else:
            data[col + "_enc"] = 0
            encoders[col] = None

    feature_names = FEATURE_COLS + [c + "_enc" for c in CAT_COLS]
    X = data[feature_names].copy()
    y_abs = data["target_price_usd_t1"]
    # Residual from current price y_t = usdprice
    if target_mode == "residual":
        y = data["target_price_usd_t1"] - data["usdprice"]
    else:
        y = data["target_price_usd_t1"]
    return X, y, y_abs, encoders, feature_names


def time_based_split(df: pd.DataFrame, test_months: int = 18):
    max_date = df["date"].max()
    test_start = max_date - pd.DateOffset(months=test_months - 1)
    train_mask = df["date"] < test_start
    test_mask = df["date"] >= test_start
    return train_mask, test_mask


def train_and_evaluate(
    df: pd.DataFrame,
    algorithm: str = "LightGBM",
    target_mode: str = "residual",
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    test_months: int = 18,
) -> dict[str, Any]:
    data = df.copy()
    # fill macro gaps
    for c in FEATURE_COLS:
        if c not in data.columns:
            data[c] = 0.0
        data[c] = data[c].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["usdprice", "target_price_usd_t1"]).copy()
    if "date" in data.columns:
        data = data.sort_values("date").reset_index(drop=True)

    train_mask, test_mask = time_based_split(data, test_months=test_months)
    if train_mask.sum() < 50 or test_mask.sum() < 10:
        raise ValueError("Insufficient data after time-based split. Relax filters.")

    X, y, y_abs, encoders, feature_names = prepare_xy(data, target_mode=target_mode)
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test_abs = y_abs.loc[test_mask].values
    y_t_test = data.loc[test_mask, "usdprice"].values

    # Point residual / direct model
    model = _build_regressor(algorithm, n_estimators, learning_rate, max_depth)
    model.fit(X_train, y_train)
    raw_pred = model.predict(X_test)
    if target_mode == "residual":
        y_pred_abs = reconstruct_price(y_t_test, raw_pred)
    else:
        y_pred_abs = raw_pred

    # Quantile P10 / P90 (LightGBM if available, else HistGB approx via residual std)
    q10_model = q90_model = None
    if HAS_LGBM and target_mode == "residual":
        q10_model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.1,
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        q90_model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.9,
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        q10_model.fit(X_train, y_train)
        q90_model.fit(X_train, y_train)
        y_p10 = reconstruct_price(y_t_test, q10_model.predict(X_test))
        y_p90 = reconstruct_price(y_t_test, q90_model.predict(X_test))
        y_lo = np.minimum(y_p10, y_p90)
        y_hi = np.maximum(y_p10, y_p90)
        picp = float(np.mean((y_test_abs >= y_lo) & (y_test_abs <= y_hi)))
    else:
        resid_std = float(np.std(y_test_abs - y_pred_abs)) if len(y_test_abs) else 0.1
        y_lo = y_pred_abs - 1.28 * resid_std
        y_hi = y_pred_abs + 1.28 * resid_std
        picp = float(np.mean((y_test_abs >= y_lo) & (y_test_abs <= y_hi)))

    # Stage-2 shock classifier
    shock_clf = None
    if "target_pct_change_t1" in data.columns:
        y_shock_tr = (data.loc[train_mask, "target_pct_change_t1"].abs() > 0.05).astype(int)
    else:
        y_shock_tr = ((data.loc[train_mask, "target_price_usd_t1"] - data.loc[train_mask, "usdprice"]).abs()
                      / data.loc[train_mask, "usdprice"].clip(lower=1e-6) > 0.05).astype(int)

    if HAS_LGBM:
        shock_clf = lgb.LGBMClassifier(
            n_estimators=min(n_estimators, 250), max_depth=max_depth, learning_rate=learning_rate,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    else:
        shock_clf = HistGradientBoostingClassifier(
            max_iter=min(n_estimators, 250), max_depth=max_depth, learning_rate=learning_rate,
            class_weight="balanced", random_state=42,
        )
    shock_clf.fit(X_train, y_shock_tr)
    shock_proba = shock_clf.predict_proba(X_test)[:, 1]

    y_pers = y_t_test
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
        "PICP_P10_P90": picp,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }

    return {
        "model": model,
        "q10_model": q10_model,
        "q90_model": q90_model,
        "shock_clf": shock_clf,
        "encoders": encoders,
        "feature_names": feature_names,
        "target_mode": target_mode,
        "metrics": metrics,
        "y_true": y_test_abs,
        "y_pred": np.asarray(y_pred_abs),
        "y_pers": y_pers,
        "y_lo": np.asarray(y_lo),
        "y_hi": np.asarray(y_hi),
        "shock_proba": np.asarray(shock_proba),
        "dates": data.loc[test_mask, "date"].values if "date" in data.columns else None,
    }


def _row_to_frame(row: dict, feature_names: list[str], encoders: dict) -> pd.DataFrame:
    row = ensure_feature_row(row)
    feat = {c: float(row.get(c, 0.0)) for c in FEATURE_COLS}
    for col in CAT_COLS:
        enc_name = col + "_enc"
        le = encoders.get(col)
        val = str(row.get(col, "Unknown"))
        if le is not None and hasattr(le, "classes_") and val in le.classes_:
            feat[enc_name] = int(le.transform([val])[0])
        else:
            feat[enc_name] = 0
    return pd.DataFrame([{k: feat.get(k, 0) for k in feature_names}])


def predict_single(model, encoders, feature_names, target_mode, row: dict) -> float:
    X = _row_to_frame(row, feature_names, encoders)
    raw = model.predict(X)[0]
    y_t = float(ensure_feature_row(row).get("usdprice", row.get("lag_1", 0)))
    if target_mode == "residual":
        return float(reconstruct_price(y_t, raw))
    return float(raw)


def predict_two_stage(bundle: dict, row: dict) -> dict:
    """
    Returns point prediction, P10/P90 bounds, shock probability.
    bundle keys from train_and_evaluate / session_state.
    """
    row = ensure_feature_row(row)
    model = bundle["model"]
    encoders = bundle["encoders"]
    feature_names = bundle["feature_names"]
    target_mode = bundle.get("target_mode", "residual")
    X = _row_to_frame(row, feature_names, encoders)
    y_t = float(row["usdprice"])

    raw = model.predict(X)[0]
    if target_mode == "residual":
        y_hat = float(reconstruct_price(y_t, raw))
    else:
        y_hat = float(raw)

    q10_model = bundle.get("q10_model")
    q90_model = bundle.get("q90_model")
    if q10_model is not None and q90_model is not None and target_mode == "residual":
        y_p10 = float(reconstruct_price(y_t, q10_model.predict(X)[0]))
        y_p90 = float(reconstruct_price(y_t, q90_model.predict(X)[0]))
        y_lo, y_hi = min(y_p10, y_p90), max(y_p10, y_p90)
    else:
        band = abs(y_hat) * 0.03 + 0.01
        y_lo, y_hi = y_hat - band, y_hat + band

    shock_clf = bundle.get("shock_clf")
    if shock_clf is not None:
        proba = float(shock_clf.predict_proba(X)[0, 1])
    else:
        proba = 0.0

    return {
        "y_hat": y_hat,
        "y_p10": y_lo,
        "y_p90": y_hi,
        "shock_proba": proba,
        "y_t": y_t,
    }
