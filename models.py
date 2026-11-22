"""Model builders V5 — CatBoost-first, fold-safe encoding, conformal quantiles, regularized Stage-2."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingClassifier
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
    """Prefer CatBoost (best V4/V5 WF RMSE), then LGBM, XGB, HistGB, RF."""
    algs = []
    if HAS_CAT:
        algs.append("CatBoost")
    if HAS_LGBM:
        algs.append("LightGBM")
    if HAS_XGB:
        algs.append("XGBoost")
    algs.extend(["HistGradientBoosting", "Random Forest"])
    return algs


def _build_regressor(name: str, n_estimators: int, learning_rate: float, max_depth: int, random_state: int = 42):
    if name == "CatBoost" and HAS_CAT:
        return CatBoostRegressor(iterations=n_estimators, learning_rate=learning_rate, depth=max_depth,
                                 random_seed=random_state, verbose=0, allow_writing_files=False)
    if name == "LightGBM" and HAS_LGBM:
        return lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                 random_state=random_state, n_jobs=-1, verbose=-1)
    if name == "XGBoost" and HAS_XGB:
        return xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                random_state=random_state, n_jobs=-1, verbosity=0)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(max_iter=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                             random_state=random_state)
    return RandomForestRegressor(n_estimators=min(n_estimators, 200), max_depth=max_depth,
                                 min_samples_leaf=5, n_jobs=-1, random_state=random_state)


def _fit_encoders(train_df: pd.DataFrame) -> dict:
    enc = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        if col in train_df.columns:
            le.fit(train_df[col].astype(str))
        else:
            le.fit(["Unknown"])
        enc[col] = le
    return enc


def _transform(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        out[c] = df[c].values if c in df.columns else 0.0
    for col in CAT_COLS:
        le = encoders[col]
        known = set(le.classes_)
        vals = df[col].astype(str) if col in df.columns else pd.Series(["Unknown"] * len(df), index=df.index)
        mapped = [int(le.transform([v])[0]) if v in known else -1 for v in vals]
        out[col + "_enc"] = mapped
    return out


def time_based_split(df: pd.DataFrame, test_months: int = 18):
    max_date = df["date"].max()
    test_start = max_date - pd.DateOffset(months=test_months - 1)
    train_mask = df["date"] < test_start
    test_mask = df["date"] >= test_start
    return train_mask, test_mask


def _month_block_cal_split(train_df: pd.DataFrame, val_ratio: float = 0.15):
    months = sorted(train_df["date"].dt.to_period("M").astype(str).unique())
    if len(months) < 5:
        n = len(train_df); cut = int(n * (1 - val_ratio))
        return train_df.iloc[:cut], train_df.iloc[cut:]
    n_val = max(1, int(len(months) * val_ratio))
    val_m = set(months[-n_val:])
    ym = train_df["date"].dt.to_period("M").astype(str)
    tr = train_df[~ym.isin(val_m)]; va = train_df[ym.isin(val_m)]
    if len(tr) < 50 or len(va) < 15:
        n = len(train_df); cut = int(n * (1 - val_ratio))
        return train_df.iloc[:cut], train_df.iloc[cut:]
    return tr, va


def train_and_evaluate(
    df: pd.DataFrame,
    algorithm: str = "CatBoost",
    target_mode: str = "residual",
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    test_months: int = 18,
) -> dict[str, Any]:
    data = df.copy()
    for c in FEATURE_COLS:
        if c not in data.columns:
            data[c] = 0.0
        data[c] = data[c].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["usdprice", "target_price_usd_t1"]).copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date").reset_index(drop=True)

    train_mask, test_mask = time_based_split(data, test_months=test_months)
    if train_mask.sum() < 50 or test_mask.sum() < 10:
        raise ValueError("Insufficient data after time-based split.")

    train_df = data.loc[train_mask]
    test_df = data.loc[test_mask]
    encoders = _fit_encoders(train_df)  # fold-safe: train only
    X_train = _transform(train_df, encoders)
    X_test = _transform(test_df, encoders)
    feature_names = list(X_train.columns)

    y_train_abs = train_df["target_price_usd_t1"].values
    y_test_abs = test_df["target_price_usd_t1"].values
    y_t_train = train_df["usdprice"].values
    y_t_test = test_df["usdprice"].values
    if target_mode == "residual":
        y_train = y_train_abs - y_t_train
    else:
        y_train = y_train_abs

    model = _build_regressor(algorithm, n_estimators, learning_rate, max_depth)
    model.fit(X_train, y_train)
    raw_pred = model.predict(X_test)
    y_pred_abs = reconstruct_price(y_t_test, raw_pred) if target_mode == "residual" else raw_pred

    # Conformalized quantiles (split-conformal on last 15% month blocks of train)
    q10_model = q90_model = None
    qcal = 0.0
    if HAS_LGBM and target_mode == "residual":
        tr_in, va_in = _month_block_cal_split(train_df, val_ratio=0.15)
        enc_q = _fit_encoders(tr_in)
        X_tri = _transform(tr_in, enc_q); y_tri = tr_in["target_price_usd_t1"].values - tr_in["usdprice"].values
        X_vai = _transform(va_in, enc_q); y_vai = va_in["target_price_usd_t1"].values - va_in["usdprice"].values
        q10_model = lgb.LGBMRegressor(objective="quantile", alpha=0.1, n_estimators=min(n_estimators, 120),
                                      learning_rate=learning_rate, max_depth=min(max_depth, 5),
                                      random_state=42, n_jobs=-1, verbose=-1)
        q90_model = lgb.LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=min(n_estimators, 120),
                                      learning_rate=learning_rate, max_depth=min(max_depth, 5),
                                      random_state=42, n_jobs=-1, verbose=-1)
        q10_model.fit(X_tri, y_tri); q90_model.fit(X_tri, y_tri)
        lo_va = np.minimum(q10_model.predict(X_vai), q90_model.predict(X_vai))
        hi_va = np.maximum(q10_model.predict(X_vai), q90_model.predict(X_vai))
        s = np.maximum(lo_va - y_vai, y_vai - hi_va)
        qcal = float(np.quantile(s, 0.8)) if len(s) > 10 else 0.0
        # refit on full train with train encoders already used for X_train
        q10_model.fit(X_train, y_train); q90_model.fit(X_train, y_train)
        lo_te = np.minimum(q10_model.predict(X_test), q90_model.predict(X_test)) - qcal
        hi_te = np.maximum(q10_model.predict(X_test), q90_model.predict(X_test)) + qcal
        y_lo = reconstruct_price(y_t_test, lo_te)
        y_hi = reconstruct_price(y_t_test, hi_te)
        picp = float(np.mean((y_test_abs >= y_lo) & (y_test_abs <= y_hi)))
    else:
        resid_std = float(np.std(y_test_abs - y_pred_abs)) if len(y_test_abs) else 0.1
        y_lo = y_pred_abs - 1.28 * resid_std
        y_hi = y_pred_abs + 1.28 * resid_std
        picp = float(np.mean((y_test_abs >= y_lo) & (y_test_abs <= y_hi)))

    # Stage-2 regularized classifier (aligned with pipeline V4/V5)
    if "target_pct_change_t1" in train_df.columns:
        y_shock_tr = (train_df["target_pct_change_t1"] > 0.05).astype(int)
    else:
        y_shock_tr = ((train_df["target_price_usd_t1"] - train_df["usdprice"]) /
                      train_df["usdprice"].clip(lower=1e-6) > 0.05).astype(int)

    if HAS_LGBM:
        shock_clf = lgb.LGBMClassifier(
            n_estimators=150, max_depth=4, min_child_samples=50, reg_lambda=5.0,
            subsample=0.8, colsample_bytree=0.8, learning_rate=0.03,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    else:
        shock_clf = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.03, class_weight="balanced", random_state=42,
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
        "model": model, "q10_model": q10_model, "q90_model": q90_model, "qcal": qcal,
        "shock_clf": shock_clf, "encoders": encoders, "feature_names": feature_names,
        "target_mode": target_mode, "metrics": metrics,
        "y_true": y_test_abs, "y_pred": np.asarray(y_pred_abs), "y_pers": y_pers,
        "y_lo": np.asarray(y_lo), "y_hi": np.asarray(y_hi),
        "shock_proba": np.asarray(shock_proba),
        "dates": test_df["date"].values if "date" in test_df.columns else None,
    }


def _row_to_frame(row: dict, feature_names: list[str], encoders: dict) -> pd.DataFrame:
    row = ensure_feature_row(row)
    data = {c: [float(row.get(c, 0.0))] for c in FEATURE_COLS}
    for col in CAT_COLS:
        le = encoders.get(col)
        val = str(row.get(col, "Unknown"))
        if le is not None and hasattr(le, "classes_") and val in set(le.classes_):
            data[col + "_enc"] = [int(le.transform([val])[0])]
        else:
            data[col + "_enc"] = [-1]
    return pd.DataFrame([{k: data.get(k, [0])[0] for k in feature_names}])


def predict_single(model, encoders, feature_names, target_mode, row: dict) -> float:
    X = _row_to_frame(row, feature_names, encoders)
    raw = model.predict(X)[0]
    y_t = float(ensure_feature_row(row).get("usdprice", row.get("lag_1", 0)))
    return float(reconstruct_price(y_t, raw)) if target_mode == "residual" else float(raw)


def predict_two_stage(bundle: dict, row: dict) -> dict:
    row = ensure_feature_row(row)
    model = bundle["model"]; encoders = bundle["encoders"]
    feature_names = bundle["feature_names"]; target_mode = bundle.get("target_mode", "residual")
    X = _row_to_frame(row, feature_names, encoders)
    y_t = float(row["usdprice"])
    raw = model.predict(X)[0]
    y_hat = float(reconstruct_price(y_t, raw)) if target_mode == "residual" else float(raw)
    q10_model, q90_model = bundle.get("q10_model"), bundle.get("q90_model")
    qcal = float(bundle.get("qcal") or 0.0)
    if q10_model is not None and q90_model is not None and target_mode == "residual":
        lo = min(q10_model.predict(X)[0], q90_model.predict(X)[0]) - qcal
        hi = max(q10_model.predict(X)[0], q90_model.predict(X)[0]) + qcal
        y_lo, y_hi = float(reconstruct_price(y_t, lo)), float(reconstruct_price(y_t, hi))
    else:
        band = abs(y_hat) * 0.03 + 0.01
        y_lo, y_hi = y_hat - band, y_hat + band
    shock_clf = bundle.get("shock_clf")
    proba = float(shock_clf.predict_proba(X)[0, 1]) if shock_clf is not None else 0.0
    return {"y_hat": y_hat, "y_p10": y_lo, "y_p90": y_hi, "shock_proba": proba, "y_t": y_t}
