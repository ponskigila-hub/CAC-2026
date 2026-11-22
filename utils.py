"""Utility functions for FoodShock Streamlit app (V2 feature set)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "usdprice",  # y_t current price — residual basis
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "rolling_std_6",
    "rolling_min_6", "rolling_max_6", "price_to_mean6", "cv_6",
    "pct_change_1m", "pct_change_3m", "pct_change_12m",
    "month_sin", "month_cos",
    # macro / spillover
    "fx_pct_1m", "fx_pct_3m", "fx_vol_6m",
    "asean_comm_mom_1m", "country_vs_asean_mom",
]

CAT_COLS = ["countryiso3", "commodity_group", "pricetype"]

_DEFAULT_SHOCK_PROB_THRESHOLD = 0.67

def get_shock_prob_threshold() -> float:
    """Operational threshold from pipeline artefact; fallback to constant."""
    from pathlib import Path as _P
    for p in (_P(__file__).resolve().parent / "data" / "app_shock_threshold.txt",
              _P("data/app_shock_threshold.txt")):
        try:
            if p.exists():
                return float(p.read_text().strip().split()[0])
        except Exception:
            pass
    return float(_DEFAULT_SHOCK_PROB_THRESHOLD)

# Back-compat alias (evaluated at import; prefer get_shock_prob_threshold() at runtime)
SHOCK_PROB_THRESHOLD = get_shock_prob_threshold()
SHOCK_PCT_THRESHOLD = 0.05


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def reconstruct_price(y_t, residual_hat):
    """ŷ_{t+1} = y_t + Δŷ_{t+1}."""
    return np.asarray(y_t, dtype=float) + np.asarray(residual_hat, dtype=float)


def pct_change_from_prices(current: float, predicted: float) -> float:
    if current is None or current == 0 or (isinstance(current, float) and np.isnan(current)):
        return 0.0
    return (float(predicted) - float(current)) / float(current)


def classify_shock(pct_change: float, shock_proba: float | None = None) -> tuple[str, str]:
    """
    Upward affordability shock (SDG 2.c): +5% MoM policy tail + Stage-2 probability.
    pct_change may be fraction (0.05) or percent (5.0).
    """
    pct = float(pct_change)
    pct_frac = pct / 100.0 if abs(pct) > 1.5 else pct
    proba = float(shock_proba) if shock_proba is not None else 0.0
    high_prob = proba >= get_shock_prob_threshold()
    high_up = pct_frac > SHOCK_PCT_THRESHOLD
    if high_prob or high_up:
        return "FOOD SHOCK ALERT", "#F43F5E"
    if pct_frac >= 0.03 or proba >= 0.25:
        return "WASPADA / ELEVATED RISK", "#F59E0B"
    return "NORMAL", "#10B981"


def ensure_feature_row(row: dict) -> dict:
    """Fill missing feature keys with safe defaults."""
    out = dict(row)
    y = float(out.get("usdprice") or out.get("lag_1") or 1.0)
    out.setdefault("usdprice", y)
    for c in FEATURE_COLS:
        if c not in out or out[c] is None or (isinstance(out[c], float) and np.isnan(out[c])):
            if c.startswith("fx_") or "mom" in c or c.startswith("pct_"):
                out[c] = 0.0
            elif c in ("month_sin", "month_cos"):
                out[c] = 0.0 if c == "month_sin" else 1.0
            else:
                out[c] = y
    for c in CAT_COLS:
        out.setdefault(c, "Unknown")
    return out
