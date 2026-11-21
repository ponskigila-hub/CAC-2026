"""Utility functions for FoodShock Streamlit app."""
from __future__ import annotations

import numpy as np
import pandas as pd


def smape(y_true, y_pred) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def reconstruct_price(lag_1: np.ndarray | float, residual_hat: np.ndarray | float) -> np.ndarray | float:
    """Reconstruct absolute price: ŷ_{t+1} = y_t + Δŷ_{t+1}."""
    return np.asarray(lag_1) + np.asarray(residual_hat)


def classify_shock(pct_change: float) -> tuple[str, str]:
    """
    Classify predicted percentage change into alert levels.
    Returns (label, color_hex).
    """
    pct = float(pct_change) * 100 if abs(pct_change) < 1.5 else float(pct_change)
    # Accept either fraction (0.05) or already-percent (5.0)
    if abs(pct) < 1.5:
        pct = pct * 100

    if pct > 5.0:
        return "FOOD SHOCK ALERT", "#d32f2f"
    if pct >= 3.0:
        return "WASPADA / MODERATE RISE", "#f57c00"
    if pct <= -5.0:
        return "FOOD SHOCK ALERT (DROP)", "#1565c0"
    if pct <= -3.0:
        return "WASPADA / MODERATE DROP", "#0288d1"
    return "NORMAL", "#2e7d32"


def pct_change_from_prices(current: float, predicted: float) -> float:
    """Compute percentage change from current price to predicted price."""
    if current is None or current == 0 or np.isnan(current):
        return 0.0
    return (float(predicted) - float(current)) / float(current)


FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "rolling_std_6",
    "rolling_min_6", "rolling_max_6", "price_to_mean6", "cv_6",
    "pct_change_1m", "pct_change_3m", "pct_change_12m",
    "month_sin", "month_cos",
]

CAT_COLS = ["countryiso3", "commodity_group", "pricetype"]
