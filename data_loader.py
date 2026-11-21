"""Cached data loaders for FoodShock Streamlit app."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data(show_spinner=False)
def load_model_ready() -> pd.DataFrame:
    path = DATA_DIR / "foodshock_asean_model_ready.csv"
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_country_coverage() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "fpma_wfp_country_coverage.csv")


@st.cache_data(show_spinner=False)
def load_commodity_coverage() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "fpma_wfp_commodity_coverage.csv")


@st.cache_data(show_spinner=False)
def load_model_results() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "model_evaluation_results.csv")


@st.cache_data(show_spinner=False)
def load_eval_by_commodity() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "model_eval_by_commodity.csv")


@st.cache_data(show_spinner=False)
def load_eval_by_country() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "model_eval_by_country.csv")


@st.cache_data(show_spinner=False)
def load_eval_by_pricetype() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "model_eval_by_pricetype.csv")


@st.cache_data(show_spinner=False)
def load_shock_metrics() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "model_eval_shock_classification.csv")


def get_filter_options(df: pd.DataFrame) -> dict:
    """Return sorted unique values for interactive filters."""
    return {
        "countries": sorted(df["countryiso3"].dropna().unique().tolist()),
        "commodities": sorted(df["commodity_group"].dropna().unique().tolist()),
        "pricetypes": sorted(df["pricetype"].dropna().unique().tolist()),
    }
