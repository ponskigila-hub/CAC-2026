"""Preprocessing, audit coverage, time-aware FE."""
from __future__ import annotations

import streamlit as st

from application.data_loader import load_country_coverage, load_commodity_coverage, load_model_ready
from core.models import validate_panel
from core.utils import FEATURE_COLS
import pandas as pd
from application.styles import inject_css, hero, section


def render():
    inject_css()

    st.markdown(
        hero("Preprocessing & Data Audit",
             "Coverage ≥ 60 months · unit standardisation · calendar-safe time-aware features",
             pills=[("Data Integrity", "pill-green"), ("Temporal Audit", "pill-sky")]),
        unsafe_allow_html=True,
    )

    st.markdown(section("Data Integrity Rules"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="panel">
            <strong style="color:#6EE7B7;">Audit ≥ 60 months</strong><br/>
            <span style="color:#94A3B8; font-size:0.85rem;">
            Only series with at least 60 monthly observations are retained for modelling.
            </span></div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="panel">
            <strong style="color:#6EE7B7;">Standardisation</strong><br/>
            <span style="color:#94A3B8; font-size:0.85rem;">
            USD prices; KG for rice/meat; L for cooking oil; explicit Retail vs Wholesale.
            </span></div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="panel">
            <strong style="color:#6EE7B7;">Calendar-Safe FE</strong><br/>
            <span style="color:#94A3B8; font-size:0.85rem;">
            Full monthly re-index without forward-fill of raw prices; lags from shift(1). Current USD price is known at forecast origin.
            </span></div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(section("Country Coverage (series ≥ 60 months)"), unsafe_allow_html=True)
    st.dataframe(load_country_coverage(), use_container_width=True, hide_index=True)

    st.markdown(section("Commodity Group Coverage"), unsafe_allow_html=True)
    st.dataframe(load_commodity_coverage(), use_container_width=True, hide_index=True)

    st.info("V9 uses seven expanding walk-forward folds (June 2019–June 2025 starts). "
            "Training labels must mature strictly before each cutoff; inner calibration uses the same rule. "
            "Series eligibility was selected retrospectively on the supplied panel, so this is not a fully prospective deployment study.")

    st.markdown(section("Residual Target Formulation"), unsafe_allow_html=True)
    st.latex(r"\Delta y_{t+1} = y_{t+1} - y_t \qquad \hat{y}_{t+1} = y_t + \widehat{\Delta y}_{t+1}")

    st.markdown(section("Feature Preview"), unsafe_allow_html=True)
    df = load_model_ready()
    validate_panel(df)
    a,b,c=st.columns(3)
    a.metric("Model-ready rows",f"{len(df):,}")
    b.metric("Duplicate series/month",int(df.duplicated(["series_id","date"]).sum()))
    c.metric("Missing required features",int(df[FEATURE_COLS].isna().sum().sum()))
    st.caption("Source: Global Food Prices Database (WFP), distributed on Kaggle by abhishekgupta56447. The original raw snapshot is not bundled. See docs/DATA_SOURCES.md.")
    cols = [
        "date", "countryiso3", "commodity_group", "pricetype", "usdprice",
        "lag_1", "lag_12", "rolling_mean_6", "cv_6", "pct_change_1m",
        "month_sin", "month_cos", "target_price_usd_t1",
    ]
    existing = [c for c in cols if c in df.columns]
    st.dataframe(df[existing].head(12), use_container_width=True, hide_index=True)

    st.markdown('<p class="footer-note">FoodShock · MLSC 2026 · SDG 2 — Zero Hunger</p>', unsafe_allow_html=True)
