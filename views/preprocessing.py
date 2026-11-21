"""Preprocessing, audit coverage, zero-leakage FE."""
from __future__ import annotations

import streamlit as st

from data_loader import load_country_coverage, load_commodity_coverage, load_model_ready
from styles import inject_css, hero, section


def render():
    inject_css()

    st.markdown(
        hero("Preprocessing & Data Audit",
             "Coverage ≥ 60 months · unit standardisation · calendar-safe zero-leakage features",
             pills=[("Data Integrity", "pill-green"), ("Zero Leakage", "pill-sky")]),
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
            Full monthly re-index without forward-fill of raw prices; lags from shift(1).
            </span></div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(section("Country Coverage (series ≥ 60 months)"), unsafe_allow_html=True)
    st.dataframe(load_country_coverage(), use_container_width=True, hide_index=True)

    st.markdown(section("Commodity Group Coverage"), unsafe_allow_html=True)
    st.dataframe(load_commodity_coverage(), use_container_width=True, hide_index=True)

    st.info(
        "**IDN & THA:** series end around May 2024 and March 2020 respectively. "
        "The official test window (Jun 2024 – Nov 2025) therefore contains only "
        "training history for these countries — consistent with strict time-based validation."
    )

    st.markdown(section("Residual Target Formulation"), unsafe_allow_html=True)
    st.latex(r"\Delta y_{t+1} = y_{t+1} - y_t \qquad \hat{y}_{t+1} = y_t + \widehat{\Delta y}_{t+1}")

    st.markdown(section("Feature Preview"), unsafe_allow_html=True)
    df = load_model_ready()
    cols = [
        "date", "countryiso3", "commodity_group", "pricetype", "usdprice",
        "lag_1", "lag_12", "rolling_mean_6", "cv_6", "pct_change_1m",
        "month_sin", "month_cos", "target_price_usd_t1",
    ]
    existing = [c for c in cols if c in df.columns]
    st.dataframe(df[existing].head(12), use_container_width=True, hide_index=True)

    st.markdown('<p class="footer-note">FoodShock · MLSC 2026 · SDG 2 — Zero Hunger</p>', unsafe_allow_html=True)
