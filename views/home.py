"""Home & Executive Overview."""
from __future__ import annotations

import streamlit as st

from data_loader import load_model_results, load_model_ready
from styles import inject_css, hero, section, kpi_card


def render():
    inject_css()

    st.markdown(
        hero(
            "FoodShock",
            "Cross-Country ASEAN Food Price Forecasting & Early Shock Detection System",
            pills=[
                ("SDG 2 · Zero Hunger", "pill-green"),
                ("Residual Forecasting", "pill-sky"),
                ("Early Warning", "pill-amber"),
            ],
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "FoodShock forecasts monthly staple-food prices across ASEAN markets and "
        "issues early warnings for potential price shocks — supporting price stability "
        "monitoring and evidence-based policy under **SDG 2 (Zero Hunger)**."
    )

    df = load_model_ready()
    results = load_model_results()
    n_countries = df["countryiso3"].nunique()
    n_commodities = df["commodity_group"].nunique()
    n_obs = len(df)
    tuned = results[results["model"].astype(str).str.contains("Tuned_Residual", na=False)]
    if len(tuned):
        best = tuned.iloc[0]
    else:
        best = results.sort_values("MAE").iloc[0]

    st.markdown(
        f"""
        <div class="kpi-grid">
            {kpi_card("ASEAN Countries", str(n_countries), "Long-series coverage ≥ 60 months")}
            {kpi_card("Commodity Groups", str(n_commodities), "Rice · Meat · Cooking Oil")}
            {kpi_card("Clean Observations", f"{n_obs:,}", "Model-ready panel")}
            {kpi_card("Best RMSE / R²", f"{best['RMSE']:.4f}", f"R² = {best['R2']:.4f} · {best['model']}")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(section("System Architecture"), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow-row">
            <div class="flow-step"><div class="num">STEP 01</div><div class="title">Audit & Scope</div>
            <div class="desc">≥ 60 months · 7 countries · 5 commodity groups</div></div>
            <div class="flow-step"><div class="num">STEP 02</div><div class="title">Standardise</div>
            <div class="desc">USD prices · KG / L units · Retail vs Wholesale</div></div>
            <div class="flow-step"><div class="num">STEP 03</div><div class="title">Calendar-Safe FE</div>
            <div class="desc">No forward-fill of prices · lags from shift(1)</div></div>
            <div class="flow-step"><div class="num">STEP 04</div><div class="title">Residual Forecast</div>
            <div class="desc">Δyₜ₊₁ then reconstruct absolute price</div></div>
            <div class="flow-step"><div class="num">STEP 05</div><div class="title">Shock Alert</div>
            <div class="desc">Flag |%Δ| &gt; 5% as Food Shock</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(section("Why Residual Forecasting?"), unsafe_allow_html=True)
    st.info(
        "Absolute price targets mix incompatible scales (Rice ≈ USD 0.77 vs Beef ≈ USD 6.7). "
        "Training on residuals removes level bias so models focus on *changes*. "
        "After reconstruction, the tuned residual model reaches **RMSE 0.2445** and **R² 0.9890**, "
        "improving on the Persistence baseline for RMSE and R²."
    )

    st.markdown(section("Official Benchmark Results"), unsafe_allow_html=True)
    st.dataframe(
        results.sort_values("MAE").style.format(
            {"MAE": "{:.4f}", "RMSE": "{:.4f}", "sMAPE": "{:.2f}", "R2": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<p class="footer-note">FoodShock · MLSC 2026 · SDG 2 — Zero Hunger</p>',
        unsafe_allow_html=True,
    )
