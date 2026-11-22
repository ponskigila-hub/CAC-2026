"""Home & Executive Overview — V2 metrics."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from data_loader import load_model_results, load_model_ready, load_shock_metrics
from styles import inject_css, hero, section, kpi_card


def _fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "—"


def render():
    inject_css()

    st.markdown(
        hero(
            "FoodShock",
            "Cross-Country ASEAN Food Price Forecasting & Early Shock Detection System",
            pills=[
                ("SDG 2 · Zero Hunger", "pill-green"),
                ("Walk-Forward V2", "pill-sky"),
                ("Two-Stage Alert", "pill-amber"),
            ],
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "FoodShock forecasts monthly staple-food prices across ASEAN markets using "
        "**residual forecasting**, **quantile uncertainty (P10–P90)**, and a **Stage-2 probabilistic "
        "shock classifier** — supporting SDG 2 (Zero Hunger) price-stability monitoring."
    )

    df = load_model_ready()
    results = load_model_results()
    shock = load_shock_metrics()

    n_countries = df["countryiso3"].nunique()
    n_commodities = df["commodity_group"].nunique()
    n_obs = len(df)

    # Prefer best non-persistence by RMSE_micro if available
    r = results.copy()
    if "RMSE_micro" in r.columns:
        r_ml = r[~r["model"].astype(str).str.contains("Persistence", case=False, na=False)]
        best = r_ml.sort_values("RMSE_micro").iloc[0] if len(r_ml) else r.iloc[0]
        rmse_val = best.get("RMSE_micro", best.get("RMSE", float("nan")))
        ss = best.get("SS_RMSE", float("nan"))
        best_name = str(best["model"])
    else:
        best = r.sort_values("MAE").iloc[0]
        rmse_val = best.get("RMSE", float("nan"))
        ss = float("nan")
        best_name = str(best["model"])

    recall = shock.iloc[0]["Recall"] if len(shock) else float("nan")
    picp = shock.iloc[0].get("PICP_P10_P90", float("nan")) if len(shock) else float("nan")

    st.markdown(
        f"""
        <div class="kpi-grid">
            {kpi_card("ASEAN Countries", str(n_countries), "OOS via expanding walk-forward")}
            {kpi_card("Commodity Groups", str(n_commodities), "Rice · Meat · Cooking Oil")}
            {kpi_card("Clean Observations", f"{n_obs:,}", "Model-ready panel + macro features")}
            {kpi_card("Best RMSE / SS", _fmt(rmse_val), f"SS_RMSE={_fmt(ss,3)} · {best_name[:28]}")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(section("System Architecture (V2)"), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow-row">
            <div class="flow-step"><div class="num">STEP 01</div><div class="title">Audit & Macro FE</div>
            <div class="desc">≥60 months · implied FX · ASEAN spillover</div></div>
            <div class="flow-step"><div class="num">STEP 02</div><div class="title">Walk-Forward</div>
            <div class="desc">6 expanding folds · all 7 countries OOS</div></div>
            <div class="flow-step"><div class="num">STEP 03</div><div class="title">Point Residual</div>
            <div class="desc">Δy from y_t (usdprice) · fair Optuna</div></div>
            <div class="flow-step"><div class="num">STEP 04</div><div class="title">Quantile P10–P90</div>
            <div class="desc">Upside tail interval · PICP ≈ 80%</div></div>
            <div class="flow-step"><div class="num">STEP 05</div><div class="title">Shock Classifier</div>
            <div class="desc">P(Shock) · thr≈0.50 · ALPS +5%</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(section("Official Walk-Forward Benchmark"), unsafe_allow_html=True)
    show_cols = [c for c in [
        "model", "MAE_micro", "RMSE_micro", "sMAPE_micro", "MAE_macro",
        "SS_RMSE", "DM_pvalue", "Persistence_RMSE",
    ] if c in results.columns]
    if not show_cols:
        show_cols = list(results.columns)
    fmt = {c: "{:.4f}" for c in show_cols if c != "model"}
    st.dataframe(
        results[show_cols].style.format(fmt, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(section("Stage-2 Shock Classifier & Quantile Coverage"), unsafe_allow_html=True)
    if len(shock):
        s = shock.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Recall", f"{float(s['Recall'])*100:.1f}%")
        c2.metric("F1", f"{float(s['F1']):.3f}")
        c3.metric("PR-AUC", f"{float(s.get('PR_AUC', float('nan'))):.3f}")
        c4.metric("ROC-AUC", f"{float(s.get('ROC_AUC', float('nan'))):.3f}")
        c5.metric("PICP P10–P90", f"{float(s.get('PICP_P10_P90', float('nan')))*100:.1f}%")
        st.caption(
            f"Base rate={float(s.get('Base_Rate', 0)):.3f} · "
            f"Optimal probability threshold={float(s.get('Optimal_Threshold', 0.50)):.2f} · "
            f"Policy MoM threshold=+5% (WFP ALPS / AFSIS)"
        )
    else:
        st.info("Shock metrics file not found.")

    st.markdown(
        '<p class="footer-note">FoodShock · MLSC 2026 · SDG 2 — Zero Hunger</p>',
        unsafe_allow_html=True,
    )
