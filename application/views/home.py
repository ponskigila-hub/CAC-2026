"""Home & Executive Overview — walk-forward metrics."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from application.data_loader import load_model_results, load_model_ready, load_shock_metrics
from application.styles import inject_css, hero, section, kpi_card

def _shock_ops_row(shock_df):
    """Prefer Operational Stage-2 metrics for app KPIs."""
    if shock_df is None or len(shock_df) == 0:
        return None, None
    ops = shock_df[shock_df["method"].astype(str).str.contains("Operational|Calibrated_Operational", case=False, na=False)]
    f1 = shock_df[shock_df["method"].astype(str).str.contains("F1", case=False, na=False)]
    return (ops.iloc[0] if len(ops) else shock_df.iloc[0]), (f1.iloc[0] if len(f1) else None)



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
                ("7-Fold Walk-Forward", "pill-sky"),
                ("Two-Stage Alert", "pill-amber"),
            ],
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "FoodShock forecasts monthly staple-food prices across ASEAN markets using "
        "**residual forecasting**, **calibrated quantile intervals (80% target coverage)**, and a **Stage-2 "
        "shock risk score** — supporting SDG 2 (Zero Hunger) price-stability monitoring."
    )

    df = load_model_ready()
    results = load_model_results()
    shock = load_shock_metrics()
    shock_ops, shock_f1 = _shock_ops_row(shock)
    if len(shock) and "Hybrid_Operational_Alert" in shock.method.values:
        shock_ops=shock.loc[shock.method=="Hybrid_Operational_Alert"].iloc[0]

    n_countries = df["countryiso3"].nunique()
    n_commodities = df["commodity_group"].nunique()
    n_obs = len(df)

    best=results.sort_values("RMSE_micro").iloc[0]
    rmse_val=best["RMSE_micro"]; ss=best["SS_RMSE"]; best_name=str(best["model"])
    stage2=shock.loc[shock.method=="Stage2_Only"].iloc[0] if "Stage2_Only" in shock.method.values else shock_ops

    st.markdown(
        f"""
        <div class="kpi-grid">
            {kpi_card("ASEAN Countries", str(n_countries), "Model-ready country coverage")}
            {kpi_card("Commodity Groups", str(n_commodities), "Rice · Meat · Cooking Oil")}
            {kpi_card("Clean Observations", f"{n_obs:,}", "Model-ready panel + macro features")}
            {kpi_card("Lowest benchmark RMSE", _fmt(rmse_val), f"SS_RMSE={_fmt(ss,3)} · {best_name[:28]}")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(section("System Architecture"), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow-row">
            <div class="flow-step"><div class="num">STEP 01</div><div class="title">Audit & Macro FE</div>
            <div class="desc">≥60 months · implied FX · ASEAN spillover</div></div>
            <div class="flow-step"><div class="num">STEP 02</div><div class="title">Walk-Forward</div>
            <div class="desc">7 expanding folds · all 7 countries OOS</div></div>
            <div class="flow-step"><div class="num">STEP 03</div><div class="title">Point Residual</div>
            <div class="desc">Δy from y_t (usdprice) · fair Optuna</div></div>
            <div class="flow-step"><div class="num">STEP 04</div><div class="title">Quantile P10–P90</div>
            <div class="desc">Upside tail interval · 80% target coverage</div></div>
            <div class="flow-step"><div class="num">STEP 05</div><div class="title">Shock Classifier</div>
            <div class="desc">Hybrid alert · validation threshold · empirical +5% MoM</div></div>
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

    st.markdown(section("Hybrid Alert & Quantile Coverage"), unsafe_allow_html=True)
    if shock_ops is not None:
        s = shock_ops
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Hybrid precision", f"{float(s['Precision'])*100:.1f}%")
        c2.metric("Hybrid recall", f"{float(s['Recall'])*100:.1f}%")
        c3.metric("Stage-2 PR-AUC", f"{float(stage2['PR_AUC']):.3f}")
        c4.metric("Hybrid FPR", f"{float(s['FPR'])*100:.1f}%")
        c5.metric("80% interval coverage", f"{float(s.get('Overall_PICP_P10_P90', float('nan')))*100:.1f}%")
        st.caption(
            f"Base rate={float(s.get('Base_Rate', 0)):.3f} · "
            f"Mean fold risk-score threshold={float(s.get('Mean_Train_Threshold', 0.50)):.2f} · "
            f"Policy MoM threshold=+5% (empirical upper-tail rule; inspired by WFP ALPS / AFSIS)"
        )
    else:
        st.info("Shock metrics file not found.")

    st.info("FoodShock is a historical screening and market-investigation tool, not a guaranteed early-warning service. The frozen CatBoost system is specified in advance; benchmark ranking is reported independently. Price levels can have high R² even when incremental improvement over persistence is small.")
    st.markdown(
        '<p class="footer-note">FoodShock · MLSC 2026 · SDG 2 — Zero Hunger</p>',
        unsafe_allow_html=True,
    )
