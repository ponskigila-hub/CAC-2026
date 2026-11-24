"""V9 official benchmark, exact official holdout, and custom training."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from application.data_loader import (
    load_model_ready, load_model_results, load_shock_metrics,
    load_eval_by_commodity, load_eval_by_pricetype, get_filter_options,
)
from core.models import available_algorithms, train_and_evaluate, make_bundle
from core.model_store import load_manifest
from pathlib import Path
DATA=Path(__file__).resolve().parents[2]/"data"
FIG=DATA.parent/"figures"
from application.styles import inject_css, apply_plot_style, hero, section, COLORS


def render():
    inject_css()
    apply_plot_style()

    st.markdown(
        hero(
            "Model Training & Evaluation",
            "Residual vs Direct · live hyperparameters · expanding / time-based validation",
            pills=[("7-Fold Walk-Forward", "pill-sky"), ("vs Persistence", "pill-amber")],
        ),
        unsafe_allow_html=True,
    )

    st.markdown(section("Official Walk-Forward Benchmark (pre-computed)"), unsafe_allow_html=True)
    results = load_model_results()
    show_cols = [c for c in [
        "model", "MAE_micro", "RMSE_micro", "sMAPE_micro", "R2_micro",
        "MAE_macro", "SS_RMSE", "DM_stat", "DM_pvalue", "SS_RMSE_CI95",
    ] if c in results.columns]
    if not show_cols:
        show_cols = list(results.columns)
    fmt = {c: "{:.4f}" for c in show_cols if c != "model" and pd.api.types.is_numeric_dtype(results[c])}
    st.dataframe(
        results[show_cols].style.format(fmt, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    st.markdown(section("Hybrid Alert & Baselines (Walk-Forward OOS)"), unsafe_allow_html=True)
    shock = load_shock_metrics()
    if len(shock):
        st.dataframe(shock.style.format(precision=4), use_container_width=True, hide_index=True)
    else:
        st.warning("model_eval_shock_classification.csv not found")

    st.caption("PR-AUC and ROC-AUC refer to each continuous ranking score, not the binary hybrid decision. Hybrid rows report decision metrics only.")
    t1, t2, t3 = st.tabs(["By Commodity", "By Price Type", "By Country"])
    with t1:
        st.dataframe(load_eval_by_commodity(), use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(load_eval_by_pricetype(), use_container_width=True, hide_index=True)

    with t3:
        st.dataframe(pd.read_csv(DATA/"model_eval_by_country.csv"),use_container_width=True,hide_index=True)
    st.markdown(section("Feature Ablation & Explainability"),unsafe_allow_html=True)
    st.dataframe(pd.read_csv(DATA/"v9_feature_ablation.csv"),use_container_width=True,hide_index=True)
    st.caption("Controlled ablation: the same fitted time periods and tuned CatBoost parameters; no OOS-driven feature selection.")
    st.image(str(FIG/"shap_summary.png"),caption="SHAP on the exact frozen official CatBoost holdout model (not a retrained proxy).",use_container_width=True)
    st.markdown(section("Frozen Official Holdout Model"),unsafe_allow_html=True)
    manifest=load_manifest()
    st.caption(f"Fit through {manifest['official_fit_end']} · calibration {manifest['official_cal_start']} to {manifest['official_cal_end']} · 18-month holdout starts {manifest['official_holdout_cutoff']}")
    st.json({"parameters":manifest["official_params"],"threshold":manifest["official_threshold"],"qcal":manifest["official_qcal"],"holdout_metrics":manifest["official_holdout_metrics"]},expanded=False)
    with st.expander("Per-fold temporal split and validation thresholds"):
        st.dataframe(pd.read_csv(DATA/"v9_fold_metrics.csv"),hide_index=True)
    st.markdown("---")
    st.markdown(section("Interactive Training Panel"), unsafe_allow_html=True)
    st.caption(
        "Custom training uses the same purged fit/calibration engine with an 18-month holdout. "
        "Its metrics and fitted threshold are separate from the official seven-fold benchmark."
    )

    df = load_model_ready()
    opts = get_filter_options(df)
    left, right = st.columns([1, 1.15])

    with left:
        st.markdown("##### Configuration")
        algorithm = st.selectbox("Algorithm", available_algorithms(), index=0)
        target_mode = st.radio(
            "Target Approach",
            options=["residual", "direct"],
            format_func=lambda x: (
                "Residual (Δy = yₜ₊₁ − yₜ)" if x == "residual" else "Direct (yₜ₊₁)"
            ),
        )
        sel_countries = st.multiselect("Countries (empty = All)", opts["countries"])
        sel_commodities = st.multiselect("Commodities (empty = All)", opts["commodities"])
        n_estimators = st.slider("n_estimators", 50, 600, 150, 25)
        learning_rate = st.select_slider(
            "learning_rate", options=[0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15], value=0.05
        )
        max_depth = st.slider("max_depth", 3, 12, 6)
        tune = st.checkbox("Tune within the fit period (25 Optuna trials)",value=False)
        run = st.button("Run Training & Evaluation", type="primary", use_container_width=True)

    with right:
        st.markdown("##### Results")
        if run:
            mask = pd.Series(True, index=df.index)
            if sel_countries:
                mask &= df["countryiso3"].isin(sel_countries)
            if sel_commodities:
                mask &= df["commodity_group"].isin(sel_commodities)
            train_df = df.loc[mask].copy()
            if len(train_df) < 200:
                st.error("Too few rows after filtering.")
            else:
                with st.spinner("Training point + quantile + shock classifier..."):
                    try:
                        out = train_and_evaluate(
                            train_df, algorithm=algorithm, target_mode=target_mode,
                            n_estimators=n_estimators, learning_rate=learning_rate,
                            max_depth=max_depth,tune_trials=25 if tune else 0,
                        )
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        out = None

                if out:
                    st.session_state["two_stage_bundle"]=make_bundle(out)
                    st.session_state["trained_metrics"]=out["metrics"]

                    m = out["metrics"]
                    st.success(f"Done · fit={m['n_train']:,} · calibration={m['n_calibration']:,} · test={m['n_test']:,}")
                    st.caption(f"Fit ends {out['fit_end']} · calibration ends {out['cal_end']} · holdout starts {out['cutoff']} · threshold={out['threshold']:.3f}")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("MAE", f"{m['MAE']:.4f}", f"{m['MAE']-m['Persistence_MAE']:+.4f} vs Pers.")
                    d2.metric("RMSE", f"{m['RMSE']:.4f}", f"{m['RMSE']-m['Persistence_RMSE']:+.4f} vs Pers.")
                    d3.metric("sMAPE", f"{m['sMAPE']:.2f}%")
                    d4.metric("80% interval coverage", f"{m['PICP_P10_P90']*100:.1f}%" if pd.notna(m['PICP_P10_P90']) else "Not available")

                    st.write("Hybrid alert metrics (custom holdout):", {k:m[k] for k in ("Precision","Recall","F1","FPR","TP","FP","FN","TN")})
                    st.write("Stage-2 ranking metrics:", {k:m[k] for k in ("PR_AUC","ROC_AUC")})
                    comp = pd.DataFrame([
                        {"Model": m["model"], "MAE": m["MAE"], "RMSE": m["RMSE"],
                         "sMAPE": m["sMAPE"], "R2": m["R2"]},
                        {"Model": "Persistence (y_t)", "MAE": m["Persistence_MAE"],
                         "RMSE": m["Persistence_RMSE"], "sMAPE": m["Persistence_sMAPE"],
                         "R2": m["Persistence_R2"]},
                    ])
                    st.dataframe(
                        comp.style.format(
                            {"MAE": "{:.4f}", "RMSE": "{:.4f}", "sMAPE": "{:.2f}", "R2": "{:.4f}"}
                        ),
                        use_container_width=True, hide_index=True,
                    )

                    fig, ax = plt.subplots(figsize=(5.5, 4.8))
                    ax.scatter(out["y_true"], out["y_pred"], alpha=0.28, s=14,
                               c=COLORS["primary"], label="Model")
                    lims = [min(out["y_true"].min(), out["y_pred"].min()),
                            max(out["y_true"].max(), out["y_pred"].max())]
                    ax.plot(lims, lims, color=COLORS["coral"], ls="--", lw=1.4, label="Perfect fit")
                    ax.set_xlabel("Actual (USD)")
                    ax.set_ylabel("Predicted (USD)")
                    ax.set_title(m["model"])
                    ax.legend(fontsize=8)
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)

        elif "trained_metrics" in st.session_state:
            st.info("Model already in session — re-run to update, or open Prediction.")
            st.json(st.session_state["trained_metrics"])
        else:
            st.caption("Configure parameters on the left, then run training.")

    st.markdown('<p class="footer-note">FoodShock · MLSC 2026 · SDG 2 — Zero Hunger</p>', unsafe_allow_html=True)
