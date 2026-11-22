"""Interactive model training — V2 residual + two-stage artefacts."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_loader import (
    load_model_ready, load_model_results, load_shock_metrics,
    load_eval_by_commodity, load_eval_by_pricetype, get_filter_options,
)
from models import available_algorithms, train_and_evaluate
from styles import inject_css, apply_plot_style, hero, section, COLORS


def render():
    inject_css()
    apply_plot_style()

    st.markdown(
        hero(
            "Model Training & Evaluation",
            "Residual vs Direct · live hyperparameters · expanding / time-based validation",
            pills=[("Walk-Forward V2", "pill-sky"), ("vs Persistence", "pill-amber")],
        ),
        unsafe_allow_html=True,
    )

    st.markdown(section("Official Walk-Forward Benchmark (pre-computed)"), unsafe_allow_html=True)
    results = load_model_results()
    show_cols = [c for c in [
        "model", "MAE_micro", "RMSE_micro", "sMAPE_micro", "R2_micro",
        "MAE_macro", "SS_RMSE", "DM_stat", "DM_pvalue",
    ] if c in results.columns]
    if not show_cols:
        show_cols = list(results.columns)
    fmt = {c: "{:.4f}" for c in show_cols if c != "model"}
    st.dataframe(
        results[show_cols].style.format(fmt, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    st.markdown(section("Stage-2 Shock Classifier (Walk-Forward OOF)"), unsafe_allow_html=True)
    shock = load_shock_metrics()
    if len(shock):
        st.dataframe(shock.style.format(precision=4), use_container_width=True, hide_index=True)
    else:
        st.warning("model_eval_shock_classification.csv not found")

    t1, t2 = st.tabs(["By Commodity", "By Price Type"])
    with t1:
        st.dataframe(load_eval_by_commodity(), use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(load_eval_by_pricetype(), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(section("Interactive Training Panel"), unsafe_allow_html=True)
    st.caption(
        "Interactive runs use a **time-based hold-out** (last N months) for speed. "
        "Paper results use **6-fold expanding-window walk-forward** (see benchmark table above)."
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
        n_estimators = st.slider("n_estimators", 50, 600, 250, 50)
        learning_rate = st.select_slider(
            "learning_rate", options=[0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15], value=0.05
        )
        max_depth = st.slider("max_depth", 3, 12, 6)
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
                            max_depth=max_depth,
                        )
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        out = None

                if out:
                    # Persist full two-stage bundle
                    for k in ("model", "q10_model", "q90_model", "shock_clf",
                              "encoders", "feature_names", "target_mode", "metrics"):
                        st.session_state[k if k != "model" else "trained_model"] = out[k]
                    st.session_state["trained_encoders"] = out["encoders"]
                    st.session_state["trained_features"] = out["feature_names"]
                    st.session_state["trained_mode"] = out["target_mode"]
                    st.session_state["trained_metrics"] = out["metrics"]
                    st.session_state["q10_model"] = out["q10_model"]
                    st.session_state["q90_model"] = out["q90_model"]
                    st.session_state["shock_clf"] = out["shock_clf"]
                    st.session_state["two_stage_bundle"] = {
                        "model": out["model"],
                        "q10_model": out["q10_model"],
                        "q90_model": out["q90_model"],
                        "shock_clf": out["shock_clf"],
                        "encoders": out["encoders"],
                        "feature_names": out["feature_names"],
                        "target_mode": out["target_mode"],
                    }

                    m = out["metrics"]
                    st.success(f"Done · train={m['n_train']:,} · test={m['n_test']:,}")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("MAE", f"{m['MAE']:.4f}", f"{m['MAE']-m['Persistence_MAE']:+.4f} vs Pers.")
                    d2.metric("RMSE", f"{m['RMSE']:.4f}", f"{m['RMSE']-m['Persistence_RMSE']:+.4f} vs Pers.")
                    d3.metric("sMAPE", f"{m['sMAPE']:.2f}%")
                    d4.metric("PICP P10–P90", f"{m.get('PICP_P10_P90', float('nan'))*100:.1f}%")

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
