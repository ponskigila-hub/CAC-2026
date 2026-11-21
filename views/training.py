"""Interactive model training — residual vs direct."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_loader import (
    load_model_ready, load_model_results,
    load_eval_by_commodity, load_eval_by_pricetype, get_filter_options,
)
from models import available_algorithms, train_and_evaluate
from styles import inject_css, apply_plot_style, hero, section, COLORS


def render():
    inject_css()
    apply_plot_style()

    st.markdown(
        hero("Model Training & Evaluation",
             "Residual vs Direct forecasting · live hyperparameters · time-based validation",
             pills=[("Walk-Forward", "pill-sky"), ("vs Persistence", "pill-amber")]),
        unsafe_allow_html=True,
    )

    st.markdown(section("Official Benchmark (pre-computed)"), unsafe_allow_html=True)
    results = load_model_results()
    st.dataframe(
        results.sort_values("MAE").style.format(
            {"MAE": "{:.4f}", "RMSE": "{:.4f}", "sMAPE": "{:.2f}", "R2": "{:.4f}"}
        ),
        use_container_width=True, hide_index=True,
    )

    t1, t2 = st.tabs(["By Commodity", "By Price Type"])
    with t1:
        st.dataframe(load_eval_by_commodity(), use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(load_eval_by_pricetype(), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(section("Interactive Training Panel"), unsafe_allow_html=True)

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
                "Residual (Δy = yₜ₊₁ − yₜ)" if x == "residual"
                else "Direct (yₜ₊₁)"
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
                with st.spinner("Training (time-based split, last 18 months test)..."):
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
                    st.session_state["trained_model"] = out["model"]
                    st.session_state["trained_encoders"] = out["encoders"]
                    st.session_state["trained_features"] = out["feature_names"]
                    st.session_state["trained_mode"] = out["target_mode"]
                    st.session_state["trained_metrics"] = out["metrics"]

                    m = out["metrics"]
                    st.success(f"Done · train={m['n_train']:,} · test={m['n_test']:,}")

                    mae_delta = m["MAE"] - m["Persistence_MAE"]
                    rmse_delta = m["RMSE"] - m["Persistence_RMSE"]
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("MAE", f"{m['MAE']:.4f}", f"{mae_delta:+.4f} vs Pers.")
                    d2.metric("RMSE", f"{m['RMSE']:.4f}", f"{rmse_delta:+.4f} vs Pers.")
                    d3.metric("sMAPE", f"{m['sMAPE']:.2f}%")
                    d4.metric("R²", f"{m['R2']:.4f}")

                    comp = pd.DataFrame([
                        {"Model": m["model"], "MAE": m["MAE"], "RMSE": m["RMSE"],
                         "sMAPE": m["sMAPE"], "R2": m["R2"]},
                        {"Model": "Persistence (lag_1)", "MAE": m["Persistence_MAE"],
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
