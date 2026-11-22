"""Prediction & Early Shock Detection."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_loader import load_model_ready, get_filter_options
from models import predict_single, train_and_evaluate, available_algorithms
from styles import inject_css, hero, section, status_badge
from utils import classify_shock, pct_change_from_prices, FEATURE_COLS

EXAMPLE_PATH = Path(__file__).parent.parent / "user_analysis_example" / "contoh_bulk_test.csv"


def _ensure_default_model():
    if "trained_model" in st.session_state:
        return
    df = load_model_ready()
    algs = available_algorithms()
    alg = "LightGBM" if "LightGBM" in algs else algs[0]
    with st.spinner("Initialising default residual model (one-time)..."):
        out = train_and_evaluate(
            df, algorithm=alg, target_mode="residual",
            n_estimators=200, learning_rate=0.05, max_depth=6,
        )
    st.session_state["trained_model"] = out["model"]
    st.session_state["trained_encoders"] = out["encoders"]
    st.session_state["trained_features"] = out["feature_names"]
    st.session_state["trained_mode"] = out["target_mode"]
    st.session_state["trained_metrics"] = out["metrics"]


def _badge_kind(label: str) -> str:
    if "SHOCK" in label.upper():
        return "shock"
    if "WASPADA" in label.upper() or "MODERATE" in label.upper():
        return "watch"
    return "normal"


def render():
    inject_css()

    st.markdown(
        hero("Testing & Early Shock Detection",
             "What-if simulation · P10–P90 intervals · Stage-2 shock probability · alert badges",
             pills=[("t+1 Forecast", "pill-sky"), ("Early Warning", "pill-amber")]),
        unsafe_allow_html=True,
    )

    _ensure_default_model()
    model = st.session_state["trained_model"]
    encoders = st.session_state["trained_encoders"]
    feature_names = st.session_state["trained_features"]
    target_mode = st.session_state["trained_mode"]

    st.caption(f"Active mode: **{target_mode}** · {len(feature_names)} features")

    tab_single, tab_bulk = st.tabs(["Single Case / What-If", "Bulk CSV Testing"])

    with tab_single:
        df = load_model_ready()
        opts = get_filter_options(df)
        c1, c2, c3 = st.columns(3)
        with c1:
            country = st.selectbox("Country", opts["countries"], key="pred_country")
        with c2:
            commodity = st.selectbox("Commodity Group", opts["commodities"], key="pred_com")
        with c3:
            pricetype = st.selectbox("Price Type", opts["pricetypes"], key="pred_pt")

        ref = df[
            (df["countryiso3"] == country)
            & (df["commodity_group"] == commodity)
            & (df["pricetype"] == pricetype)
        ]
        defaults = {c: float(ref[c].median()) for c in FEATURE_COLS if c in ref.columns} if len(ref) else {}
        if "usdprice" not in defaults and len(ref):
            defaults["usdprice"] = float(ref["usdprice"].median())
        defaults.setdefault("usdprice", 1.0)
        for c in FEATURE_COLS:
            defaults.setdefault(c, defaults.get("lag_1", 1.0))

        st.markdown(section("Input Features"), unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            usdprice = st.number_input("Current price (USD)", value=defaults["usdprice"], format="%.4f")
            lag_1 = st.number_input("lag_1", value=defaults.get("lag_1", usdprice), format="%.4f")
            lag_2 = st.number_input("lag_2", value=defaults.get("lag_2", lag_1), format="%.4f")
        with r2:
            lag_3 = st.number_input("lag_3", value=defaults.get("lag_3", lag_1), format="%.4f")
            lag_6 = st.number_input("lag_6", value=defaults.get("lag_6", lag_1), format="%.4f")
            lag_12 = st.number_input("lag_12", value=defaults.get("lag_12", lag_1), format="%.4f")
        with r3:
            rolling_mean_6 = st.number_input("rolling_mean_6", value=defaults.get("rolling_mean_6", lag_1), format="%.4f")
            rolling_std_6 = st.number_input("rolling_std_6", value=max(defaults.get("rolling_std_6", 0.05), 0.001), format="%.4f")
            pct_change_1m = st.number_input("pct_change_1m", value=defaults.get("pct_change_1m", 0.0), format="%.4f")

        row = {
            "countryiso3": country, "commodity_group": commodity, "pricetype": pricetype,
            "usdprice": usdprice, "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3,
            "lag_6": lag_6, "lag_12": lag_12,
            "rolling_mean_3": defaults.get("rolling_mean_3", lag_1),
            "rolling_mean_6": rolling_mean_6, "rolling_std_6": rolling_std_6,
            "rolling_min_6": defaults.get("rolling_min_6", lag_1 * 0.95),
            "rolling_max_6": defaults.get("rolling_max_6", lag_1 * 1.05),
            "price_to_mean6": lag_1 / rolling_mean_6 if rolling_mean_6 else 1.0,
            "cv_6": rolling_std_6 / rolling_mean_6 if rolling_mean_6 else 0.0,
            "pct_change_1m": pct_change_1m,
            "pct_change_3m": defaults.get("pct_change_3m", 0.0),
            "pct_change_12m": defaults.get("pct_change_12m", 0.0),
            "month_sin": defaults.get("month_sin", 0.0),
            "month_cos": defaults.get("month_cos", 1.0),
        }

        if st.button("Predict t+1 Price", type="primary", key="btn_single"):
            pred = predict_single(model, encoders, feature_names, target_mode, row)
            pct = pct_change_from_prices(lag_1, pred)
            label, _ = classify_shock(pct)
            kind = _badge_kind(label)
            delta_usd = pred - lag_1

            st.markdown(section("Prediction Result"), unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Price t+1", f"USD {pred:.4f}", f"{delta_usd:+.4f} USD")
            m2.metric("% Change vs lag_1", f"{pct*100:+.2f}%")
            with m3:
                st.markdown(status_badge(label, kind), unsafe_allow_html=True)

            st.markdown(
                """
                | Status | Operational rule (demo) |
                |--------|-------------------------|
                | **NORMAL** | Predicted %Δ < 3% and low Stage-2 risk |
                | **WASPADA / ELEVATED RISK** | 3–5% move or moderate shock probability |
                | **FOOD SHOCK ALERT** | %Δ > 5% (policy ALPS/AFSIS) or high Stage-2 probability (opt. thr. ≈ 0.35) |

                Full production alerts combine **point forecast**, **P90 upside bound**, and **calibrated Stage-2 probability**.
                """
            )

    with tab_bulk:
        st.markdown(section("Bulk CSV Testing"), unsafe_allow_html=True)
        if EXAMPLE_PATH.exists():
            with open(EXAMPLE_PATH, "rb") as f:
                st.download_button(
                    "Download example bulk test CSV (25 rows)",
                    data=f.read(), file_name="contoh_bulk_test.csv", mime="text/csv",
                )

        uploaded = st.file_uploader("Upload CSV for bulk prediction", type=["csv"])
        if uploaded is not None:
            try:
                bulk = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Cannot read CSV: {e}")
                return

            st.write(f"Uploaded: **{bulk.shape[0]}** rows × **{bulk.shape[1]}** columns")
            st.dataframe(bulk.head(8), use_container_width=True)

            if "lag_1" not in bulk.columns:
                st.error("CSV must contain at least column: lag_1")
                return

            preds, pcts, labels = [], [], []
            for _, r in bulk.iterrows():
                row = r.to_dict()
                for c in FEATURE_COLS:
                    if c not in row or pd.isna(row.get(c)):
                        row[c] = row.get("lag_1", 0)
                for c in ["countryiso3", "commodity_group", "pricetype"]:
                    if c not in row or pd.isna(row.get(c)):
                        row[c] = "Unknown"
                try:
                    p = predict_single(model, encoders, feature_names, target_mode, row)
                except Exception:
                    p = float(row.get("lag_1", 0))
                pct = pct_change_from_prices(row.get("lag_1", p), p)
                lab, _ = classify_shock(pct)
                preds.append(p)
                pcts.append(pct * 100)
                labels.append(lab)

            bulk = bulk.copy()
            bulk["pred_price_t1"] = preds
            bulk["pred_pct_change"] = pcts
            bulk["shock_status"] = labels

            st.markdown(section("Bulk Results"), unsafe_allow_html=True)
            st.dataframe(bulk, use_container_width=True)
            st.bar_chart(bulk["shock_status"].value_counts())
            st.download_button(
                "Download predictions CSV",
                data=bulk.to_csv(index=False).encode("utf-8"),
                file_name="foodshock_bulk_predictions.csv",
                mime="text/csv",
            )

    st.markdown('<p class="footer-note">FoodShock · MLSC 2026 · SDG 2 — Zero Hunger</p>', unsafe_allow_html=True)
