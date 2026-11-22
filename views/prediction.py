"""Prediction & Early Shock Detection — two-stage V2."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_loader import load_model_ready, get_filter_options
from models import predict_single, predict_two_stage, train_and_evaluate, available_algorithms
from styles import inject_css, hero, section, status_badge
from utils import (
    classify_shock, pct_change_from_prices, FEATURE_COLS,
    SHOCK_PROB_THRESHOLD, ensure_feature_row,
)

EXAMPLE_PATH = Path(__file__).parent.parent / "user_analysis_example" / "contoh_bulk_test.csv"


def _ensure_default_model():
    if "two_stage_bundle" in st.session_state and st.session_state["two_stage_bundle"].get("model") is not None:
        return
    if "trained_model" in st.session_state and st.session_state.get("shock_clf") is not None:
        st.session_state["two_stage_bundle"] = {
            "model": st.session_state["trained_model"],
            "q10_model": st.session_state.get("q10_model"),
            "q90_model": st.session_state.get("q90_model"),
            "shock_clf": st.session_state.get("shock_clf"),
            "encoders": st.session_state["trained_encoders"],
            "feature_names": st.session_state["trained_features"],
            "target_mode": st.session_state.get("trained_mode", "residual"),
        }
        return

    df = load_model_ready()
    algs = available_algorithms()
    alg = "LightGBM" if "LightGBM" in algs else algs[0]
    with st.spinner("Initialising two-stage residual model (point + P10/P90 + shock classifier)..."):
        out = train_and_evaluate(
            df, algorithm=alg, target_mode="residual",
            n_estimators=200, learning_rate=0.05, max_depth=6,
        )
    st.session_state["trained_model"] = out["model"]
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


def _badge_kind(label: str) -> str:
    u = label.upper()
    if "SHOCK" in u:
        return "shock"
    if "WASPADA" in u or "ELEVATED" in u or "MODERATE" in u:
        return "watch"
    return "normal"


def render():
    inject_css()

    st.markdown(
        hero(
            "Testing & Early Shock Detection",
            "What-if simulation · P10–P90 intervals · Stage-2 shock probability · alert badges",
            pills=[("Two-Stage V2", "pill-sky"), ("P(Shock) ≥ 0.35", "pill-amber")],
        ),
        unsafe_allow_html=True,
    )

    _ensure_default_model()
    bundle = st.session_state["two_stage_bundle"]
    st.caption(
        f"Active mode: **{bundle.get('target_mode', 'residual')}** · "
        f"{len(bundle.get('feature_names', []))} features · "
        f"Stage-2 threshold = {SHOCK_PROB_THRESHOLD}"
    )

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
        defaults = {}
        if len(ref):
            for c in FEATURE_COLS:
                if c in ref.columns:
                    defaults[c] = float(ref[c].median())
            defaults["usdprice"] = float(ref["usdprice"].median())
        defaults.setdefault("usdprice", 1.0)
        for c in FEATURE_COLS:
            defaults.setdefault(c, defaults.get("usdprice", 1.0) if c not in (
                "fx_pct_1m", "fx_pct_3m", "fx_vol_6m", "asean_comm_mom_1m",
                "country_vs_asean_mom", "pct_change_1m", "pct_change_3m",
                "pct_change_12m", "month_sin", "month_cos", "cv_6",
            ) else 0.0)

        st.markdown(section("Input Features"), unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            usdprice = st.number_input("Current price y_t (USD)", value=defaults["usdprice"], format="%.4f")
            lag_1 = st.number_input("lag_1", value=defaults.get("lag_1", usdprice), format="%.4f")
            lag_2 = st.number_input("lag_2", value=defaults.get("lag_2", lag_1), format="%.4f")
        with r2:
            lag_3 = st.number_input("lag_3", value=defaults.get("lag_3", lag_1), format="%.4f")
            lag_6 = st.number_input("lag_6", value=defaults.get("lag_6", lag_1), format="%.4f")
            lag_12 = st.number_input("lag_12", value=defaults.get("lag_12", lag_1), format="%.4f")
        with r3:
            rolling_mean_6 = st.number_input("rolling_mean_6", value=defaults.get("rolling_mean_6", lag_1), format="%.4f")
            fx_pct_1m = st.number_input("fx_pct_1m (macro)", value=float(defaults.get("fx_pct_1m", 0.0)), format="%.4f")
            asean_mom = st.number_input("asean_comm_mom_1m", value=float(defaults.get("asean_comm_mom_1m", 0.0)), format="%.4f")

        row = {
            "countryiso3": country, "commodity_group": commodity, "pricetype": pricetype,
            "usdprice": usdprice, "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3,
            "lag_6": lag_6, "lag_12": lag_12,
            "rolling_mean_3": defaults.get("rolling_mean_3", lag_1),
            "rolling_mean_6": rolling_mean_6,
            "rolling_std_6": defaults.get("rolling_std_6", 0.05),
            "rolling_min_6": defaults.get("rolling_min_6", lag_1 * 0.95),
            "rolling_max_6": defaults.get("rolling_max_6", lag_1 * 1.05),
            "price_to_mean6": lag_1 / rolling_mean_6 if rolling_mean_6 else 1.0,
            "cv_6": defaults.get("cv_6", 0.0),
            "pct_change_1m": defaults.get("pct_change_1m", 0.0),
            "pct_change_3m": defaults.get("pct_change_3m", 0.0),
            "pct_change_12m": defaults.get("pct_change_12m", 0.0),
            "month_sin": defaults.get("month_sin", 0.0),
            "month_cos": defaults.get("month_cos", 1.0),
            "fx_pct_1m": fx_pct_1m,
            "fx_pct_3m": defaults.get("fx_pct_3m", 0.0),
            "fx_vol_6m": defaults.get("fx_vol_6m", 0.0),
            "asean_comm_mom_1m": asean_mom,
            "country_vs_asean_mom": defaults.get("country_vs_asean_mom", 0.0),
        }

        if st.button("Predict t+1 (Two-Stage)", type="primary", key="btn_single"):
            out = predict_two_stage(bundle, row)
            pct = pct_change_from_prices(out["y_t"], out["y_hat"])
            label, _ = classify_shock(pct, out["shock_proba"])
            kind = _badge_kind(label)
            delta_usd = out["y_hat"] - out["y_t"]

            st.markdown(section("Prediction Result"), unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted ŷₜ₊₁", f"USD {out['y_hat']:.4f}", f"{delta_usd:+.4f}")
            m2.metric("% Change", f"{pct*100:+.2f}%")
            m3.metric("P(Shock)", f"{out['shock_proba']*100:.1f}%")
            with m4:
                st.markdown(status_badge(label, kind), unsafe_allow_html=True)

            st.markdown(
                f"**Uncertainty interval [P10, P90]:** "
                f"USD **{out['y_p10']:.4f}** — **{out['y_p90']:.4f}**"
            )
            st.markdown(
                """
                | Status | Rule |
                |--------|------|
                | **NORMAL** | mild %Δ and low Stage-2 probability |
                | **WASPADA / ELEVATED RISK** | ~3–5% move or moderate P(Shock) |
                | **FOOD SHOCK ALERT** | P(Shock) ≥ **0.35** or |%Δ| > **+5%** (ALPS/AFSIS) |
                """
            )

    with tab_bulk:
        st.markdown(section("Bulk CSV Testing"), unsafe_allow_html=True)
        if EXAMPLE_PATH.exists():
            with open(EXAMPLE_PATH, "rb") as f:
                st.download_button(
                    "Download example bulk test CSV",
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

            preds, p10s, p90s, pcts, probas, labels = [], [], [], [], [], []
            for _, r in bulk.iterrows():
                row = ensure_feature_row(r.to_dict())
                try:
                    out = predict_two_stage(bundle, row)
                except Exception:
                    yt = float(row.get("usdprice", row.get("lag_1", 0)))
                    out = {"y_hat": yt, "y_p10": yt, "y_p90": yt, "shock_proba": 0.0, "y_t": yt}
                pct = pct_change_from_prices(out["y_t"], out["y_hat"])
                lab, _ = classify_shock(pct, out["shock_proba"])
                preds.append(out["y_hat"])
                p10s.append(out["y_p10"])
                p90s.append(out["y_p90"])
                pcts.append(pct * 100)
                probas.append(out["shock_proba"])
                labels.append(lab)

            bulk = bulk.copy()
            bulk["pred_price_t1"] = preds
            bulk["pred_p10"] = p10s
            bulk["pred_p90"] = p90s
            bulk["pred_pct_change"] = pcts
            bulk["shock_proba"] = probas
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
