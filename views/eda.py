"""Interactive Exploratory Data Analysis."""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from data_loader import load_model_ready, get_filter_options
from styles import inject_css, apply_plot_style, hero, section, COLORS


def render():
    inject_css()
    apply_plot_style()

    st.markdown(
        hero("Exploratory Data Analysis", "Trends · volatility · Retail vs Wholesale",
             pills=[("Interactive Filters", "pill-sky"), ("Panel Data", "pill-green")]),
        unsafe_allow_html=True,
    )

    df = load_model_ready()
    opts = get_filter_options(df)

    with st.container():
        st.markdown(section("Control Panel"), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            countries = st.multiselect("Country", opts["countries"], default=opts["countries"][:4])
        with c2:
            commodities = st.multiselect("Commodity Group", opts["commodities"], default=opts["commodities"])
        with c3:
            pricetypes = st.multiselect("Price Type", opts["pricetypes"], default=opts["pricetypes"])

    mask = (
        df["countryiso3"].isin(countries if countries else opts["countries"])
        & df["commodity_group"].isin(commodities if commodities else opts["commodities"])
        & df["pricetype"].isin(pricetypes if pricetypes else opts["pricetypes"])
    )
    sub = df.loc[mask].copy()
    st.caption(f"Filtered rows: **{len(sub):,}** / {len(df):,}")

    if sub.empty:
        st.warning("No data matches the selected filters.")
        return

    st.markdown(section("Price Trend (USD)"), unsafe_allow_html=True)
    trend = sub.groupby(["date", "commodity_group"])["usdprice"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.0))
    for i, grp in enumerate(trend["commodity_group"].unique()):
        g = trend[trend["commodity_group"] == grp]
        ax.plot(g["date"], g["usdprice"], label=grp, lw=1.8, color=COLORS["chart"][i % len(COLORS["chart"])])
    ax.set_ylabel("Mean USD Price")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title("Average Price by Commodity Group")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.markdown(section("Volatility & Distributions"), unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        vals = sub["pct_change_1m"].dropna().clip(-0.5, 0.5)
        ax.hist(vals, bins=48, color=COLORS["primary"], edgecolor="#0F172A", alpha=0.9)
        ax.axvline(0.05, color=COLORS["coral"], ls="--", lw=1.3, label="+5% threshold")
        ax.axvline(-0.05, color=COLORS["sky"], ls="--", lw=1.3, label="−5%")
        ax.set_xlabel("pct_change_1m")
        ax.set_title("1-Month % Change Distribution")
        ax.legend(fontsize=7)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with col_b:
        vol = sub.groupby("countryiso3")["pct_change_1m"].std().sort_values(ascending=True).dropna()
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.barh(vol.index, vol.values, color=COLORS["primary"])
        ax.set_xlabel("Std of pct_change_1m")
        ax.set_title("Volatility by Country")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    st.markdown(section("Retail vs Wholesale"), unsafe_allow_html=True)
    if sub["pricetype"].nunique() > 1:
        comp = sub.groupby(["commodity_group", "pricetype"])["usdprice"].mean().unstack(fill_value=0)
        st.dataframe(comp.style.format("{:.3f}"), use_container_width=True)
    else:
        st.info("Select both Retail and Wholesale to compare market types.")

    st.markdown(section("Summary Statistics (USD)"), unsafe_allow_html=True)
    summary = sub.groupby("commodity_group")["usdprice"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    ).round(4)
    st.dataframe(summary, use_container_width=True)

    st.markdown('<p class="footer-note">FoodShock · MLSC 2026 · SDG 2 — Zero Hunger</p>', unsafe_allow_html=True)
