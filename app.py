"""
FoodShock — Streamlit Interactive Web Application
Cross-Country ASEAN Food Price Forecasting & Early Shock Detection System
MLSC 2026 · SDG 2 — Zero Hunger
"""
from __future__ import annotations

import streamlit as st

from application.styles import inject_css
from application.views import home, eda, preprocessing, training, prediction
from application.data_loader import load_model_ready

st.set_page_config(
    page_title="FoodShock · MLSC 2026",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

PAGES = {
    "🏠  Home": home,
    "📊  EDA": eda,
    "🔧  Preprocessing": preprocessing,
    "⚙️  Model Training": training,
    "🔮  Prediction & Alert": prediction,
}

with st.sidebar:
    try:
        df=load_model_ready()
    except Exception as exc:
        st.error(f"V9 data integrity check failed: {exc}. Run python src/run_v9.py and restart the app.")
        st.stop()
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="logo">🌾 FoodShock</div>
            <div class="tag">MLSC 2026 · SDG 2: Zero Hunger</div>
        </div>
        <div class="sidebar-stats">
            <strong>{df.countryiso3.nunique()}</strong> ASEAN countries<br/>
            <strong>{df.commodity_group.nunique()}</strong> commodity groups<br/>
            <strong>{len(df):,}</strong> model-ready observations<br/>
            V9 · Temporal validation
        </div>
        """,
        unsafe_allow_html=True,
    )
    selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.72rem; color:#64748B; line-height:1.5; padding:0 0.2rem;">
        Early Food Price Shock detection<br/>
        ASEAN panel · Retail &amp; Wholesale<br/>
        <span style="color:#10B981;">●</span> Interactive analytics
        </div>
        """,
        unsafe_allow_html=True,
    )

PAGES[selection].render()
