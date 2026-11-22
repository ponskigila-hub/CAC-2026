"""
FoodShock — Streamlit Interactive Web Application
Cross-Country ASEAN Food Price Forecasting & Early Shock Detection System
MLSC 2026 · SDG 2 — Zero Hunger
"""
from __future__ import annotations

import streamlit as st

from styles import inject_css
from views import home, eda, preprocessing, training, prediction

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
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="logo">🌾 FoodShock</div>
            <div class="tag">MLSC 2026 · SDG 2: Zero Hunger</div>
        </div>
        <div class="sidebar-stats">
            <strong>7</strong> ASEAN countries<br/>
            <strong>5</strong> commodity groups<br/>
            <strong>27,950</strong> clean observations<br/>
            Residual forecasting · Zero leakage
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
