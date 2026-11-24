"""Modern design system for FoodShock Streamlit application."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {
    "bg": "#0F172A",
    "surface": "#1E293B",
    "surface2": "#334155",
    "border": "#475569",
    "text": "#E2E8F0",
    "muted": "#94A3B8",
    "primary": "#10B981",
    "primary_dim": "#059669",
    "amber": "#F59E0B",
    "coral": "#F43F5E",
    "sky": "#38BDF8",
    "violet": "#A78BFA",
    "chart": ["#10B981", "#38BDF8", "#F59E0B", "#A78BFA", "#F43F5E", "#14B8A6"],
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', 'Plus Jakarta Sans', system-ui, sans-serif;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
div[data-testid="stToolbar"] {{display: none;}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0B1220 0%, #0F172A 40%, #132033 100%);
    border-right: 1px solid {COLORS["surface2"]};
}}
section[data-testid="stSidebar"] * {{
    color: {COLORS["text"]};
}}

.hero {{
    background: linear-gradient(135deg, #064E3B 0%, #0F172A 45%, #1E293B 100%);
    border: 1px solid {COLORS["surface2"]};
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(16, 185, 129, 0.08);
    position: relative;
    overflow: hidden;
}}
.hero::after {{
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(16,185,129,0.18) 0%, transparent 70%);
    pointer-events: none;
}}
.hero h1 {{
    margin: 0;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.85rem;
    font-weight: 800;
    color: #ECFDF5 !important;
    letter-spacing: -0.02em;
}}
.hero p {{
    margin: 0.4rem 0 0 0;
    color: #A7F3D0 !important;
    font-size: 0.95rem;
    opacity: 0.95;
}}
.hero .badge-row {{
    margin-top: 0.85rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}}
.pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}
.pill-green {{ background: rgba(16,185,129,0.18); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.35); }}
.pill-sky {{ background: rgba(56,189,248,0.15); color: #7DD3FC; border: 1px solid rgba(56,189,248,0.3); }}
.pill-amber {{ background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); }}
.pill-rose {{ background: rgba(244,63,94,0.15); color: #FDA4AF; border: 1px solid rgba(244,63,94,0.3); }}

.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.85rem;
    margin-bottom: 1.25rem;
}}
@media (max-width: 900px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
.kpi-card {{
    background: linear-gradient(180deg, {COLORS["surface"]} 0%, #172033 100%);
    border: 1px solid {COLORS["surface2"]};
    border-radius: 14px;
    padding: 1.1rem 1.15rem;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    border-color: rgba(16,185,129,0.45);
    box-shadow: 0 8px 24px rgba(16,185,129,0.12);
}}
.kpi-card .label {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {COLORS["muted"]};
    margin-bottom: 0.35rem;
}}
.kpi-card .value {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: #ECFDF5;
    line-height: 1.15;
}}
.kpi-card .sub {{
    margin-top: 0.3rem;
    font-size: 0.75rem;
    color: {COLORS["muted"]};
}}

.section-title {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #ECFDF5;
    border-left: 3px solid {COLORS["primary"]};
    padding-left: 0.7rem;
    margin: 1.4rem 0 0.75rem 0;
}}

.panel {{
    background: rgba(30, 41, 59, 0.72);
    border: 1px solid {COLORS["surface2"]};
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}}

.status-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 180px;
    padding: 0.65rem 1.25rem;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.03em;
    text-align: center;
}}
.status-normal {{ background: rgba(16,185,129,0.2); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.45); }}
.status-watch {{ background: rgba(245,158,11,0.2); color: #FCD34D; border: 1px solid rgba(245,158,11,0.45); }}
.status-shock {{ background: rgba(244,63,94,0.2); color: #FDA4AF; border: 1px solid rgba(244,63,94,0.5); }}

.flow-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.6rem;
    margin: 0.75rem 0 1rem 0;
}}
@media (max-width: 900px) {{
    .flow-row {{ grid-template-columns: 1fr 1fr; }}
}}
.flow-step {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["surface2"]};
    border-radius: 12px;
    padding: 0.85rem 0.9rem;
    text-align: left;
}}
.flow-step .num {{
    font-size: 0.7rem;
    font-weight: 700;
    color: {COLORS["primary"]};
    letter-spacing: 0.06em;
}}
.flow-step .title {{
    font-weight: 700;
    font-size: 0.88rem;
    color: #F1F5F9;
    margin: 0.2rem 0;
}}
.flow-step .desc {{
    font-size: 0.72rem;
    color: {COLORS["muted"]};
    line-height: 1.35;
}}

.sidebar-brand {{
    text-align: center;
    padding: 0.6rem 0.4rem 1rem 0.4rem;
}}
.sidebar-brand .logo {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: #6EE7B7;
    letter-spacing: -0.02em;
}}
.sidebar-brand .tag {{
    font-size: 0.72rem;
    color: {COLORS["muted"]};
    margin-top: 0.15rem;
}}
.sidebar-stats {{
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 10px;
    padding: 0.7rem 0.85rem;
    font-size: 0.75rem;
    color: {COLORS["muted"]};
    line-height: 1.55;
    margin: 0.5rem 0 0.75rem 0;
}}
.sidebar-stats strong {{ color: #A7F3D0; }}

.footer-note {{
    font-size: 0.75rem;
    color: {COLORS["muted"]};
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid {COLORS["surface2"]};
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid {COLORS["surface2"]};
    border-radius: 10px;
    overflow: hidden;
}}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def apply_plot_style():
    sns.set_theme(style="darkgrid", palette=COLORS["chart"])
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "#1E293B",
        "savefig.facecolor": "none",
        "axes.edgecolor": "#475569",
        "axes.labelcolor": "#E2E8F0",
        "xtick.color": "#94A3B8",
        "ytick.color": "#94A3B8",
        "text.color": "#E2E8F0",
        "grid.color": "#334155",
        "grid.alpha": 0.6,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.facecolor": "#1E293B",
        "legend.edgecolor": "#475569",
        "legend.labelcolor": "#E2E8F0",
    })


def kpi_card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div>'
        f'</div>'
    )


def hero(title: str, subtitle: str, pills: list | None = None) -> str:
    pill_html = ""
    if pills:
        items = "".join(f'<span class="pill {cls}">{txt}</span>' for txt, cls in pills)
        pill_html = f'<div class="badge-row">{items}</div>'
    return (
        f'<div class="hero">'
        f'<h1>{title}</h1>'
        f'<p>{subtitle}</p>'
        f'{pill_html}'
        f'</div>'
    )


def section(title: str) -> str:
    return f'<p class="section-title">{title}</p>'


def status_badge(label: str, kind: str) -> str:
    cls = {"normal": "status-normal", "watch": "status-watch", "shock": "status-shock"}.get(kind, "status-normal")
    return f'<div class="status-badge {cls}">{label}</div>'
