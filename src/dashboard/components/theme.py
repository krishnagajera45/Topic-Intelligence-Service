"""
Shared theme, CSS, and reusable UI components for the Streamlit dashboard.

Provides a consistent, polished look across all pages.
"""
import streamlit as st


# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6C5CE7",
    "primary_light": "#A29BFE",
    "secondary": "#00CEC9",
    "accent": "#FD79A8",
    "success": "#00B894",
    "warning": "#FDCB6E",
    "danger": "#E17055",
    "dark": "#2D3436",
    "dark_bg": "#0E1117",
    "card_bg": "#1A1D23",
    "card_border": "#2D3142",
    "text": "#DFE6E9",
    "text_muted": "#B2BEC3",
    "gradient_start": "#6C5CE7",
    "gradient_end": "#00CEC9",
}


def inject_custom_css():
    """Inject custom CSS for a polished, modern dark-theme look."""
    st.markdown("""
    <style>
    /* ── Global ─────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1D23 0%, #0E1117 100%);
        border-right: 1px solid #2D3142;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #A29BFE;
    }

    /* ── Cards ──────────────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1A1D23 0%, #22262E 100%);
        border: 1px solid #2D3142;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108,92,231,0.15);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C5CE7, #00CEC9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }
    .metric-card .metric-label {
        font-size: 0.82rem;
        color: #B2BEC3;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    .metric-card .metric-icon {
        font-size: 1.6rem;
        margin-bottom: 0.4rem;
    }

    /* ── Section Headers ────────────────────────────────────────── */
    .section-header {
        font-size: 1.35rem;
        font-weight: 600;
        color: #DFE6E9;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #6C5CE7;
        display: inline-block;
    }

    /* ── Page Hero ──────────────────────────────────────────────── */
    .page-hero {
        background: linear-gradient(135deg, rgba(108,92,231,0.12) 0%, rgba(0,206,201,0.08) 100%);
        border: 1px solid rgba(108,92,231,0.25);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .page-hero h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        background: linear-gradient(135deg, #6C5CE7, #00CEC9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .page-hero p {
        color: #B2BEC3;
        font-size: 1rem;
        margin: 0;
    }

    /* ── Status Badges ──────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-high   { background: rgba(225,112,85,0.2); color: #E17055; border: 1px solid rgba(225,112,85,0.4); }
    .badge-medium { background: rgba(253,203,110,0.2); color: #FDCB6E; border: 1px solid rgba(253,203,110,0.4); }
    .badge-low    { background: rgba(0,184,148,0.2); color: #00B894; border: 1px solid rgba(0,184,148,0.4); }
    .badge-success{ background: rgba(0,184,148,0.2); color: #00B894; border: 1px solid rgba(0,184,148,0.4); }
    .badge-info   { background: rgba(108,92,231,0.2); color: #A29BFE; border: 1px solid rgba(108,92,231,0.4); }

    /* ── Info Cards ─────────────────────────────────────────────── */
    .info-card {
        background: linear-gradient(135deg, #1A1D23, #22262E);
        border: 1px solid #2D3142;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-card h3 {
        color: #A29BFE;
        margin-top: 0;
        font-size: 1.1rem;
    }
    .info-card p, .info-card li {
        color: #B2BEC3;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* ── Tabs ───────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #1A1D23;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6C5CE7, #00CEC9) !important;
    }

    /* ── Expander tweaks ────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 500;
        font-size: 0.95rem;
    }

    /* ── DataFrames ─────────────────────────────────────────────── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Footer ─────────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        color: #636E72;
        font-size: 0.78rem;
        border-top: 1px solid #2D3142;
        margin-top: 3rem;
    }
    .app-footer a { color: #6C5CE7; text-decoration: none; }
    .app-footer a:hover { color: #A29BFE; }

    /* ── Timeline ───────────────────────────────────────────────── */
    .timeline-item {
        border-left: 3px solid #6C5CE7;
        padding: 0.75rem 1.25rem;
        margin-bottom: 0.5rem;
        background: rgba(108,92,231,0.05);
        border-radius: 0 8px 8px 0;
    }
    .timeline-item .tl-title {
        font-weight: 600;
        color: #DFE6E9;
    }
    .timeline-item .tl-desc {
        color: #B2BEC3;
        font-size: 0.88rem;
    }

    /* ── Comparison Table ───────────────────────────────────────── */
    .compare-table {
        width: 100%;
        border-collapse: collapse;
    }
    .compare-table th {
        background: rgba(108,92,231,0.15);
        color: #A29BFE;
        padding: 0.6rem 1rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .compare-table td {
        padding: 0.5rem 1rem;
        border-bottom: 1px solid #2D3142;
        color: #DFE6E9;
        font-size: 0.88rem;
    }
    .compare-table tr:hover td {
        background: rgba(108,92,231,0.05);
    }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, description: str = "", icon: str = ""):
    """Render a hero-style page header."""
    st.markdown(f"""
    <div class="page-hero">
        <h1>{icon} {title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(icon: str, value, label: str):
    """Render a styled metric card with icon, value, and label."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def status_badge(text: str, level: str = "info"):
    """Render a colored status badge. Levels: high, medium, low, success, info."""
    return f'<span class="badge badge-{level}">{text}</span>'


def section_header(text: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def info_card(title: str, content: str):
    """Render an info card with title and HTML content."""
    st.markdown(f"""
    <div class="info-card">
        <h3>{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)


def render_footer(dataset_label: str = ""):
    """Render the app footer."""
    _label = dataset_label or "Online Topic Modeling System"
    st.markdown(f"""
    <div class="app-footer">
        <strong>{_label}</strong> v2.0 &nbsp;·&nbsp;
        Powered by <a href="https://maartengr.github.io/BERTopic/" target="_blank">BERTopic</a>,
        <a href="https://streamlit.io" target="_blank">Streamlit</a> &
        <a href="https://fastapi.tiangolo.com" target="_blank">FastAPI</a>
        <br/>COMP-EE 798 — Final Year Project
    </div>
    """, unsafe_allow_html=True)
