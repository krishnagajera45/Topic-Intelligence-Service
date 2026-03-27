"""
Topic Modeling System — Entry Point.

This is the landing page that redirects to Project Overview on first visit.
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config as _load_cfg
_cfg = _load_cfg()
_ds_title = _cfg.active_dataset.replace("_", " ").title()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{_ds_title} Topic Modeling",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Redirect to Project Overview ─────────────────────────────────────────────
st.switch_page("pages/0_Project_Overview.py")

