from pathlib import Path

import streamlit as st


def inject_custom_css():
    """Load the shared CSS file into the Streamlit app."""
    css_path = Path(__file__).resolve().parents[2] / "styles" / "app.css"
    if not css_path.exists():
        return

    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)