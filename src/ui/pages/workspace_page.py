import streamlit as st

from src.services.gemini_service import get_configured_api_key
from src.ui.components.chat_panel import render_chat_panel
from src.ui.components.navbar import render_navbar
from src.ui.components.sidebar import render_sidebar


def render_workspace_page():
    """Render the main workspace where files are uploaded and questions are asked."""
    render_navbar(
        "Your workspace",
        "Study Friend",
        "Upload PDFs. Ask questions. Get answers from your notes.",
        "workspace-hero",
    )

    api_key = get_configured_api_key().strip()

    with st.sidebar:
        render_sidebar()

    render_chat_panel(api_key)