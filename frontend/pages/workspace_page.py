import streamlit as st

from src.services.gemini_service import get_configured_api_key
from src.services.knowledge_base_service import build_knowledge_base
from frontend.components.chat_panel import render_chat_panel
from frontend.components.navbar import render_navbar
from frontend.components.sidebar import render_sidebar


def render_workspace_page():
    """Render the main workspace where files are uploaded and questions are asked."""
    
    # Auto-load KB on page init if user has files and KB not yet loaded
    if st.session_state.kb is None and st.session_state.current_user:
        user_id = st.session_state.current_user.get("email", "default")
        try:
            # Prepare knowledge base for querying without uploading files
            kb = build_knowledge_base(None, user_id=user_id)
            if kb:
                st.session_state.kb = kb
                st.session_state.chat = []
                st.session_state.uploaded_names = []
        except Exception as e:
            print(f"Note: Could not auto-load KB on init: {e}")
    
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