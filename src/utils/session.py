import streamlit as st


def init_state():
    """Initialize all Streamlit session state keys used by the app."""
    if "kb" not in st.session_state:
        st.session_state.kb = None
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "uploaded_names" not in st.session_state:
        st.session_state.uploaded_names = []
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None


def reset_workspace_state():
    """Clear workspace-related session state on sign out."""
    st.session_state.kb = None
    st.session_state.chat = []
    st.session_state.uploaded_names = []
    st.session_state.authenticated = False
    st.session_state.current_user = None