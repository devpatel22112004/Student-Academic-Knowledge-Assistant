import streamlit as st

# This module manages the Streamlit session state for the knowledge assistant app. It includes functions to initialize the session state with default values and to reset workspace-related state when a user signs out. The session state keys include the knowledge base, chat history, uploaded document names, user information, and authentication status.
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

# This function resets the workspace-related session state keys to their default values, effectively clearing the knowledge base, chat history, uploaded document names, and user information. It also sets the authentication status to False. This is typically called when a user signs out to ensure a clean state for the next user.
def reset_workspace_state():
    """Clear workspace-related session state on sign out."""
    st.session_state.kb = None
    st.session_state.chat = []
    st.session_state.uploaded_names = []
    st.session_state.authenticated = False
    st.session_state.current_user = None