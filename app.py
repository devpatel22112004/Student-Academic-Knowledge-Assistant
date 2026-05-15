import streamlit as st

from src.ui.pages.auth_page import render_auth_page
from src.ui.pages.workspace_page import render_workspace_page
from src.utils.css_loader import inject_custom_css
from src.utils.env import launch_streamlit_app, running_inside_streamlit
from src.utils.session import init_state


def main():
    """Thin Streamlit entrypoint that routes to auth or workspace screens."""
    st.set_page_config(
        page_title="Student Academic Assistant",
        page_icon="📚",
        layout="wide",
    )

    inject_custom_css()
    init_state()

    if not st.session_state.authenticated:
        render_auth_page()
        return

    render_workspace_page()


if __name__ == "__main__":
    if running_inside_streamlit():
        main()
    else:
        launch_streamlit_app()
