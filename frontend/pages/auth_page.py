import streamlit as st

from src.services.auth_service import authenticate_user, register_user
from frontend.components.navbar import render_navbar
from frontend.ui.kit import UI


def render_auth_page():
    """Render the login and registration screen."""
    render_navbar(
        "Your Study Partner",
        "Your Study Partner",
        "Smart answers from your notes.",
        "page-hero",
    )

    left_col, right_col = st.columns([0.95, 1.05], gap="large")

    with left_col:
        UI.html(
            '''
            <div class="auth-shell">
                <div class="auth-hero">Upload once. Search forever.</div>
                <div class="auth-subtitle">
                    Keep your PDFs, ask questions anytime, get real answers.
                </div>
            </div>
            ''',
        )

    with right_col:
        UI.html(
            '''
            <div class="auth-panel-head">
                <div class="auth-card-title">Be a Happy Student</div>
            </div>
            ''',
        )

        login_tab, register_tab = st.tabs(["Login", "Register"])

        with login_tab:
            with st.form("login_form", clear_on_submit=False):
                UI.html(
                    '<div class="auth-form-tip">Welcome back! Let\'s pick up where you left off.</div>',
                )
                login_email = st.text_input("Email address", placeholder="name@example.com")
                login_password = st.text_input("Password", type="password", placeholder="Enter your password")
                login_submit = st.form_submit_button("Sign in", use_container_width=True)

            if login_submit:
                result = authenticate_user(st.session_state.users, login_email, login_password)
                if not result["ok"]:
                    st.error(result["message"])
                else:
                    st.session_state.authenticated = True
                    st.session_state.current_user = result["user"]
                    st.success(result["message"])
                    st.rerun()

        with register_tab:
            with st.form("register_form", clear_on_submit=False):
                UI.html(
                    '<div class="auth-form-tip">Create your account. Keep all your notes in one place.</div>',
                )
                reg_name = st.text_input("Full name", placeholder="Your name")
                reg_email = st.text_input("Email address", placeholder="name@example.com")
                reg_password = st.text_input("Create password", type="password", placeholder="Set a strong password")
                reg_confirm = st.text_input("Confirm password", type="password", placeholder="Repeat your password")
                register_submit = st.form_submit_button("Create account", use_container_width=True)

            if register_submit:
                result = register_user(st.session_state.users, reg_name, reg_email, reg_password, reg_confirm)
                if not result["ok"]:
                    st.error(result["message"])
                else:
                    st.session_state.authenticated = True
                    st.session_state.current_user = result["user"]
                    st.success(result["message"])
                    st.rerun()