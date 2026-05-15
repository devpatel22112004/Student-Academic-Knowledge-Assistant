import streamlit as st

from src.services.knowledge_base_service import build_knowledge_base
from src.utils.session import reset_workspace_state


def render_sidebar():
    """Render the workspace sidebar with account info, upload, and process actions."""
    if st.session_state.current_user:
        user_initial = "".join(part[0] for part in st.session_state.current_user["name"].split()[:2]).upper()
        if not user_initial:
            user_initial = st.session_state.current_user["name"][0].upper()

        with st.popover(user_initial):
            st.markdown(
                f'''
                <div class="popover-account">
                    <div class="popover-account-label">Logged in account</div>
                    <div class="popover-account-name">{st.session_state.current_user["name"]}</div>
                    <div class="popover-account-mail">{st.session_state.current_user["email"]}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

            if st.button("Sign out", use_container_width=True, key="sidebar_signout_button"):
                reset_workspace_state()
                st.rerun()

    st.markdown('<div class="sidebar-section-title">Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-section-subtitle">Upload your PDFs here. Search them whenever you need.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    files = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload as many files as you want.",
    )

    if st.button("Process Documents", use_container_width=True):
        if not files:
            st.warning("Please upload at least one PDF or TXT file.")
        else:
            with st.spinner("Processing your files..."):
                kb = build_knowledge_base(files)

            if kb is None:
                st.error("Uploaded files had no readable text.")
            else:
                st.session_state.kb = kb
                st.session_state.uploaded_names = [f.name for f in files]
                st.success("All set! Now you can ask your questions.")

    if st.session_state.kb is not None:
        st.markdown('<div class="status-pill">Ready</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_names:
        st.markdown(" Your Uploaded Files")
        for name in st.session_state.uploaded_names:
            st.markdown(f"- {name}")