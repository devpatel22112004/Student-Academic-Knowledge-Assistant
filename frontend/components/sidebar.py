import streamlit as st
from src.services.knowledge_base_service import build_knowledge_base
from src.utils.session import reset_workspace_state
from src.utils.user_files import get_user_files
from frontend.ui.kit import UI

# The sidebar component handles user account display and document upload functionality. It shows the logged-in user's name and email in a popover, allows them to sign out, and provides an interface to upload PDF or TXT files. Once files are uploaded and processed into a knowledge base, it displays the status and lists the uploaded file names.
def render_sidebar():
    if st.session_state.current_user:
        user_initial = "".join(part[0] for part in st.session_state.current_user["name"].split()[:2]).upper()
        if not user_initial:
            user_initial = st.session_state.current_user["name"][0].upper()

        with st.popover(user_initial):
            UI.html(
                f'''
                <div class="popover-account">
                    <div class="popover-account-label">Logged in account</div>
                    <div class="popover-account-name">{UI.escape(st.session_state.current_user["name"])}</div>
                    <div class="popover-account-mail">{UI.escape(st.session_state.current_user["email"])}</div>
                </div>
                ''',
            )

            if st.button("Sign out", use_container_width=True, key="sidebar_signout_button"):
                reset_workspace_state()
                st.rerun()

    UI.html('<div class="sidebar-section-title">Workspace</div>')
    UI.html(
        '<div class="sidebar-section-subtitle">Upload your PDFs here. Search them whenever you need.</div>',
    )
    UI.html('<div class="sidebar-divider"></div>')

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
            # Get user_id from logged-in user
            user_id = st.session_state.current_user.get("email", "default") if st.session_state.current_user else "default"
            
            with st.spinner("Processing your files..."):
                kb = build_knowledge_base(files, user_id=user_id)

            # Handle all scenarios
            if kb and kb.get("error") == "duplicate":
                # ❌ ALL files are duplicates
                st.session_state.kb = kb  # Set KB so user can ask questions!
                st.warning(f"⚠️ {kb['message']}")
                st.info(kb.get('details', 'These files are already in your knowledge base. You can ask questions about them!'))
                for dup_file in kb.get("duplicate_files", []):
                    st.markdown(f"  ✓ {dup_file['name']} (Already uploaded)")
                st.session_state.uploaded_names = [f.name for f in files]
                
            elif kb and kb.get("error") == "mixed":
                # ⚠️ MIXED: Some duplicates, some new
                st.session_state.kb = kb  # Set KB with new files
                st.warning(f"⚠️ {kb['message']}")
                st.info(kb.get('details', ''))
                
                if kb.get("duplicate_files"):
                    st.markdown("**Already in your knowledge base:**")
                    for dup_file in kb.get("duplicate_files", []):
                        st.markdown(f"  ✓ {dup_file['name']}")
                
                if kb.get("new_files"):
                    st.success(f"**✅ New files added ({len(kb.get('new_files', []))})**")
                    for new_file in kb.get("new_files", []):
                        st.markdown(f"  + {new_file}")
                
                st.session_state.uploaded_names = [f.name for f in files]
                
            elif kb is None:
                st.error("❌ Uploaded files had no readable text.")
                
            else:
                # ✅ ALL files are NEW - success!
                st.session_state.kb = kb
                st.session_state.uploaded_names = [f.name for f in files]
                vectors_count = kb.get("vectors_count", len(kb.get("chunks", [])))
                st.success(f"✅ {kb.get('message', 'Processed!')}")
                st.markdown(f"📊 Stored {vectors_count} chunks in knowledge base")

    if st.session_state.kb is not None:
        UI.html('<div class="status-pill">Ready</div>')

    if st.session_state.uploaded_names:
        st.markdown(" Your Uploaded Files")
        for name in st.session_state.uploaded_names:
            st.markdown(f"- {name}")
    
    # Display user's previous uploads history
    if st.session_state.current_user:
        user_id = st.session_state.current_user.get("email", "default")
        user_files = get_user_files(user_id)
        
        if user_files:
            st.divider()
            st.markdown("**📚 Your File History**")
            st.caption("Files you've previously uploaded")
            
            for file_info in user_files:
                file_name = file_info.get("name", "Unknown")
                file_date = file_info.get("date", "Unknown date")
                st.markdown(f"✓ {file_name} ({file_date})")