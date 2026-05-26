import streamlit as st
from src.services.knowledge_base_service import build_knowledge_base
from src.utils.session import reset_workspace_state
from src.utils.user_files import get_user_files
from frontend.ui.kit import UI


def _build_history_file_options(user_files):
    options = []
    option_map = {}

    for file_info in user_files:
        file_name = file_info.get("name", "Unknown")
        file_date = file_info.get("date", "Unknown date")
        file_hash = file_info.get("hash")

        if not file_hash:
            continue

        label = f"{file_name} | {file_date} | {file_hash[:8]}"
        options.append(label)
        option_map[label] = file_hash

    return options, option_map


def _sync_selected_files_with_history(user_files):
    available_hashes = [file_info.get("hash") for file_info in user_files if file_info.get("hash")]
    if not available_hashes:
        st.session_state.selected_file_hashes = []
        st.session_state.selected_file_labels = []
        return

    current_hashes = st.session_state.get("selected_file_hashes", [])
    if not current_hashes:
        st.session_state.selected_file_hashes = available_hashes
        st.session_state.selected_file_labels = []
        return

    filtered_hashes = [file_hash for file_hash in current_hashes if file_hash in available_hashes]
    st.session_state.selected_file_hashes = filtered_hashes or available_hashes

# Sidebar for account actions, uploads, and file history.
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
            user_id = st.session_state.current_user.get("email", "default") if st.session_state.current_user else "default"

            with st.spinner("Processing your files..."):
                kb = build_knowledge_base(files, user_id=user_id)

            if kb and kb.get("error") == "duplicate":
                st.session_state.kb = kb
                st.warning(f"⚠️ {kb['message']}")
                st.info(kb.get('details', 'These files are already in your knowledge base. You can ask questions about them!'))
                for dup_file in kb.get("duplicate_files", []):
                    st.markdown(f"  ✓ {dup_file['name']} (Already uploaded)")
                st.session_state.uploaded_names = [f.name for f in files]
                user_files = get_user_files(user_id)
                st.session_state.selected_file_hashes = [file_info.get("hash") for file_info in user_files if file_info.get("hash")]

            elif kb and kb.get("error") == "mixed":
                st.session_state.kb = kb
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
                user_files = get_user_files(user_id)
                st.session_state.selected_file_hashes = [file_info.get("hash") for file_info in user_files if file_info.get("hash")]

            elif kb is None:
                st.error("❌ Uploaded files had no readable text.")

            else:
                st.session_state.kb = kb
                st.session_state.uploaded_names = [f.name for f in files]
                user_files = get_user_files(user_id)
                st.session_state.selected_file_hashes = [file_info.get("hash") for file_info in user_files if file_info.get("hash")]
                vectors_count = kb.get("vectors_count", len(kb.get("chunks", [])))
                st.success(f"✅ {kb.get('message', 'Processed!')}")
                st.markdown(f"📊 Stored {vectors_count} chunks in knowledge base")

    if st.session_state.kb is not None:
        UI.html('<div class="status-pill">Ready</div>')

    if st.session_state.uploaded_names:
        st.markdown(" Your Uploaded Files")
        for name in st.session_state.uploaded_names:
            st.markdown(f"- {name}")
    
    if st.session_state.current_user:
        user_id = st.session_state.current_user.get("email", "default")
        user_files = get_user_files(user_id)
        _sync_selected_files_with_history(user_files)
        
        if user_files:
            st.divider()
            st.markdown("**📚 Your File History**")
            st.caption("Files you've previously uploaded")

            options, option_map = _build_history_file_options(user_files)
            hash_to_label = {file_hash: label for label, file_hash in option_map.items()}
            default_labels = [hash_to_label[file_hash] for file_hash in st.session_state.selected_file_hashes if file_hash in hash_to_label]

            current_labels = st.session_state.get("selected_file_labels", [])
            valid_labels = [label for label in current_labels if label in options] if current_labels else default_labels or options
            if valid_labels != current_labels:
                st.session_state.selected_file_labels = valid_labels

            chosen_labels = st.multiselect(
                "Select files to search",
                options=options,
                help="Queries will search only the selected files. Leave all selected for the broadest answer.",
                key="selected_file_labels",
            )

            selected_hashes = [option_map[label] for label in chosen_labels if label in option_map]
            st.session_state.selected_file_hashes = selected_hashes or [option_map[label] for label in options]
            
            for file_info in user_files:
                file_name = file_info.get("name", "Unknown")
                file_date = file_info.get("date", "Unknown date")
                st.markdown(f"✓ {file_name} ({file_date})")