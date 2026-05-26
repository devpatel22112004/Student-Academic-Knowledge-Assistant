import streamlit as st
from src.core.answer_generation import generate_extractive_answer
from src.core.embeddings import get_embedding_model
from src.core.retrieval import find_relevant_chunks
from src.services.gemini_service import generate_with_flash
from frontend.components.source_cards import prepare_source_items
from frontend.ui.kit import UI

# Main Q&A panel for questions, answers, and chat history.
def render_chat_panel(api_key):
    st.markdown("Ask Away")

    question = st.chat_input("What do you want to know?")

    if question:
        if st.session_state.kb is None:
            st.warning("Please upload and process documents first.")
        else:
            kb = st.session_state.kb
            user_id = kb.get("user_id", "default")
            selected_file_hashes = st.session_state.get("selected_file_hashes", [])

            if kb.get("model") is None:
                with st.spinner("Preparing search model for your first question..."):
                    kb["model"] = get_embedding_model()
                    st.session_state.kb = kb
            
            relevant = find_relevant_chunks(
                question,
                kb["model"],
                num_results=5,
                user_id=user_id,
                file_hashes=selected_file_hashes or None
            )

            # Use Gemini when available, otherwise fall back to extractive answers.
            if api_key.strip():
                try:
                    with st.spinner("Generating answer..."):
                        answer_text = generate_with_flash(api_key.strip(), question, relevant)
                except Exception:
                    fallback, _ = generate_extractive_answer(question, relevant)
                    st.warning("AI model response unavailable right now. Showing grounded answer from your uploaded files.")
                    answer_text = fallback
            else:
                fallback, _ = generate_extractive_answer(question, relevant)
                answer_text = fallback

            st.session_state.chat.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "sources": sorted({chunk["source"] for chunk in relevant}),
                    "source_items": prepare_source_items(relevant),
                }
            )

    if st.session_state.chat:
        for item in reversed(st.session_state.chat):
            with st.chat_message("user"):
                st.markdown(item["question"])

            with st.chat_message("assistant"):
                UI.html(
                    f'''
                    <div class="answer-shell">
                        <div class="answer-label">Answer</div>
                        <div class="answer-text">{UI.nl2br(item["answer"])}</div>
                    </div>
                    ''' ,
                )

                UI.html('<div class="source-wrap">')
                UI.html('<div class="source-title">Sources used</div>')
                for src in item["sources"]:
                    UI.html(f'<span class="source-pill">{UI.escape(src)}</span>')
                UI.html('</div>')

                with st.expander("View source details"):
                    for source_item in item.get("source_items", []):
                        UI.html(
                            f'''
                            <div class="source-preview">
                                <div class="source-preview-title">{UI.escape(source_item["source"])}</div>
                                <div class="source-preview-text">{UI.escape(source_item["preview"])}</div>
                            </div>
                            ''',
                        )
    else:
        st.info("Upload files, process them, and ask your first question.")