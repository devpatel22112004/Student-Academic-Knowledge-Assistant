import streamlit as st

from src.core.answer_generation import generate_extractive_answer
from src.core.retrieval import find_relevant_chunks
from src.services.gemini_service import generate_with_flash
from frontend.components.source_cards import prepare_source_items


def render_chat_panel(api_key):
    """Render the chat input, answer flow, and chat history."""
    st.markdown("Ask Away")

    question = st.chat_input("What do you want to know?")

    if question:
        if st.session_state.kb is None:
            st.warning("Please upload and process documents first.")
        else:
            kb = st.session_state.kb
            relevant = find_relevant_chunks(
                question,
                kb["index"],
                kb["chunks"],
                kb["model"],
                num_results=5,
            )

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
                st.markdown(
                    f'''
                    <div class="answer-shell">
                        <div class="answer-label">Answer</div>
                        <div class="answer-text">{item["answer"].replace(chr(10), "<br>")}</div>
                    </div>
                    ''' ,
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="source-wrap">', unsafe_allow_html=True)
                st.markdown('<div class="source-title">Sources used</div>', unsafe_allow_html=True)
                for src in item["sources"]:
                    st.markdown(f'<span class="source-pill">{src}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("View source details"):
                    for source_item in item.get("source_items", []):
                        st.markdown(
                            f'''
                            <div class="source-preview">
                                <div class="source-preview-title">{source_item["source"]}</div>
                                <div class="source-preview-text">{source_item["preview"]}</div>
                            </div>
                            ''',
                            unsafe_allow_html=True,
                        )
    else:
        st.info("Upload files, process them, and ask your first question.")