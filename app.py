import io
import os
import sys
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from main import (
    build_search_index,
    chunk_text,
    create_embeddings,
    find_relevant_chunks,
    generate_extractive_answer,
)


def inject_custom_css():
    # STEP 1: CUSTOM UI THEME APPLY KARNA
    # Yeh CSS Streamlit ka default look हटाकर Notebook-style interface banata hai.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --bg-main: linear-gradient(180deg, #f7fbff 0%, #eef5ff 45%, #f7f9fc 100%);
            --text-main: #11233f;
            --muted: #53627b;
            --accent: #116a7b;
            --border: rgba(17, 35, 63, 0.14);
        }

        .stApp {
            background: var(--bg-main);
            color: var(--text-main);
            font-family: 'IBM Plex Sans', sans-serif;
        }

        h1, h2, h3, [data-testid="stSidebar"] h2 {
            font-family: 'Outfit', sans-serif !important;
            letter-spacing: 0.1px;
        }

        .clean-card {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 16px rgba(11, 24, 45, 0.05);
        }

        .muted {
            color: var(--muted);
            font-size: 0.93rem;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.78);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] .stButton > button,
        .stButton > button {
            border-radius: 10px;
            border: 1px solid var(--border);
        }

        .source-box {
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.45rem 0.6rem;
            margin-top: 0.35rem;
            background: rgba(255, 255, 255, 0.76);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_uploaded_documents(uploaded_files):
    """
    STEP 2: USER KI UPLOADED FILES KO READ KARNA
    Output format: [(source_label, text_content), ...]
    Example: ("notes.pdf - Page 1", "Binary search is...")
    """
    all_documents = []
    for uploaded in uploaded_files:
        name = uploaded.name
        suffix = os.path.splitext(name)[1].lower()

        if suffix == ".pdf":
            # PDF ko memory buffer se read karke page-wise text extract karte hain.
            pdf_reader = PdfReader(io.BytesIO(uploaded.getvalue()))
            for i, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_documents.append((f"{name} - Page {i}", page_text))
        elif suffix == ".txt":
            # TXT file ko direct decode karke text list me add karte hain.
            text = uploaded.getvalue().decode("utf-8", errors="ignore")
            if text.strip():
                all_documents.append((name, text))

    return all_documents


def build_knowledge_base(uploaded_files):
    """
    STEP 3: IN-MEMORY KNOWLEDGE BASE BANANA (RAG PIPELINE)
    Flow: documents -> chunks -> embeddings -> FAISS index
    """
    documents = read_uploaded_documents(uploaded_files)
    if not documents:
        return None

    # Large text ko small chunks me todte hain.
    chunks = chunk_text(documents)
    # Har chunk ka vector embedding banate hain.
    embeddings, model = create_embeddings(chunks)
    # Fast similarity search ke liye FAISS index banate hain.
    index = build_search_index(embeddings)

    return {
        "chunks": chunks,
        "model": model,
        "index": index,
    }


def generate_with_flash(api_key, question, relevant_chunks):
    """
    STEP 4: GEMINI FLASH SE FINAL ANSWER GENERATE KARNA
    Yeh function retrieved context ko prompt me bhejta hai aur grounded answer mangta hai.
    """
    import google.generativeai as genai

    # Provide strict context so output remains grounded in uploaded notes.
    context = "\n\n".join(
        [f"Source: {chunk['source']}\nText: {chunk['text']}" for chunk in relevant_chunks]
    )

    prompt = f"""
You are an academic assistant. Answer only from the given context.

Rules:
1) If context has the answer, respond clearly in short bullet points.
2) If answer is missing, say: "I could not find this in your uploaded documents."
3) Do not add internet knowledge.

Question:
{question}

Context:
{context}
"""

    # API key configure karne ke baad hi model calls possible hoti hain.
    genai.configure(api_key=api_key)

    # Pick a currently supported Flash model from the account/project.
    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-1.5-flash",
    ]

    available = []
    for model_info in genai.list_models():
        methods = getattr(model_info, "supported_generation_methods", [])
        if "generateContent" in methods:
            available.append(model_info.name)

    model_name = None
    for candidate in preferred:
        if candidate in available:
            model_name = candidate
            break

    if model_name is None:
        flash_models = [m for m in available if "flash" in m.lower()]
        if flash_models:
            model_name = flash_models[0]

    if model_name is None:
        raise RuntimeError("No supported Flash model is available for this API key.")

    # Selected model par final generation request bhejte hain.
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    if response and getattr(response, "text", ""):
        return response.text

    raise RuntimeError("Flash model returned an empty response.")


def init_state():
    # STEP 5: STREAMLIT SESSION STATE INITIALIZE KARNA
    # Isse हर rerun me chat/index reset nahi hota.
    if "kb" not in st.session_state:
        st.session_state.kb = None
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "uploaded_names" not in st.session_state:
        st.session_state.uploaded_names = []


def get_configured_api_key():
    # API KEY RESOLUTION ORDER:
    # 1) Streamlit secrets 2) Environment variable
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY", "")


def running_inside_streamlit():
    # Check karta hai code Streamlit runtime context me execute ho raha hai ya nahi.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def launch_streamlit_app():
    # Agar user `python app.py` run kare to app ko automatically `streamlit run` mode me launch kar do.
    app_path = Path(__file__).resolve()
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
        ],
    )


def main():
    # STEP 6: MAIN FRONTEND FLOW START
    st.set_page_config(
        page_title="Student Academic Assistant",
        page_icon="📚",
        layout="wide",
    )
    inject_custom_css()
    init_state()

    st.markdown("<h2>Study Workspace</h2>", unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload notes from sidebar, click process, then ask questions.</div>', unsafe_allow_html=True)

    # API key is loaded silently from secrets/env; no dedicated UI section needed.
    api_key = get_configured_api_key().strip()

    # LEFT SIDEBAR = Upload + processing only
    with st.sidebar:
        st.markdown("## Workspace")
        st.markdown("### Upload Notes")
        files = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="You can upload multiple notes files.",
        )

        if st.button("Process Documents", use_container_width=True):
            if not files:
                st.warning("Please upload at least one PDF or TXT file.")
            else:
                # Uploaded files ka index ready karte hain taaki question-answer possible ho.
                with st.spinner("Indexing your documents..."):
                    kb = build_knowledge_base(files)
                if kb is None:
                    st.error("Uploaded files had no readable text.")
                else:
                    st.session_state.kb = kb
                    st.session_state.uploaded_names = [f.name for f in files]
                    st.success(f"Knowledge base ready with {len(kb['chunks'])} chunks.")

        if st.session_state.uploaded_names:
            st.markdown("### Uploaded Files")
            for name in st.session_state.uploaded_names:
                st.markdown(f"- {name}")

    # MAIN AREA = Chat-style question answering area
    st.markdown("### Ask Questions")

    question = st.chat_input("Type your academic question here...")

    if question:
        if st.session_state.kb is None:
            st.warning("Please upload and process documents first.")
        else:
            kb = st.session_state.kb
            # User question ke liye top relevant chunks retrieve karte hain.
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
                        # Primary path: Gemini Flash response grounded on retrieved chunks.
                        answer_text = generate_with_flash(api_key.strip(), question, relevant)
                except Exception:
                    # Fallback path: still answer from local retrieval when API/model call fails.
                    fallback, _ = generate_extractive_answer(question, relevant)
                    st.warning("AI model response unavailable right now. Showing grounded answer from your uploaded files.")
                    answer_text = fallback
            else:
                # API key na ho to local extractive mode me answer dete hain.
                fallback, _ = generate_extractive_answer(question, relevant)
                answer_text = fallback

            # Chat history me new Q/A add karte hain.
            st.session_state.chat.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "sources": sorted({chunk["source"] for chunk in relevant}),
                }
            )

    if st.session_state.chat:
        # Latest answer ऊपर दिखाने ke liye reverse order me render karte hain.
        for item in reversed(st.session_state.chat):
            with st.chat_message("user"):
                st.markdown(item["question"])
            with st.chat_message("assistant"):
                st.markdown(item["answer"])
                st.markdown("Sources")
                for src in item["sources"]:
                    st.markdown(f'<div class="source-box">{src}</div>', unsafe_allow_html=True)
    else:
        st.info("Upload files, process them, and ask your first question.")


if __name__ == "__main__":
    # Direct python run aur streamlit run dono cases handle karte hain.
    if running_inside_streamlit():
        main()
    else:
        launch_streamlit_app()