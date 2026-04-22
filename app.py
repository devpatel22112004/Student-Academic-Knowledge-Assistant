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
    # STEP 1: UI styling yahan apply hoti hai.
    # Small project ke liye internal CSS theek hai; larger project me external CSS zyada clean hoti hai.
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

        .status-pill {
            display: inline-block;
            border: 1px solid rgba(17, 106, 123, 0.25);
            background: rgba(17, 106, 123, 0.08);
            color: #0f5160;
            border-radius: 999px;
            padding: 0.2rem 0.62rem;
            font-size: 0.78rem;
            font-weight: 600;
            margin-top: 0.35rem;
            margin-bottom: 0.5rem;
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

        .answer-shell {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem 1.05rem;
            margin-top: 0.35rem;
            box-shadow: 0 6px 16px rgba(11, 24, 45, 0.05);
        }

        .answer-label {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.45rem;
        }

        .answer-text {
            font-size: 0.98rem;
            line-height: 1.65;
            color: var(--text-main);
        }

        .source-wrap {
            margin-top: 0.75rem;
        }

        .source-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }

        .source-pill {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            margin: 0.18rem 0.28rem 0 0;
            border-radius: 999px;
            border: 1px solid rgba(17, 106, 123, 0.18);
            background: rgba(17, 106, 123, 0.08);
            color: #0f5160;
            font-size: 0.85rem;
        }

        .source-preview {
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            margin-top: 0.45rem;
            background: rgba(255, 255, 255, 0.7);
        }

        .source-preview-title {
            font-size: 0.84rem;
            font-weight: 700;
            color: #0f5160;
            margin-bottom: 0.25rem;
        }

        .source-preview-text {
            font-size: 0.88rem;
            color: var(--muted);
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_uploaded_documents(uploaded_files):
    """
    STEP 2: Uploaded files ka text read karna.
    Return format: [(source_label, text_content), ...]
    """
    all_documents = []
    for uploaded in uploaded_files:
        name = uploaded.name
        suffix = os.path.splitext(name)[1].lower()

        if suffix == ".pdf":
            # PDF se page-wise text nikalte hain.
            pdf_reader = PdfReader(io.BytesIO(uploaded.getvalue()))
            for i, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_documents.append((f"{name} - Page {i}", page_text))
        elif suffix == ".txt":
            # TXT file ko direct text me convert karte hain.
            text = uploaded.getvalue().decode("utf-8", errors="ignore")
            if text.strip():
                all_documents.append((name, text))

    return all_documents


def build_knowledge_base(uploaded_files):
    """
    STEP 3: RAG knowledge base banana.
    Flow: documents -> chunks -> embeddings -> FAISS index
    """
    documents = read_uploaded_documents(uploaded_files)
    if not documents:
        return None

    # Text ko small chunks me todte hain.
    chunks = chunk_text(documents)
    # Har chunk ka embedding banate hain.
    embeddings, model = create_embeddings(chunks)
    # Fast search ke liye FAISS index banate hain.
    index = build_search_index(embeddings)

    return {
        "chunks": chunks,
        "model": model,
        "index": index,
    }


def generate_with_flash(api_key, question, relevant_chunks):
    """
    STEP 4: Gemini Flash se final answer banana.
    Yeh function retrieved context ko prompt me bhejta hai.
    """
    import google.generativeai as genai

    # Sirf retrieved context prompt me bhejte hain.
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

    # API key set karne ke baad hi model call hota hai.
    genai.configure(api_key=api_key)

    # Available Flash models me se supported model choose karte hain.
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

    # Selected model par answer generate karte hain.
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    if response and getattr(response, "text", ""):
        return response.text

    raise RuntimeError("Flash model returned an empty response.")


def init_state():
    # STEP 5: Session state initialize karna.
    # Isse rerun par data reset nahi hota.
    if "kb" not in st.session_state:
        st.session_state.kb = None
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "uploaded_names" not in st.session_state:
        st.session_state.uploaded_names = []


def get_configured_api_key():
    # API key pehle secrets se, phir environment se aati hai.
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY", "")


def running_inside_streamlit():
    # Check karta hai app Streamlit ke andar chal raha hai ya nahi.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def prepare_source_items(relevant_chunks):
    # Same source repeat hone par first useful snippet hi show karte hain.
    source_items = []
    seen = set()
    for chunk in relevant_chunks:
        src = chunk["source"]
        if src in seen:
            continue
        seen.add(src)
        preview = " ".join(chunk["text"].split())[:240]
        if len(preview) == 240:
            preview += "..."
        source_items.append({"source": src, "preview": preview})
    return source_items


def launch_streamlit_app():
    # Agar user direct python app.py chalaye, to Streamlit mode me launch karo.
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
    # STEP 6: Main app UI start.
    st.set_page_config(
        page_title="Student Academic Assistant",
        page_icon="📚",
        layout="wide",
    )
    inject_custom_css()
    init_state()

    st.markdown("<h2>Study Workspace</h2>", unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload your notes, process once, then ask questions with grounded answers.</div>', unsafe_allow_html=True)

    # API key background me secrets/env se load hoti hai.
    api_key = get_configured_api_key().strip()

    # Left sidebar me sirf upload aur process rakha hai.
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
                    st.success("Documents processed. You can ask questions now.")

        if st.session_state.kb is not None:
            st.markdown('<div class="status-pill">Ready</div>', unsafe_allow_html=True)

        if st.session_state.uploaded_names:
            st.markdown("### Uploaded Files")
            for name in st.session_state.uploaded_names:
                st.markdown(f"- {name}")

    # Main area me user question aur answer render hota hai.
    st.markdown("### Ask Questions")

    question = st.chat_input("Type your academic question here...")

    if question:
        if st.session_state.kb is None:
            st.warning("Please upload and process documents first.")
        else:
            kb = st.session_state.kb
            # Question ke liye top relevant chunks nikalte hain.
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
                        # Primary path: Gemini Flash answer.
                        answer_text = generate_with_flash(api_key.strip(), question, relevant)
                except Exception:
                    # Fallback: local extractive answer.
                    fallback, _ = generate_extractive_answer(question, relevant)
                    st.warning("AI model response unavailable right now. Showing grounded answer from your uploaded files.")
                    answer_text = fallback
            else:
                # API key na ho to local mode use hota hai.
                fallback, _ = generate_extractive_answer(question, relevant)
                answer_text = fallback

            # New question-answer chat history me save hota hai.
            st.session_state.chat.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "sources": sorted({chunk["source"] for chunk in relevant}),
                    "source_items": prepare_source_items(relevant),
                }
            )

    if st.session_state.chat:
        # Latest chat sabse upar dikhate hain.
        for item in reversed(st.session_state.chat):
            with st.chat_message("user"):
                st.markdown(item["question"])
            with st.chat_message("assistant"):
                # Answer ko ek clean card me render karte hain.
                st.markdown(
                    f'''
                    <div class="answer-shell">
                        <div class="answer-label">Answer</div>
                        <div class="answer-text">{item["answer"].replace(chr(10), "<br>")}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

                # Sources ko short pills me dikhate hain taaki clutter na lage.
                st.markdown('<div class="source-wrap">', unsafe_allow_html=True)
                st.markdown('<div class="source-title">Sources used</div>', unsafe_allow_html=True)
                for src in item["sources"]:
                    st.markdown(f'<span class="source-pill">{src}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Professional source panel: source name + short retrieved snippet.
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


if __name__ == "__main__":
    # Direct python run aur streamlit run dono support.
    if running_inside_streamlit():
        main()
    else:
        launch_streamlit_app()