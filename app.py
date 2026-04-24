import io # file ko savekare bina uska memory me handle karna.
import os # File handling aur environment variables ke liye.
import sys # python progrmam ka system level interaction ke liye.
from pathlib import Path 

import streamlit as st
from pypdf import PdfReader

# Main logic aur UI ke liye helper functions ko yahan se import karte hain.
from main import ( 
    build_search_index, 
    chunk_text,
    create_embeddings,
    find_relevant_chunks,
    generate_extractive_answer,
)

#extrenal CSS file se custom styles inject karne ke liye function.
def inject_custom_css(): 
    css_path = Path(__file__).resolve().parent / "styles" / "app.css" #PATH 
    if not css_path.exists(): 
        return

    css = css_path.read_text(encoding="utf-8") 
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True) # Inject custom CSS

# Uploaded PDF aur TXT files se text nikalne ke liye function.
def read_uploaded_documents(uploaded_files): 
    all_documents = []
    for uploaded in uploaded_files: # Har uploaded file ke liye uska name aur suffix nikalte hain.
        name = uploaded.name
        suffix = os.path.splitext(name)[1].lower() #SUFFIX PDF ya TXT hona chahiye, warna ignore kar denge.

        if suffix == ".pdf": 
            # PDF se page-wise text nikalte hain.
            pdf_reader = PdfReader(io.BytesIO(uploaded.getvalue()))
            for i, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_documents.append((f"{name} - Page {i}", page_text))
        elif suffix == ".txt":
            # TXT file ko direct text me convert karte hain.
            text = uploaded.getvalue().decode("utf-8", errors="ignore") #UTF-8 decode karte hain, errors ignore karte hain taaki koi bhi non-text content problem na kare.
            if text.strip():
                all_documents.append((name, text))

    return all_documents

# Uploaded documents ko RAG knowledge base me convert karne ke liye function.
#Flow: documents -> chunks -> embeddings -> FAISS index
def build_knowledge_base(uploaded_files):
    documents = read_uploaded_documents(uploaded_files)
    if not documents:
        return None
    #YAHA MAIN.PY KE LOGIC USE HO RAHE HE JO HAMNE IMPORT KIYE THE.
    chunks = chunk_text(documents) 
    embeddings, model = create_embeddings(chunks)
    index = build_search_index(embeddings) 

    return {
        "chunks": chunks,
        "model": model,
        "index": index,
    }

#Gemini Flash se final answer banana.
#Yeh function retrieved context ko prompt me bhejta hai.
def generate_with_flash(api_key, question, relevant_chunks):
    
    import google.generativeai as genai #LIBRARY IMPORT FOR GEMINI FLASH ANSWER GENERATION

    # Flash models ko context ke sath prompt bhejne ke liye format karte hain.
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
    # Knowledge base, chat history, aur uploaded file names ke liye session state variables banate hain.
    # Agar yeh variables pehle se exist nahi karte, to unhe initialize kar dete hain.
    # Isse hota yeh hai ki user jab naye question ke liye app ko rerun karega, to uska previous data loss nahi hoga.
    if "kb" not in st.session_state: 
        st.session_state.kb = None
    if "chat" not in st.session_state: 
        st.session_state.chat = []
    if "uploaded_names" not in st.session_state:
        st.session_state.uploaded_names = []

#API KEY LENA
def get_configured_api_key():
    # API key pehle secrets se, phir environment se aati hai.
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY", "") 

#
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