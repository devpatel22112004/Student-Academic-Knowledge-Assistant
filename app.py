import io # file ko savekare bina uska memory me handle karna.
import hashlib # password ko plain text me store karne se bachane ke liye.
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

    # Flash models ke liye preference order define karte hain. Jo pehle available hoga, use karenge.
    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-1.5-flash",
    ]

    # Available models me se Flash supported model ko select karte hain.
    available = []
    for model_info in genai.list_models():
        methods = getattr(model_info, "supported_generation_methods", [])
        if "generateContent" in methods:
            available.append(model_info.name)

    # Preferred order me se pehla available model select karte hain.
    model_name = None
    for candidate in preferred:
        if candidate in available:
            model_name = candidate
            break

    # Agar preferred models me se koi bhi available nahi hai, to kisi bhi Flash supported model ko select kar lete hain.
    if model_name is None:
        flash_models = [m for m in available if "flash" in m.lower()] #
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
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None


def hash_password(password):
    # Prototype auth ke liye password ko hash karke store karte hain.
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def render_auth_screen():
    # Page ke top par short product title aur tagline dikhate hain.
    st.markdown(
        """
        <div class="page-hero">
            <div class="page-title">Your Study Partner</div>
            <div class="page-subtitle">Smart answers from your notes.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Screen ko do parts me divide karte hain: left side intro, right side auth form.
    left_col, right_col = st.columns([0.95, 1.05], gap="large")

    with left_col:
        # Left panel me simple benefit text rakhte hain taaki screen friendly lage.
        st.markdown(
            """
            <div class="auth-shell">
                <div class="auth-hero">Upload once. Search forever.</div>
                <div class="auth-subtitle">
                    Keep your PDFs, ask questions anytime, get real answers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        # Right side par short header dikhate hain jo user ko form ka purpose batata hai.
        st.markdown(
            """
            <div class="auth-panel-head">
                <div class="auth-card-title">Be a Happy Student</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Login aur Register ko tabs me rakha hai taaki UI compact rahe.
        login_tab, register_tab = st.tabs(["Login", "Register"])

        with login_tab:
            # Login form me email aur password lete hain.
            with st.form("login_form", clear_on_submit=False):
                # Chhota helper text user ko context deta hai.
                st.markdown('<div class="auth-form-tip">Welcome back! Let\'s pick up where you left off.</div>', unsafe_allow_html=True)
                # Email field: existing account ko identify karne ke liye.
                login_email = st.text_input("Email address", placeholder="name@example.com")
                # Password field: hidden input, security ke liye.
                login_password = st.text_input("Password", type="password", placeholder="Enter your password")
                # Form submit button.
                login_submit = st.form_submit_button("Sign in", use_container_width=True)

            # Button click hone ke baad credentials validate karte hain.
            if login_submit:
                # Email ko normalize karke user lookup karte hain.
                user = st.session_state.users.get(login_email.strip().lower())
                if not user:
                    # Agar user nahi mila to register karne ke liye bolte hain.
                    st.error("User not found. Please register first.")
                elif user["password"] != hash_password(login_password):
                    # Password hash mismatch hone par login reject karte hain.
                    st.error("Invalid password.")
                else:
                    # Successful login par session state me auth flag set hota hai.
                    st.session_state.authenticated = True
                    # Current user ka data session me store hota hai.
                    st.session_state.current_user = user
                    # User ko success message dikhate hain.
                    st.success(f"Welcome back, {user['name']}!")
                    # Rerun se app auth screen se main workspace me chali jati hai.
                    st.rerun()

        with register_tab:
            # Register form me new account ke liye details lete hain.
            with st.form("register_form", clear_on_submit=False):
                # Chhota helper text form ko friendly banata hai.
                st.markdown('<div class="auth-form-tip">Create your account. Keep all your notes in one place.</div>', unsafe_allow_html=True)
                # Naam field: account owner ka display name.
                reg_name = st.text_input("Full name", placeholder="Your name")
                # Email field: future login ke liye unique key.
                reg_email = st.text_input("Email address", placeholder="name@example.com")
                # Password field: new account password.
                reg_password = st.text_input("Create password", type="password", placeholder="Set a strong password")
                # Confirm password field: typo avoid karne ke liye.
                reg_confirm = st.text_input("Confirm password", type="password", placeholder="Repeat your password")
                # Account create button.
                register_submit = st.form_submit_button("Create account", use_container_width=True)

            # Register button click hone par validation + save hota hai.
            if register_submit:
                # Email ko clean format me lete hain.
                email_key = reg_email.strip().lower()
                if not reg_name.strip() or not email_key or not reg_password:
                    # Blank values allow nahi karte.
                    st.error("Name, email, and password are required.")
                elif reg_password != reg_confirm:
                    # Dono password same hone chahiye.
                    st.error("Passwords do not match.")
                elif email_key in st.session_state.users:
                    # Same email pe duplicate account nahi banana.
                    st.error("This email is already registered.")
                else:
                    # User data ko session_state me save karte hain.
                    st.session_state.users[email_key] = {
                        "name": reg_name.strip(),
                        "email": email_key,
                        # Password plain text me nahi, hash form me save hota hai.
                        "password": hash_password(reg_password),
                    }
                    # Register ke baad user ko automatically logged in mark kar dete hain.
                    st.session_state.authenticated = True
                    # Current user ka record session me set hota hai.
                    st.session_state.current_user = st.session_state.users[email_key]
                    # Success message ke baad main app me rerun hota hai.
                    st.success("Account created successfully.")
                    st.rerun()

#API KEY LENA
def get_configured_api_key():
    # API key pehle secrets se, phir environment se aati hai.
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY", "") 

# Check karta hai ki app Streamlit ke andar chal raha hai ya nahi, taaki uske hisab se behavior decide kar sake.
def running_inside_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx 

        return get_script_run_ctx() is not None 
    except Exception:
        return False

#source detalis and preview text nikalne ke liye function.
def prepare_source_items(relevant_chunks):
    source_items = []
    seen = set() #duplicate handle karne ke liye set banate hain.
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

#direct run button ke liye function
def launch_streamlit_app():
    app_path = Path(__file__).resolve() 
    #process switch kanrne ke liye 
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

#main function
def main():
    st.set_page_config( 
        page_title="Student Academic Assistant",
        page_icon="📚",
        layout="wide",
    )

    inject_custom_css()
    init_state() #Session state initialize karte hain taaki knowledge base, chat history, aur uploaded file names store ho sake.

    if not st.session_state.authenticated:
        render_auth_screen()
        return

 
    st.markdown(
        """
        <div class="workspace-hero">
            <div class="workspace-kicker">Your workspace</div>
            <div class="workspace-title">Study Friend</div>
            <div class="workspace-subtitle">Upload PDFs. Ask questions. Get answers from your notes.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    
    api_key = get_configured_api_key().strip()

    with st.sidebar:
        if st.session_state.current_user:
            # User ka short monogram nikalte hain taaki avatar button compact rahe.
            user_initial = "".join(part[0] for part in st.session_state.current_user["name"].split()[:2]).upper()
            if not user_initial:
                # Agar naam unexpected ho to first letter fallback use karte hain.
                user_initial = st.session_state.current_user["name"][0].upper()
            # Round avatar button click karne par popover open hota hai.
            with st.popover(user_initial):
                # Popover me current account details show karte hain.
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
                # Sign out click hone par session reset karke login screen par bhejte hain.
                if st.button("Sign out", use_container_width=True, key="sidebar_signout_button"):
                    # Authentication flag off karte hain.
                    st.session_state.authenticated = False
                    # Current user clear karte hain.
                    st.session_state.current_user = None
                    # Processed knowledge base reset hota hai.
                    st.session_state.kb = None
                    # Chat history clear hoti hai.
                    st.session_state.chat = []
                    # Uploaded file list clear hoti hai.
                    st.session_state.uploaded_names = []
                    # UI ko fresh auth screen par rerun karte hain.
                    st.rerun()
        st.markdown('<div class="sidebar-section-title">Workspace</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-subtitle">Upload your PDFs here. Search them whenever you need.</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        # File uploader user se PDFs/TXT leta hai.
        files = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload as many files as you want.",
        )
       
        # Process button click hone par uploaded files ko knowledge base me convert karte hain.
        if st.button("Process Documents", use_container_width=True):
            if not files:
                # Agar file nahi hai to warning dikhate hain.
                st.warning("Please upload at least one PDF or TXT file.")
            else:
                # Uploaded files ka index ready karte hain taaki question-answer possible ho.
                with st.spinner("Processing your files..."):
                    # File reading, chunking, embedding aur index building yahan hota hai.
                    kb = build_knowledge_base(files) 
                if kb is None:
                    # Agar text extract nahi hua to error dikhate hain.
                    st.error("Uploaded files had no readable text.")
                else:
                    # Knowledge base session me store kar dete hain.
                    st.session_state.kb = kb
                    # Uploaded file names bhi session me save karte hain.
                    st.session_state.uploaded_names = [f.name for f in files]
                    # Success message user ko confirm karta hai ke files ready hain.
                    st.success("All set! Now you can ask your questions.")
        
        if st.session_state.kb is not None:
            # Ready badge sirf tab dikhate hain jab knowledge base available ho.
            st.markdown('<div class="status-pill">Ready</div>', unsafe_allow_html=True)

        if st.session_state.uploaded_names:
            # Uploaded file list user ko quick overview deti hai.
            st.markdown(" Your Uploaded Files")
            for name in st.session_state.uploaded_names:
                st.markdown(f"- {name}")

    # Main area me user question aur answer render hota hai.
    # Main question section ka short title.
    st.markdown("Ask Away")
    
    # Chat input se user ka next question aata hai.
    question = st.chat_input("What do you want to know?")

    if question:
        if st.session_state.kb is None:
            # Documents process kiye bina answer generate nahi hota.
            st.warning("Please upload and process documents first.")
        else:
            # Saved knowledge base ko local variable me lete hain.
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
                    # API key available ho to Gemini Flash se answer banate hain.
                    with st.spinner("Generating answer..."):
                        # Primary path: Gemini Flash answer.
                        answer_text = generate_with_flash(api_key.strip(), question, relevant)
                except Exception:
                    # Fallback: local extractive answer.
                    # Agar AI model fail ho jaye to uploaded text se direct answer nikalte hain.
                    fallback, _ = generate_extractive_answer(question, relevant)
                    st.warning("AI model response unavailable right now. Showing grounded answer from your uploaded files.")
                    answer_text = fallback
            else:
                # API key na ho to local mode use hota hai.
                # Bina API key ke bhi grounded extractive answer milta rahe.
                fallback, _ = generate_extractive_answer(question, relevant)
                answer_text = fallback

            # New question-answer chat history me save hota hai.
            # Ye history session me rehti hai taaki rerun ke baad bhi chat dikh sake.
            st.session_state.chat.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "sources": sorted({chunk["source"] for chunk in relevant}),
                    "source_items": prepare_source_items(relevant),
                }
            )
    #letest chat upper rakhta he
    if st.session_state.chat:
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

                # Sources ko visually distinct pills me render karte hain.
                st.markdown('<div class="source-wrap">', unsafe_allow_html=True)
                st.markdown('<div class="source-title">Sources used</div>', unsafe_allow_html=True)
                for src in item["sources"]:
                    st.markdown(f'<span class="source-pill">{src}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                #expand karke dekhne ke liye
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