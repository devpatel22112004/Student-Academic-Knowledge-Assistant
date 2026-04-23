# Student Academic Knowledge Assistant

A Python-based academic assistant that helps students ask questions over their own notes using retrieval and optional Gemini Flash generation.

---

## 📌 What Does This Project Do?

**Problem:** Students have many PDF and TXT notes, but finding relevant information is hard.

**Solution:** This project:
1. Reads your PDF and TXT files from the `data` folder
2. Splits text into chunks and converts chunks into embeddings
3. Retrieves the most relevant chunks for each question
4. Generates grounded answers from your own study material

**Key Point:** All answers come from YOUR documents, not the internet.


---

## 🚀 How to Use

### Step 1: Setup (First Time Only)

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 1.1: Optional (Gemini Flash API Key)

If you want LLM-generated answers (instead of only extractive local answers), create a Google AI Studio API key.

Best option: store it once in Streamlit secrets so you do not need to paste it every time.

1. Create the file `.streamlit/secrets.toml`
2. Add:

```bash
GEMINI_API_KEY="your_api_key_here"
```

Optional terminal env var:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Step 2: Add Your Documents

Put your PDF and TXT files in the `data` folder:
```
data/
  ├── notes.pdf
  ├── lectures.txt
  ├── chapter1.pdf
  └── study_guide.txt
```

### Step 3: Run the Program

```bash
python main.py
```

### Step 3 (Web Frontend)

Run the frontend app:

```bash
python -m streamlit run app.py
```

You can also run:

```bash
python app.py
```

Then:
1. Upload PDF/TXT files from UI
2. Click **Process Documents**
3. The app auto-loads the API key from Streamlit secrets or environment
4. Ask questions in chat box

### Step 4: Ask Your Questions

The program will load everything and ask for your questions. Just type and press Enter.

Type `quit` or `exit` to stop the program.

---

## 📁 Project Structure

```
Student-Academic-Knowledge-Assistant/
├── app.py                       # Streamlit frontend (upload + chat + Gemini)
├── main.py                      # CLI version of the assistant
├── .streamlit/
│   └── secrets.toml             # Optional local API key config
├── data/                        # Put your PDF/TXT files here
│   ├── mumbaiindiansinfo.txt
│   └── pdfs/
├── requirements.txt             # Packages needed
└── README.md                    # This file
```

**Note:** `.streamlit/secrets.toml` is local and should not be committed.

---

## 🔧 How It Works (Step by Step)

The core flow has 6 steps:

**Step 1: Find Documents**
- Looks for all PDF and TXT files in `data` folder

**Step 2: Read Content**
- Opens each file and reads the text inside

**Step 3: Break into Chunks**
- Splits large documents into smaller pieces (700 characters each)
- Chunks overlap by 120 characters to preserve context

**Step 4: Create Embeddings**
- Converts each chunk into a "vector" (list of numbers)
- Vectors represent the meaning of the text

**Step 5: Build Search Index**
- Stores vectors in a FAISS index for super-fast search

**Step 6: Answer Questions**
- Retrieves the top relevant chunks
- Uses Gemini Flash (if API key is available)
- Falls back to local extractive answer if API/model is unavailable

---

## 📦 Requirements

Install packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**What each package does:**
- `langchain-text-splitters`: Splits text into chunks
- `faiss-cpu`: Fast similarity search
- `sentence-transformers`: Converts text to vectors (embeddings)
- `pypdf`: Reads PDF files
- `numpy`: Handles number arrays
- `streamlit`: Web interface
- `google-generativeai`: Gemini Flash API integration

---

## ❓ FAQ

**Q: Can I use Word documents (.docx)?**
A: Not in this version. Use PDF or TXT files only.

**Q: How many documents can I use?**
A: As many as you want! The program will process all of them.

**Q: Does it need internet?**
A: Local retrieval works without internet. Gemini Flash generation requires internet and API access.

**Q: Do I need a .env file?**
A: No. You can store `GEMINI_API_KEY` in `.streamlit/secrets.toml` or set it as an environment variable.

**Q: Where does it save my questions?**
A: Chat history is kept in Streamlit session memory while the app is running.

**Q: Why is the first run slow?**
A: The embedding model takes time to download on first use. After that, it's cached and fast.

**Q: HF_TOKEN warning appears, is it required?**
A: Not required. HF token is optional and only helps with faster download/rate limits.

---

## 🎯 How to Explain This to Your Professor

**What it does:**
1. Reads PDF and TXT files from the `data` folder
2. Splits text into chunks (700 characters with 120 overlap)
3. Creates AI vectors of each chunk using sentence-transformers
4. Builds a FAISS index for fast similarity search
5. Retrieves relevant chunks and answers from those chunks
6. Optionally uses Gemini Flash for better natural responses

**Technologies used:**
- **Python**: Programming language
- **FAISS**: Facebook AI Similarity Search (fast vector search)
- **Sentence-Transformers**: Pre-trained model for text embeddings
- **PyPDF**: PDF text extraction
- **LangChain**: Text splitting library
- **NumPy**: Numeric array operations

**Why it's good:**
- Simple and easy to understand
- Uses industry-standard tools
- All answers come from YOUR documents
- Works locally even without API keys
- Fast and efficient

---

## 🛠️ Troubleshooting

**"No PDF or TXT files found"**
- Check that files are in the `data` folder
- Make sure file names end with `.pdf` or `.txt`

**"ModuleNotFoundError"**
- Run: `pip install -r requirements.txt`

**Program runs slowly**
- First run is slower (downloading embedding model)
- Subsequent runs are faster

**Can't find the answer I'm looking for**
- Try asking your question differently
- Add more relevant documents to `data` folder

---

## 📚 Code Explanation

### [main.py](main.py)

- `find_all_documents()`: Finds all PDF/TXT files in `data` folder
- `read_document_content()`: Reads content from PDF or TXT
- `chunk_text()`: Breaks text into manageable pieces
- `create_embeddings()`: Converts chunks to vectors
- `build_search_index()`: Creates FAISS index
- `find_relevant_chunks()`: Finds similar chunks for your question
- `main()`: Main program that runs all 6 steps

### [app.py](app.py)

- Streamlit frontend for upload, processing, and Q/A
- Gemini Flash integration with automatic supported-model selection
- Fallback local extractive answer if API/model call fails
- Source preview panel for retrieved context

---

**Version:** 1.0 (Simplified)
**Last Updated:** 2026
**Language:** Python 3.8+

---

## 🌐 Frontend Features (New)

- Clean modern interface
- Multi-file upload (PDF + TXT)
- RAG retrieval from uploaded files
- Gemini Flash answer generation
- Source pills and source preview details
- Fallback local extractive mode when API key is missing