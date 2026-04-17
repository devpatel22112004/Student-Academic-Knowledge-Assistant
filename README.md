# Student Academic Knowledge Assistant

A simple Python program that helps students find relevant information from their academic documents using AI-powered search.

---

## 📌 What Does This Project Do?

**Problem:** Students have many PDF and TXT notes, but finding relevant information is hard.

**Solution:** This program:
1. Reads your PDF and TXT files from the `data` folder
2. Converts the text into searchable vectors (called embeddings)
3. When you ask a question, it finds the most relevant pieces from your documents
4. Shows you the answers directly from your own study material

**Key Point:** All answers come from YOUR documents, not the internet.


---

## 🚀 How to Use

### Step 1: Setup (First Time Only)

```bash
# Install all required packages
pip install -r requirements.txt
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

### Step 4: Ask Your Questions

The program will load everything and ask for your questions. Just type and press Enter.

Type `quit` or `exit` to stop the program.

---

## 📁 Project Structure

```
Student-Academic-Knowledge-Assistant/
├── main.py                      # Main program (ONLY FILE YOU RUN)
├── data/                        # Put your PDF/TXT files here
│   ├── mumbaiindiansinfo.txt
│   └── pdfs/
├── requirements.txt             # Packages needed
└── README.md                    # This file
```

**Note:** The `outputs` folder is NOT used. The `scripts` folder is NOT used. Delete them if you want.

---

## 🔧 How It Works (Step by Step)

Your program has 6 main steps:

**Step 1: Find Documents**
- Looks for all PDF and TXT files in `data` folder

**Step 2: Read Content**
- Opens each file and reads the text inside

**Step 3: Break into Chunks**
- Splits large documents into smaller pieces (1000 characters each)
- Chunks overlap by 200 characters to keep context

**Step 4: Create Embeddings**
- Converts each chunk into a "vector" (list of numbers)
- Vectors represent the meaning of the text

**Step 5: Build Search Index**
- Stores vectors in a FAISS index for super-fast search

**Step 6: Answer Questions**
- Converts your question into a vector
- Finds the 5 most similar chunks
- Shows you the results

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

---

## ❓ FAQ

**Q: Can I use Word documents (.docx)?**
A: Not in this version. Use PDF or TXT files only.

**Q: How many documents can I use?**
A: As many as you want! The program will process all of them.

**Q: Does it need internet?**
A: No! Everything runs locally on your computer.

**Q: Do I need a .env file?**
A: No. This project does not require API keys or environment variables.

**Q: Where does it save my questions?**
A: Nowhere. Everything runs in memory.

**Q: Why is the first run slow?**
A: The embedding model takes time to download on first use. After that, it's cached and fast.

**Q: HF_TOKEN warning appears, is it required?**
A: Not required. HF token is optional and only helps with faster download/rate limits.

---

## 🎯 How to Explain This to Your Professor

**What it does:**
1. Reads PDF and TXT files from the `data` folder
2. Splits text into chunks (1000 characters with 200 overlap)
3. Creates AI vectors of each chunk using sentence-transformers
4. Builds a FAISS index for fast similarity search
5. When user asks a question, finds the 5 most similar chunks and shows them

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
- No internet or API keys needed
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

---

**Version:** 1.0 (Simplified)
**Last Updated:** 2026
**Language:** Python 3.8+