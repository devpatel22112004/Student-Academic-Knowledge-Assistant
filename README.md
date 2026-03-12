# Student Academic Knowledge Assistant (RAG-Based AI System)

This project is an AI-powered academic assistant that helps students interact with their study materials.

Students can upload PDFs such as lecture notes, assignments, and past exam papers. The system allows users to ask questions about the uploaded documents and generates answers using Retrieval Augmented Generation (RAG).

## Features

- Upload academic PDFs
- Ask questions related to documents
- AI-generated answers using document context
- Source citation (document name and page number)

## Tech Stack

- Python
- LangChain
- FAISS Vector Database
- Sentence Transformers
- Streamlit

## Phase 1: Research & Setup (1–15 March)

### Objectives

- Understand RAG architecture
- Set up development environment
- Create project repository structure

### What we do in this phase

1. Learn the RAG pipeline at a high level:
	- Document loading
	- Text chunking
	- Embeddings
	- Vector storage and retrieval
	- LLM answer generation with context
2. Prepare a clean Python environment for this project.
3. Install core libraries for upcoming RAG work.
4. Build and run a working PDF text extraction script.

### How to set up (run in terminal)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Output of Phase 1

- A working script: `scripts/pdf_loader.py`
- It loads one or more PDFs and extracts page-wise text
- It saves extracted text files in `outputs/`

### Run the PDF loader

```bash
source .venv/bin/activate
python scripts/pdf_loader.py --input data/pdfs --output outputs
```

You can also pass a single PDF file:

```bash
python scripts/pdf_loader.py --input data/pdfs/sample.pdf --output outputs
```

### Notes

- For scanned PDFs (image-only), text extraction may return very little text.
- OCR support can be added in a later phase if needed.