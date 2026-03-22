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

## Phase 2: Document Processing Pipeline (16–31 March)

### Objective

- Build a complete document ingestion and indexing pipeline.

### Tasks completed in this phase

1. Read PDFs and extract clean page-level text.
2. Split text into smaller overlapping chunks.
3. Generate dense vector embeddings for each chunk.
4. Store all vectors in FAISS vector database.
5. Save metadata (chunk ID, source file, page number, chunk text).

### Script used

- `scripts/document_pipeline.py`

### Run command

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/document_pipeline.py --input data/pdfs --output outputs/vector_store
```

### One-command direct run (recommended)

```bash
bash run_phase2.sh
```

Direct script with custom parameters:

```bash
bash run_phase2.sh data/pdfs outputs/vector_store 600 100 sentence-transformers/all-MiniLM-L6-v2
```

Manual fallback (if you want full control every time):

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/document_pipeline.py \
	--input data/pdfs \
	--output outputs/vector_store \
	--chunk-size 600 \
	--chunk-overlap 100 \
	--model sentence-transformers/all-MiniLM-L6-v2
```

Single PDF input example:

```bash
python scripts/document_pipeline.py --input data/pdfs/samp.pdf --output outputs/vector_store
```

### Optional tuning parameters

```bash
python scripts/document_pipeline.py \
	--input data/pdfs \
	--output outputs/vector_store \
	--chunk-size 600 \
	--chunk-overlap 100 \
	--model sentence-transformers/all-MiniLM-L6-v2
```

### Output artifacts

- `outputs/vector_store/index.faiss` → FAISS index file
- `outputs/vector_store/metadata.json` → chunk metadata for traceability
- `outputs/vector_store/vectors_shape.json` → vector count and embedding dimension

### How to explain this to your sir

- In Phase 2, we convert raw PDF text into semantic units (chunks).
- Every chunk is converted into a numerical embedding using a transformer model.
- These embeddings are indexed in FAISS for fast similarity search.
- Metadata is preserved to support source citation in later phases.
- This phase prepares the system for Phase 3: retrieval + answer generation.

## Codespace Low Disk Warning (Permanent Fix Guide)

If you see warning like **"Low disk space available"**, this is about **Codespace container storage**, not your local laptop disk.

### Why this happens

- Python environment (`.venv`) can become large.
- `sentence-transformers` dependency chain installs `torch`, and sometimes heavy `nvidia` packages.
- pip / Hugging Face caches can grow to multiple GB.

### Quick safe cleanup

```bash
bash cleanup_codespace_space.sh
```

### Prevention (already applied)

- `run_phase2.sh` now uses `pip install --no-cache-dir -r requirements.txt` with `--install-deps`.
- This avoids growing pip cache on repeated installs.

### If disk reaches ~100%

- Yes, work can fail (save, install, git operations, indexing).
- Run cleanup script first.
- If still low, recreate `.venv` as a fresh smaller environment.