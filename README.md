# Student Academic Knowledge Assistant (RAG-Based AI System)

This project is an AI-powered academic assistant that helps students interact with their study materials.

Students can upload study material files (PDFs and text files). The system allows users to ask questions about the uploaded documents and generates answers using Retrieval Augmented Generation (RAG).

## Features

- Upload academic PDFs/TXT files
- Ask questions related to documents
- AI-generated answers using document context
- Source citation (document name and page number)

## Tech Stack

- Python
- LangChain Text Splitters
- FAISS Vector Database
- Sentence Transformers

## Dependency Policy

This project keeps only **required dependencies for Phase 1 + Phase 2**.

**REQUIRED (Original Project):**

- `pypdf` (PDF text extraction)
- `langchain-text-splitters` (document chunking)
- `sentence-transformers` (embedding generation)
- `faiss-cpu` (vector database/index)
- `numpy` (embedding matrix handling)

**ADDITIONAL – OCR Fallback (Our Enhancement):**

- `pytesseract` + `pdf2image` + `Pillow` (scanned PDF support)
- System packages: `tesseract-ocr`, `poppler-utils`

To install system packages for OCR fallback:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

### Environment setup (one-time)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

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
4. Build and run a working PDF/TXT text extraction script.

### Manual setup (run in terminal)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Output of Phase 1

- A working script: `scripts/pdf_loader.py`
- It loads one or more PDF/TXT files and extracts text
- It saves extracted text files in `outputs/`

### One-command direct run (recommended)

```bash
bash run_phase1.sh
```

First-time setup or dependency issue:

```bash
bash run_phase1.sh --install-deps
```

### Run the PDF loader

```bash
source .venv/bin/activate
python scripts/pdf_loader.py --input data --output outputs
```

You can also pass a single file (PDF or TXT):

```bash
python scripts/pdf_loader.py --input data/pdfs/sample.pdf --output outputs
python scripts/pdf_loader.py --input data/tesla.txt --output outputs
```

### Notes

- For scanned PDFs (image-only), text extraction may return very little text.
- OCR support can be added in a later phase if needed.

## Phase 2: Document Processing Pipeline (16–31 March)

### Objective

- Build a complete document ingestion and indexing pipeline.

### Tasks completed in this phase

1. Read PDF/TXT inputs and extract clean text.
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
python scripts/document_pipeline.py --input data --output outputs/vector_store
```

### [ADDITIONAL] Scanned PDF support (OCR Fallback – Our Enhancement)

This feature was **not part of the original project requirements**. We added it to handle scanned/image-only PDFs.

**How it works:**

1. Pipeline tries normal PDF text extraction first (using `pypdf`).
2. If a page has no selectable text, OCR fallback runs automatically using Tesseract.
3. Chunks are created from extracted text (normal or OCR).

**To disable OCR fallback** and use only text-based PDFs:

```bash
python scripts/document_pipeline.py --input data --output outputs/vector_store --disable-ocr-fallback
```

**Tune OCR quality:**

```bash
python scripts/document_pipeline.py --input data --output outputs/vector_store --ocr-dpi 400 --ocr-lang eng+hin
```

- `--ocr-dpi`: Render resolution (default 300; higher = better quality, slower).
- `--ocr-lang`: Tesseract language codes (e.g., `eng`, `eng+hin` for bilingual).

### One-command direct run (recommended)

```bash
bash run_phase2.sh
```

Direct script with custom parameters:

```bash
bash run_phase2.sh data outputs/vector_store 600 100 sentence-transformers/all-MiniLM-L6-v2
```

Manual fallback (if you want full control every time):

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/document_pipeline.py \
	--input data \
	--output outputs/vector_store \
	--chunk-size 600 \
	--chunk-overlap 100 \
	--model sentence-transformers/all-MiniLM-L6-v2
```

Single file input examples:

```bash
python scripts/document_pipeline.py --input data/pdfs/samp.pdf --output outputs/vector_store
python scripts/document_pipeline.py --input data/tesla.txt --output outputs/vector_store
```

### Optional tuning parameters

```bash
python scripts/document_pipeline.py \
	--input data \
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

- In Phase 2, we convert raw document text (PDF/TXT) into semantic units (chunks).
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

### Prevention (already applied)

- `run_phase2.sh` now uses `pip install --no-cache-dir -r requirements.txt` with `--install-deps`.
- This avoids growing pip cache on repeated installs.

### If disk reaches ~100%

- Yes, work can fail (save, install, git operations, indexing).
- Recreate `.venv` and reinstall only required dependencies.