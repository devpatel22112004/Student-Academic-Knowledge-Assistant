# Student Academic Knowledge Assistant (RAG-Based AI System)

This project is an AI-powered academic assistant that helps students interact with their study materials (PDFs and TXT files).

The system allows users to upload documents and ask questions about them, generating answers using Retrieval Augmented Generation (RAG) with source citations.

---

## Quick Start

```bash
# Setup (one-time)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run everything (PDF + TXT support)
bash run_phase1.sh data outputs
bash run_phase2.sh data outputs/vector_store

# Query your documents (Phase 3 - coming soon)
```

---

## Full Documentation

**See [DOCUMENTATION.md](DOCUMENTATION.md) for complete details:**
- Architecture and data flow
- Dependency explanation (including why TXT needs NO extra libraries)
- Code structure with examples
- All use cases and troubleshooting

---

## Files Structure

```
Student-Academic-Knowledge-Assistant/
├── data/                       # Place your PDFs and TXT files here
│   ├── pdfs/                   #   └─ PDF documents
│   ├── tesla.txt               #   └─ TXT documents
│   └── notes.txt
│
├── scripts/
│   ├── pdf_loader.py           # Phase 1: Document extraction (PDF + TXT)
│   └── document_pipeline.py     # Phase 2: Chunking + embedding + indexing
│
├── outputs/
│   ├── phase1_extracted/       # Phase 1 output (extracted text)
│   └── vector_store/           # Phase 2 output (FAISS index + metadata)
│
├── run_phase1.sh               # Phase 1 convenient runner
├── run_phase2.sh               # Phase 2 convenient runner
├── requirements.txt            # Dependencies
├── README.md                   # This file (quick start)
└── DOCUMENTATION.md            # Detailed technical guide
```

---

## Features

✅ **PDF Support** — Full page-wise text extraction  
✅ **TXT Support** — Plain text file ingestion  
✅ **Semantic Chunking** — Smart text splitting with overlap  
✅ **Dense Embeddings** — Sentence-Transformers model  
✅ **Fast Retrieval** — FAISS vector indexing  
✅ **Metadata Tracking** — Source document + page reference  
✅ **OCR Fallback** — [ADDITIONAL] Scanned PDF support  

---

## Tech Stack

- **Document Loading**: `pypdf`, native Python
- **Chunking**: `langchain-text-splitters`
- **Embeddings**: `sentence-transformers`
- **Indexing**: `faiss-cpu`
- **Matrix Operations**: `numpy`

---

## Next Steps

1. Place PDFs/TXT files in `data/` folder
2. Run: `bash run_phase2.sh data outputs/vector_store`
3. Check `outputs/vector_store/metadata.json` for indexed chunks
4. Phase 3 (retrieval + QA) coming soon!



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