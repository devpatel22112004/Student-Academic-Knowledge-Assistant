# Student Academic Knowledge Assistant (RAG-Based AI System)

This project is a Retrieval Augmented Generation (RAG) foundation system for academic content.
It helps process study documents and retrieve relevant context for user questions.

Current completed scope:
- Phase 1: Document loading and text extraction
- Phase 2: Chunking, embeddings, and vector indexing
- Phase 3: Retrieval and retrieval-quality evaluation

---

## 1. Project Goal (0-Level Explanation)

The project solves this problem:
- Students have many notes and files
- They want fast, relevant answers from their own material
- Generic chatbot answers may be incorrect or unrelated

So this system does the following:
1. Read academic documents
2. Convert document text into searchable vector format
3. Retrieve top relevant chunks for a user query
4. (Next phases) use retrieved chunks with an LLM for final answer generation

Important architecture point:
- Answers must come from uploaded documents, not random internet text

---

## 2. What Is Completed So Far

### Phase 1 (Completed)

Module:
- [scripts/pdf_loader.py](scripts/pdf_loader.py)

Runner:
- [run_phase1.sh](run_phase1.sh)

What it does:
1. Scans input path for supported files
2. Extracts text per format
3. Saves normalized text output in extracted folder

Supported formats in Phase 1:
- PDF
- TXT
- RTF
- DOCX
- DOC

Output default:
- [outputs/extracted](outputs/extracted)

---

### Phase 2 (Completed)

Module:
- [scripts/document_pipeline.py](scripts/document_pipeline.py)

Runner:
- [run_phase2.sh](run_phase2.sh)

What it does:
1. Reads supported documents
2. Splits text into chunks with overlap
3. Generates embeddings using sentence-transformers
4. Stores vectors in FAISS
5. Saves metadata and vector shape info

Output default:
- [outputs/vector_store](outputs/vector_store)

Main output artifacts:
1. [outputs/vector_store/index.faiss](outputs/vector_store/index.faiss)
2. [outputs/vector_store/metadata.json](outputs/vector_store/metadata.json)
3. [outputs/vector_store/vectors_shape.json](outputs/vector_store/vectors_shape.json)

---

### Phase 3 (Completed)

Module:
- [scripts/retrieval_system.py](scripts/retrieval_system.py)

Runner:
- [run_phase3.sh](run_phase3.sh)

Modes:
1. Query mode
- Semantic retrieval for real user query
- Returns top-k relevant chunks

2. Eval mode
- Retrieval quality metrics
- Outputs exact_hit_rate, source_hit_rate, mrr

---

## 3. Additional Enhancements Added Beyond Basic Scope

These are enhancements added during development:

1. OCR fallback for scanned PDF pages
- Implemented in [scripts/document_pipeline.py](scripts/document_pipeline.py)
- Used when PDF text extraction returns empty text for image-based pages

2. Extended document format support
- Added support for RTF, DOCX, DOC (in addition to PDF/TXT)
- Implemented in both:
  - [scripts/pdf_loader.py](scripts/pdf_loader.py)
  - [scripts/document_pipeline.py](scripts/document_pipeline.py)

3. Better output alignment
- Phase 1 default output aligned to [outputs/extracted](outputs/extracted)

4. Cleaner comments and readability improvements
- Line-by-line simple comments added in core scripts and runners

5. Null character sanitization
- Fixed hidden null byte issue from some RTF extraction paths

---

## 4. Full Folder Structure

Current practical structure:

```text
Student-Academic-Knowledge-Assistant/
├── data/
│   ├── pdfs/                     # Optional nested PDF folder
│   ├── cskinfo.rtf               # Example input
│   ├── mumbaiindiansinfo.txt     # Example input
│   └── rcbinfo.pdf               # Example input
│
├── scripts/
│   ├── pdf_loader.py             # Phase 1 extraction module
│   ├── document_pipeline.py      # Phase 2 indexing module
│   └── retrieval_system.py       # Phase 3 retrieval module
│
├── outputs/
│   ├── extracted/                # Phase 1 output text files
│   │   ├── cskinfo.txt
│   │   ├── mumbaiindiansinfo.txt
│   │   └── rcbinfo.txt
│   │
│   └── vector_store/             # Phase 2 output artifacts
│       ├── index.faiss
│       ├── metadata.json
│       └── vectors_shape.json
│
├── run_phase1.sh                 # Phase 1 runner
├── run_phase2.sh                 # Phase 2 runner
├── run_phase3.sh                 # Phase 3 runner
├── run_unified.sh                # Unified Phase 1 + 2 runner
├── requirements.txt              # Project dependencies
├── README.md                     # This file
└── DOCUMENTATION.md              # Extended technical notes
```

---

## 5. Dependency Explanation (Why Each Package Is Installed)

See [requirements.txt](requirements.txt) for package list.

Core packages:
1. langchain-text-splitters
- Chunking with size and overlap

2. faiss-cpu
- Vector indexing and similarity search

3. sentence-transformers
- Text to embedding conversion

4. pypdf
- PDF text extraction

5. numpy
- Embedding matrix handling

6. python-docx
- DOCX text extraction

7. striprtf
- RTF to plain-text conversion

Additional OCR packages:
1. pytesseract
- OCR engine bridge

2. pdf2image
- Convert PDF pages to images for OCR

3. Pillow
- Image handling dependency

System tools needed for full support:
1. antiword (for DOC)
2. tesseract-ocr (for OCR)
3. poppler-utils (for OCR rendering)

---

## 6. Setup and Run Commands

### One-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional system tools:

```bash
sudo apt-get update
sudo apt-get install -y antiword tesseract-ocr poppler-utils
```

---

### Phase 1 run

Script way:

```bash
bash run_phase1.sh
```

Manual way:

```bash
python scripts/pdf_loader.py --input data --output outputs/extracted
```

---

### Phase 2 run

Script way:

```bash
bash run_phase2.sh
```

Manual way:

```bash
python scripts/document_pipeline.py --input data --output outputs/vector_store
```

---

### Phase 3 run

Query mode:

```bash
bash run_phase3.sh query --query "Who is the captain of CSK?" --top-k 5
```

Eval mode:

```bash
bash run_phase3.sh eval --top-k 5 --sample-size 20
```

---

### Unified run (Phase 1 + 2)

```bash
bash run_unified.sh
```

Use when you want extraction + indexing in one command.

---

## 7. Phase-wise Understanding

### Why Phase 1 exists if Phase 2 can read raw files?

Phase 2 can read raw files directly.
Still Phase 1 is useful for:
1. Isolated extraction debugging
2. Inspecting extracted text quality
3. Reusing extracted text for repeated experiments

### Why Phase 2 is needed?

Because retrieval needs vector index, not plain text files.
Phase 2 creates searchable vector artifacts.

### Why Phase 3 is needed?

Phase 3 is actual retriever layer.
Without it, there is no semantic search interface.

---

## 8. Current Status vs Internship Plan

Completed:
1. Phase 1
2. Phase 2
3. Phase 3

Not completed yet (future phases):
1. Phase 4 RAG answer generation with LLM
2. Phase 5 Frontend/UI
3. Later enhancement and finalization phases

---

## 9. Troubleshooting Quick Guide

1. Phase 1 says no supported files found
- Check input path and file extensions

2. DOC file fails
- Install antiword

3. OCR not working
- Install tesseract-ocr and poppler-utils

4. Phase 2 interrupted with exit code 130
- Process was manually interrupted (Ctrl+C)
- Re-run and let it finish

5. Phase 3 missing index
- Run Phase 2 first to generate vector store

---

## 10. Update Policy (Important)

Whenever new development is done, update this README in the same commit.
At minimum update these sections:
1. Completed scope
2. Folder structure
3. New scripts/modules
4. Run commands
5. Dependency explanation
6. Phase status

This keeps project documentation always synced with code.

