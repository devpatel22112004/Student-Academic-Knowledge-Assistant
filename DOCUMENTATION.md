# Student Academic Knowledge Assistant — Complete Documentation

## Overview

यह project एक **RAG-based (Retrieval Augmented Generation)** academic assistant है जो PDF और TXT दोनों files को ingestion करके semantic search के लिए vector indexing करता है।

---

## Architecture: Phase 1 + Phase 2

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Document Loading                                 │
│  ─────────────────────────────────────────────────────────  │
│  Input: PDF files / TXT files (from data/ folder)           │
│  Process: Extract text content                              │
│    • PDF → pypdf library (page-wise extraction)             │
│    • TXT → Plain Python file reading (no extra library)     │
│  Output: Extracted .txt files in outputs/ folder            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Document Indexing                                 │
│  ─────────────────────────────────────────────────────────  │
│  Input: Phase 1 ke extracted text / files                   │
│  Process:                                                   │
│    1) Chunking: RecursiveCharacterTextSplitter से chunks    │
│    2) Embedding: sentence-transformers से dense vectors     │
│    3) Indexing: FAISS (Facebook AI Similarity Search)       │
│  Output: FAISS index + metadata + vector shape              │
└─────────────────────────────────────────────────────────────┘
```

---

## Dependencies: क्या और क्यों

### REQUIRED Libraries

```
langchain-text-splitters   # Text chunking (paragraph boundaries preserve)
faiss-cpu                  # Vector similarity search index
sentence-transformers      # Dense embeddings (all-MiniLM-L6-v2 model)
pypdf                      # PDF text extraction
numpy                      # Embedding matrix operations
```

### ADDITIONAL (हमने add किए)

```
pytesseract + pdf2image    # OCR fallback (scanned PDFs के लिए)
Pillow                     # Image processing (pdf2image dependency)
```

### TXT Support के लिए

**कोई extra dependency नहीं!**

- TXT files plain text हैं
- Python में built-in `Path.read_text()` से read होते हैं
- कोई library import नहीं चाहिए

```python
# बस यह काफी है:
txt_content = txt_path.read_text(encoding="utf-8")
```

---

## Code Structure

### Phase 1: `scripts/pdf_loader.py`

**क्या करता है:**
- PDF और TXT दोनों को discover करता है
- Text extract करता है
- Output files में save करता है

**Key Functions:**

```python
def discover_documents(input_path: Path) -> list[Path]:
    """
    PDF और TXT दोनों को recursively discover करता है।
    
    Example:
        discover_documents(Path("data"))
        # Returns: [PDF files] + [TXT files] (sorted)
    """
    if input_path.is_file() and input_path.suffix.lower() in {".pdf", ".txt"}:
        return [input_path]
    if input_path.is_dir():
        pdf_files = list(input_path.rglob("*.pdf"))
        txt_files = list(input_path.rglob("*.txt"))
        return sorted(pdf_files + txt_files)
    return []

def extract_pdf_text(pdf_path: Path) -> str:
    """PDF से page-wise text निकालता है।"""
    reader = PdfReader(str(pdf_path))
    page_sections: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_sections.append(f"--- Page {page_number} ---\n{page_text.strip()}\n")
    return "\n".join(page_sections).strip()

def extract_txt_text(txt_path: Path) -> str:
    """TXT से pure text मिलता है (कोई page separator नहीं)।"""
    return txt_path.read_text(encoding="utf-8").strip()

def main():
    # सब PDFs/TXTs को process करो
    documents = discover_documents(input_path)
    for source_file in documents:
        if source_file.suffix.lower() == ".pdf":
            text = extract_pdf_text(source_file)
        else:  # .txt
            text = extract_txt_text(source_file)
        write_extracted_text(output_dir, source_file, text)
```

---

### Phase 2: `scripts/document_pipeline.py`

**क्या करता है:**
- Phase 1 के extracted content को ingest करता है
- Small semantic chunks में split करता है
- Dense embeddings generate करता है
- FAISS vector index बनाता है

**Key Functions:**

```python
def discover_documents(input_path: Path) -> list[Path]:
    """
    Phase 1 जैसे ही PDF/TXT discover करता है।
    लेकिन यहाँ directly original files process होती हैं।
    """

def load_pdf_pages(pdf_path: Path, use_ocr_fallback: bool, ...) -> list[tuple[int, str]]:
    """
    PDF को page-wise text में convert करता है।
    [ADDITIONAL] scanned PDFs के लिए OCR fallback support.
    
    Returns: [(page_number, page_text), ...]
    """

def load_txt_pages(txt_path: Path) -> list[tuple[int, str]]:
    """
    TXT को pseudo-page (1, full_text) में convert करता है।
    TXT files में "pages" नहीं होते, सब एक block है।
    
    Returns: [(1, full_txt_content)]
    """

def chunk_documents(document_files: list[Path], ...) -> list[ChunkRecord]:
    """
    Chunking को unified करता है:
    - PDF pages अलग हो सकते हैं
    - TXT एक ही "page" 1 होता है
    
    Chunking logic दोनों के लिए same है।
    """
    for source_file in document_files:
        if source_file.suffix.lower() == ".pdf":
            page_entries = load_pdf_pages(source_file, ...)
        else:
            page_entries = load_txt_pages(source_file)
        
        # अब दोनों को same format में chunks करो
        for page_number, page_text in page_entries:
            chunks = splitter.split_text(page_text)
            for chunk_text in chunks:
                chunk_records.append(ChunkRecord(...))
```

---

## Runner Scripts

### `run_phase1.sh` — Phase 1 Runner

```bash
# Default (data folder से सब PDFs/TXTs process करो)
bash run_phase1.sh

# Specific file या folder
bash run_phase1.sh data/tesla.txt outputs
bash run_phase1.sh data/pdfs outputs
bash run_phase1.sh data outputs  # Mixed PDFs + TXTs

# First-time install
bash run_phase1.sh --install-deps
```

### `run_phase2.sh` — Phase 2 Runner (Unified)

```bash
# Default (सब documents को एक vector_store में index करो)
bash run_phase2.sh

# Specific input/output
bash run_phase2.sh data outputs/vector_store
bash run_phase2.sh data/pdfs outputs/vector_store

# Custom tuning
bash run_phase2.sh data outputs/vector_store 600 100 sentence-transformers/all-MiniLM-L6-v2

# First-time install
bash run_phase2.sh --install-deps
```

---

## Complete Workflow

### 1. Setup (One-time)

```bash
# Environment बनाओ
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# System OCR packages (optional, scanned PDFs के लिए)
sudo apt-get install -y tesseract-ocr poppler-utils
```

### 2. Add Documents

```bash
# Place files in data/ folder:
# data/pdfs/lecture1.pdf
# data/pdfs/lecture2.pdf
# data/tesla.txt
```

### 3. Run Everything (Unified)

```bash
# Phase 1 + Phase 2 एक ही vector_store में
bash run_phase1.sh data outputs
bash run_phase2.sh data outputs/vector_store
```

**Output:**
```
outputs/
├── phase1_extracted/
│   ├── lecture1.txt (PDF से)
│   ├── lecture2.txt (PDF से)
│   └── tesla.txt (copy from TXT input)
│
└── vector_store/ (Unified)
    ├── index.faiss (सब documents का combined index)
    ├── metadata.json (सब chunks की metadata)
    └── vectors_shape.json (embedding stats)
```

---

## Features & Enhancements

### Original (Required)

| Feature | Status | File |
|---------|--------|------|
| PDF extraction | ✅ Included | `scripts/pdf_loader.py` |
| Text chunking | ✅ Included | `scripts/document_pipeline.py` |
| Embedding generation | ✅ Included | `scripts/document_pipeline.py` |
| FAISS indexing | ✅ Included | `scripts/document_pipeline.py` |

### Our Enhancements

| Feature | Status | File | Why Not Original |
|---------|--------|------|-------------------|
| **TXT support** | ✅ Added | Both scripts | Original was PDF-only |
| **OCR fallback** | ✅ Added | Phase 2 | Scanned PDF support |
| **Bilingual OCR** | ✅ Added | Phase 2 | Hindi + English support |

---

## Common Use Cases

### Use Case 1: Only PDFs

```bash
# सब PDFs को folder में रखो
cp lecture*.pdf data/pdfs/

# Run करो
bash run_phase2.sh data/pdfs outputs/vector_store
```

### Use Case 2: Only TXT Files

```bash
# TXT files folder में रखो
cp notes*.txt data/

# Run करो
bash run_phase2.sh data outputs/vector_store
```

### Use Case 3: Mixed (PDFs + TXT)

```bash
# Both formats में files हों:
# data/pdfs/lecture.pdf
# data/notes.txt
# data/summary.txt

bash run_phase2.sh data outputs/vector_store
```

---

## Output Artifacts Explained

### 1. `index.faiss`

**क्या है:** Binary FAISS index file

```python
# Usage (Phase 3 में):
import faiss
index = faiss.read_index("outputs/vector_store/index.faiss")
distances, indices = index.search(query_vector, k=5)
```

### 2. `metadata.json`

**क्या है:** सब chunks की metadata

```json
[
  {
    "chunk_id": 0,
    "source_file": "tesla.txt",
    "page_number": 1,
    "text": "Tesla cars are a range of electric vehicles..."
  }
]
```

### 3. `vectors_shape.json`

**क्या है:** Embedding dimension info

```json
{
  "total_vectors": 24,
  "embedding_dimension": 384
}
```

---

## Dependency Notes

### Why TXT Needed NO New Libraries

```python
# Pure Python, no external library:
content = Path("file.txt").read_text(encoding="utf-8")

# Standard library functions used:
# - pathlib.Path (built-in)
# - str.read_text() (built-in)
# - str.write_text() (built-in)
```

### Why PDF Needs `pypdf`

```python
# pypdf is required because:
from pypdf import PdfReader  # No built-in PDF support in Python

reader = PdfReader("file.pdf")
for page in reader.pages:
    text = page.extract_text()  # pypdf handles PDF parsing/rendering
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| "No supported file (.pdf/.txt) found" | Empty input folder | Add PDFs/TXTs to data/ folder |
| "No chunks were created" | Empty documents | Check if files हैं text-extractable |
| OCR not working | tesseract/poppler not installed | `sudo apt-get install tesseract-ocr poppler-utils` |
| Memory issues | Large FAISS index | Reduce chunk_size or use CPU-only mode |

---

## File Structure Reference

```
Student-Academic-Knowledge-Assistant/
├── data/
│   ├── pdfs/                    # PDF files रखने के लिए
│   ├── tesla.txt                # TXT files यहाँ
│   └── (कोई भी PDF/TXT)
│
├── scripts/
│   ├── pdf_loader.py            # Phase 1 (PDF + TXT support)
│   └── document_pipeline.py      # Phase 2 (PDF + TXT support + OCR)
│
├── outputs/
│   ├── phase1_extracted/        # Phase 1 का output
│   └── vector_store/            # Phase 2 का unified output
│
├── requirements.txt             # सब dependencies
├── run_phase1.sh               # Phase 1 runner
├── run_phase2.sh               # Phase 2 runner
├── README.md                   # Quick start guide
└── DOCUMENTATION.md            # यह file (detailed guide)
```

---

## Summary: What Changed

| Aspect | Before | After |
|--------|--------|-------|
| Input formats | PDFs only | PDFs + TXT |
| Text extraction | pypdf (PDF) | pypdf (PDF) + native Python (TXT) |
| Dependencies added | 0 for TXT | 0 for TXT (pypdf same) |
| Output folder | Single | Single (unified) |
| Runner scripts | Separate | Can be called independent or sequential |

**Main Change:** Input pipeline को flexible बनाया ताकि दोनों formats support हों without extra complexity.

---

## Next Steps (Phase 3)

Future में query/retrieval script add कर सकते हैं:

```python
# Phase 3 example structure:
def query_index(query_text, vector_store_path, k=5):
    # 1) Query को embedding में convert करो
    # 2) FAISS index से similarity search करो
    # 3) Metadata से matching chunks निकालो
    # 4) LLM को context + query दो
    # 5) Answer generate करो (source citation के साथ)
```

---

**अब आप सब कुछ समझ गए हो! 🚀**
