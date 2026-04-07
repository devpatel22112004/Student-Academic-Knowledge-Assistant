#!/usr/bin/env python3

"""
Phase 2 pipeline.
This script reads supported documents, chunks text, creates embeddings,
builds FAISS index, and saves retrieval artifacts.
"""

# Keep modern typing behavior consistent across Python versions.
from __future__ import annotations

# Parse command-line arguments.
import argparse
# Lazy import for optional format parsers.
import importlib
# Save/read metadata JSON files.
import json
# Control noisy runtime logs.
import logging
# Configure environment-level logging behavior.
import os
# Check system binary availability (.doc support).
import shutil
# Run external command for legacy .doc extraction.
import subprocess
# Dataclass model for chunk metadata.
from dataclasses import asdict, dataclass
# Path-safe file and directory handling.
from pathlib import Path

# Vector index engine.
import faiss
# Numeric arrays for embeddings.
import numpy as np
# PDF parser library.
from pypdf import PdfReader
# Embedding model wrapper.
from sentence_transformers import SentenceTransformer
# Text splitter used for chunking.
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkRecord:
    # Unique chunk identifier.
    chunk_id: int
    # Source document file name.
    source_file: str
    # Page number (or pseudo-page 1 for text docs).
    page_number: int
    # Chunk text content.
    text: str


def discover_documents(input_path: Path) -> list[Path]:
    # Allowed extensions for Phase 2 input.
    supported = {".pdf", ".txt", ".rtf", ".doc", ".docx"}

    # If input is one supported file, return it directly.
    if input_path.is_file() and input_path.suffix.lower() in supported:
        return [input_path]

    # If input is folder, collect supported files recursively.
    if input_path.is_dir():
        # Store matched files.
        matched: list[Path] = []
        # Search by each extension.
        for ext in supported:
            matched.extend(input_path.rglob(f"*{ext}"))
        # Keep output deterministic.
        return sorted(matched)

    # Return empty for unsupported/missing path.
    return []


def load_pdf_pages(
    pdf_path: Path,
    use_ocr_fallback: bool,
    ocr_dpi: int,
    ocr_lang: str,
) -> list[tuple[int, str]]:
    # Open PDF file.
    reader = PdfReader(str(pdf_path))
    # Store extracted page tuples.
    pages: list[tuple[int, str]] = []

    # Read each page with 1-based page numbering.
    for page_number, page in enumerate(reader.pages, start=1):
        # Try direct text extraction from page.
        page_text = (page.extract_text() or "").strip()

        # If page has no selectable text, optionally run OCR fallback.
        if not page_text and use_ocr_fallback:
            page_text = run_ocr_on_page(
                pdf_path=pdf_path,
                page_number=page_number,
                ocr_dpi=ocr_dpi,
                ocr_lang=ocr_lang,
            )

        # Add page only if text is available.
        if page_text:
            pages.append((page_number, page_text))

    # Return PDF pages as text tuples.
    return pages


def load_txt_pages(txt_path: Path) -> list[tuple[int, str]]:
    # Read TXT content.
    txt_content = txt_path.read_text(encoding="utf-8").strip()
    # Return empty if file has no usable text.
    if not txt_content:
        return []
    # Return pseudo-page format for unified downstream flow.
    return [(1, txt_content)]


def load_docx_pages(docx_path: Path) -> list[tuple[int, str]]:
    # Import DOCX parser lazily.
    try:
        docx_module = importlib.import_module("docx")
        docx_document = docx_module.Document
    except Exception:
        raise RuntimeError("DOCX support missing. Install dependency: python-docx")

    # Open DOCX file.
    doc = docx_document(str(docx_path))
    # Combine non-empty paragraph lines.
    text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()).strip()
    # Return empty if no text extracted.
    if not text:
        return []
    # Return pseudo-page format.
    return [(1, text)]


def load_rtf_pages(rtf_path: Path) -> list[tuple[int, str]]:
    # Import RTF converter lazily.
    try:
        striprtf_module = importlib.import_module("striprtf.striprtf")
        convert_rtf = striprtf_module.rtf_to_text
    except Exception:
        raise RuntimeError("RTF support missing. Install dependency: striprtf")

    # Read raw RTF content.
    raw = rtf_path.read_text(encoding="utf-8", errors="ignore")
    # Convert markup to plain text.
    text = convert_rtf(raw).replace("\x00", "").strip()
    # Return empty if no text extracted.
    if not text:
        return []
    # Return pseudo-page format.
    return [(1, text)]


def load_doc_pages(doc_path: Path) -> list[tuple[int, str]]:
    # Ensure antiword binary exists for .doc parsing.
    if shutil.which("antiword") is None:
        raise RuntimeError(
            "DOC support needs system tool 'antiword'. Install with: sudo apt-get install -y antiword"
        )

    # Execute antiword and capture extracted text.
    result = subprocess.run(
        ["antiword", str(doc_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    # Normalize command output.
    text = (result.stdout or "").strip()
    # Return empty if no text extracted.
    if not text:
        return []
    # Return pseudo-page format.
    return [(1, text)]


def chunk_documents(
    document_files: list[Path],
    chunk_size: int,
    chunk_overlap: int,
    use_ocr_fallback: bool,
    ocr_dpi: int,
    ocr_lang: str,
) -> list[ChunkRecord]:
    # Create splitter with configured size/overlap.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Store final chunk metadata rows.
    chunk_records: list[ChunkRecord] = []
    # Global chunk counter.
    chunk_id = 0

    # Process each document file.
    for source_file in document_files:
        # Normalize extension for routing.
        suffix = source_file.suffix.lower()
        # PDF loader path.
        if suffix == ".pdf":
            page_entries = load_pdf_pages(
                pdf_path=source_file,
                use_ocr_fallback=use_ocr_fallback,
                ocr_dpi=ocr_dpi,
                ocr_lang=ocr_lang,
            )
        # TXT loader path.
        elif suffix == ".txt":
            page_entries = load_txt_pages(source_file)
        # DOCX loader path.
        elif suffix == ".docx":
            page_entries = load_docx_pages(source_file)
        # RTF loader path.
        elif suffix == ".rtf":
            page_entries = load_rtf_pages(source_file)
        # DOC loader path.
        else:
            page_entries = load_doc_pages(source_file)

        # Split every page/text block into chunks.
        for page_number, page_text in page_entries:
            chunks = splitter.split_text(page_text)
            for chunk_text in chunks:
                # Trim whitespace-only chunks.
                normalized_text = chunk_text.strip()
                # Remove accidental null bytes from source text.
                normalized_text = normalized_text.replace("\x00", "")
                if not normalized_text:
                    continue

                # Save metadata row for each valid chunk.
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        source_file=source_file.name,
                        page_number=page_number,
                        text=normalized_text,
                    )
                )
                # Increment global chunk id.
                chunk_id += 1

    # Return all chunk rows.
    return chunk_records


def create_embeddings(chunks: list[ChunkRecord], model_name: str, show_progress: bool) -> np.ndarray:
    # Stop early if no chunks are available.
    if not chunks:
        raise ValueError(
            "No chunks were created. Input files may be empty or non-extractable. "
            "(HINT: Use OCR fallback — install pytesseract + pdf2image + system tesseract/poppler) "
            "or provide valid text-based PDF/TXT files."
        )

    # Load sentence-transformer model.
    model = SentenceTransformer(model_name)
    # Build text list from chunk records.
    texts = [chunk.text for chunk in chunks]
    # Encode text list to vector matrix.
    vectors = model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)

    # Convert to float32 for FAISS compatibility.
    return np.asarray(vectors, dtype=np.float32)


def configure_runtime_logs(verbose: bool) -> None:
    # Leave logs untouched in verbose mode.
    if verbose:
        return

    # Disable noisy progress/log env flags.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Lower log level for common libs.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    # Try optional huggingface logger controls.
    try:
        from huggingface_hub.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    # Try optional transformers logger controls.
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    # Detect embedding dimension from matrix shape.
    embedding_dimension = vectors.shape[1]
    # Create L2 distance FAISS index.
    index = faiss.IndexFlatL2(embedding_dimension)
    # Add all vectors to index.
    index.add(vectors)
    # Return in-memory index object.
    return index


def save_artifacts(output_dir: Path, index: faiss.IndexFlatL2, chunks: list[ChunkRecord], vectors: np.ndarray) -> None:
    # Create output folder if needed.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index file.
    faiss.write_index(index, str(output_dir / "index.faiss"))

    # Save chunk metadata JSON.
    (output_dir / "metadata.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Save vector matrix summary JSON.
    (output_dir / "vectors_shape.json").write_text(
        json.dumps(
            {
                "total_vectors": int(vectors.shape[0]),
                "embedding_dimension": int(vectors.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    # Create argument parser.
    parser = argparse.ArgumentParser(
        description="Phase 2 pipeline: chunk text, generate embeddings, and index in FAISS."
    )

    # Required input path argument.
    parser.add_argument(
        "--input",
        required=True,
        help="PDF/TXT/RTF/DOC/DOCX file path or directory containing documents.",
    )

    # Optional output folder argument.
    parser.add_argument(
        "--output",
        default="outputs/vector_store",
        help="Output folder where index and metadata files are saved.",
    )

    # Chunking configuration arguments.
    parser.add_argument("--chunk-size", type=int, default=600, help="Max characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap characters between chunks.")

    # Embedding model argument.
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    # Verbose logging switch.
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logs ON (default me warnings/info reduce kiye jaate hain).",
    )
    # Embedding progress bar switch.
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Embedding progress bar dikhana ho to use karein.",
    )

    # OCR disable switch.
    parser.add_argument(
        "--disable-ocr-fallback",
        action="store_true",
        help="[ADDITIONAL] OCR fallback ko disable karein (sirf selectable PDF text use karega).",
    )
    # OCR DPI tuning argument.
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=300,
        help="[ADDITIONAL] OCR render DPI (higher = better accuracy, slower). Default: 300.",
    )
    # OCR language argument.
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="[ADDITIONAL] Tesseract OCR language code (default: eng, use 'eng+hin' for bilingual).",
    )
    # Return parsed args.
    return parser.parse_args()


def main() -> None:
    # Read CLI arguments.
    args = parse_args()

    # Configure runtime logs.
    configure_runtime_logs(verbose=args.verbose)

    # Convert CLI strings to Path objects.
    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Discover supported files from input path.
    document_files = discover_documents(input_path)
    if not document_files:
        raise FileNotFoundError(
            f"No supported file (.pdf/.txt/.rtf/.doc/.docx) found at: {input_path}"
        )

    # Print discovered file count.
    print(f"Found {len(document_files)} document file(s).")

    # Build chunk records from all documents.
    chunks = chunk_documents(
        document_files=document_files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_ocr_fallback=not args.disable_ocr_fallback,
        ocr_dpi=args.ocr_dpi,
        ocr_lang=args.ocr_lang,
    )
    # Print chunk count.
    print(f"Created {len(chunks)} chunk(s).")

    # Convert chunks to embeddings.
    vectors = create_embeddings(
        chunks=chunks,
        model_name=args.model,
        show_progress=args.show_progress,
    )
    # Print vector matrix shape.
    print(f"Generated embeddings with shape: {vectors.shape}")

    # Create FAISS index from vectors.
    index = build_faiss_index(vectors=vectors)
    # Save index + metadata artifacts.
    save_artifacts(output_dir=output_dir, index=index, chunks=chunks, vectors=vectors)

    # Print completion message.
    print(f"Indexing completed. Artifacts saved in: {output_dir}")


# ---------------------------------------------------------------------------
# ADDITIONAL OCR SECTION (OPTIONAL)
#
# Why this exists:
# - Some PDFs are scanned images and return empty text via pypdf.
# - OCR converts that page image into machine-readable text.
#
# How it works:
# 1) Render one PDF page to image using pdf2image.
# 2) Read image text using pytesseract.
# 3) Return extracted text to page loader fallback path.
#
# Required extras for OCR:
# - Python packages: pytesseract, pdf2image, Pillow
# - System packages: tesseract-ocr, poppler-utils
# ---------------------------------------------------------------------------
def run_ocr_on_page(pdf_path: Path, page_number: int, ocr_dpi: int, ocr_lang: str) -> str:
    """[ADDITIONAL] OCR fallback for a single PDF page."""
    try:
        # Import OCR-related modules lazily.
        pdf2image_module = importlib.import_module("pdf2image")
        pytesseract_module = importlib.import_module("pytesseract")
        # Resolve converter function.
        convert_from_path = pdf2image_module.convert_from_path
    except Exception:
        # If OCR deps are unavailable, return empty text.
        return ""

    try:
        # Render one PDF page into image list.
        page_images = convert_from_path(
            str(pdf_path),
            dpi=ocr_dpi,
            first_page=page_number,
            last_page=page_number,
            fmt="png",
        )
        # Return empty if image rendering failed.
        if not page_images:
            return ""

        # Run OCR on the first rendered page image.
        return (pytesseract_module.image_to_string(page_images[0], lang=ocr_lang) or "").strip()
    except Exception:
        # Return empty on OCR runtime error.
        return ""


if __name__ == "__main__":
    main()