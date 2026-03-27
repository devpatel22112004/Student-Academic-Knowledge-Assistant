#!/usr/bin/env python3

"""
Phase 2 pipeline: discover documents, create chunks, generate embeddings,
build FAISS index, and save metadata artifacts.

Core inputs: PDF/TXT files
Core outputs: index.faiss, metadata.json, vectors_shape.json

Note: OCR support is optional and implemented in the ADDITIONAL section at
the end of this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkRecord:
    # Metadata saved for each chunk.
    chunk_id: int
    source_file: str
    page_number: int
    text: str


def discover_documents(input_path: Path) -> list[Path]:
    # Discover supported input files.
    if input_path.is_file() and input_path.suffix.lower() in {".pdf", ".txt"}:
        return [input_path]

    # Support recursive folder ingestion.
    if input_path.is_dir():
        pdf_files = list(input_path.rglob("*.pdf"))
        txt_files = list(input_path.rglob("*.txt"))
        return sorted(pdf_files + txt_files)

    return []


def load_pdf_pages(
    pdf_path: Path,
    use_ocr_fallback: bool,
    ocr_dpi: int,
    ocr_lang: str,
) -> list[tuple[int, str]]:
    # Load PDF content as (page_number, text) tuples.
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        # Use direct text extraction first.
        page_text = (page.extract_text() or "").strip()

        # ADDITIONAL: OCR fallback for scanned pages.
        if not page_text and use_ocr_fallback:
            page_text = run_ocr_on_page(
                pdf_path=pdf_path,
                page_number=page_number,
                ocr_dpi=ocr_dpi,
                ocr_lang=ocr_lang,
            )

        if page_text:
            pages.append((page_number, page_text))

    return pages


def load_txt_pages(txt_path: Path) -> list[tuple[int, str]]:
    # Map TXT into pseudo-page format for shared chunking logic.
    txt_content = txt_path.read_text(encoding="utf-8").strip()
    if not txt_content:
        return []
    return [(1, txt_content)]


def chunk_documents(
    document_files: list[Path],
    chunk_size: int,
    chunk_overlap: int,
    use_ocr_fallback: bool,
    ocr_dpi: int,
    ocr_lang: str,
) -> list[ChunkRecord]:
    # Split text into retrieval-ready chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunk_records: list[ChunkRecord] = []
    chunk_id = 0

    # Process each document into chunk records.
    for source_file in document_files:
        if source_file.suffix.lower() == ".pdf":
            page_entries = load_pdf_pages(
                pdf_path=source_file,
                use_ocr_fallback=use_ocr_fallback,
                ocr_dpi=ocr_dpi,
                ocr_lang=ocr_lang,
            )
        else:
            page_entries = load_txt_pages(source_file)

        for page_number, page_text in page_entries:
            chunks = splitter.split_text(page_text)
            for chunk_text in chunks:
                normalized_text = chunk_text.strip()
                if not normalized_text:
                    continue

                # Preserve source info for traceability.
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        source_file=source_file.name,
                        page_number=page_number,
                        text=normalized_text,
                    )
                )
                chunk_id += 1

    return chunk_records


def create_embeddings(chunks: list[ChunkRecord], model_name: str, show_progress: bool) -> np.ndarray:
    # Guard against empty chunk list.
    if not chunks:
        raise ValueError(
            "No chunks were created. Input files may be empty or non-extractable. "
            "(HINT: Use OCR fallback — install pytesseract + pdf2image + system tesseract/poppler) "
            "or provide valid text-based PDF/TXT files."
        )

    # Convert chunk text to dense vectors.
    model = SentenceTransformer(model_name)
    texts = [chunk.text for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)

    # Keep FAISS-compatible dtype.
    return np.asarray(vectors, dtype=np.float32)


def configure_runtime_logs(verbose: bool) -> None:
    # Keep default CLI output clean unless verbose mode is requested.
    if verbose:
        return

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    try:
        from huggingface_hub.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    # Build an L2 FAISS index from embedding matrix.
    embedding_dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(vectors)
    return index


def save_artifacts(output_dir: Path, index: faiss.IndexFlatL2, chunks: list[ChunkRecord], vectors: np.ndarray) -> None:
    # Ensure output folder exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Persist FAISS index.
    faiss.write_index(index, str(output_dir / "index.faiss"))

    # 2) Persist chunk metadata.
    (output_dir / "metadata.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 3) Persist vector shape summary.
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
    # Parse runtime options.
    parser = argparse.ArgumentParser(
        description="Phase 2 pipeline: chunk text, generate embeddings, and index in FAISS."
    )

    # Required input path.
    parser.add_argument("--input", required=True, help="PDF/TXT file path or directory containing documents.")

    # Output artifact folder.
    parser.add_argument(
        "--output",
        default="outputs/vector_store",
        help="Output folder where index and metadata files are saved.",
    )

    # Chunking controls.
    parser.add_argument("--chunk-size", type=int, default=600, help="Max characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap characters between chunks.")

    # Embedding model.
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logs ON (default me warnings/info reduce kiye jaate hain).",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Embedding progress bar dikhana ho to use karein.",
    )

    # ADDITIONAL (OCR): Optional fallback controls.
    parser.add_argument(
        "--disable-ocr-fallback",
        action="store_true",
        help="[ADDITIONAL] OCR fallback ko disable karein (sirf selectable PDF text use karega).",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=300,
        help="[ADDITIONAL] OCR render DPI (higher = better accuracy, slower). Default: 300.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="[ADDITIONAL] Tesseract OCR language code (default: eng, use 'eng+hin' for bilingual).",
    )
    return parser.parse_args()


def main() -> None:
    # Parse args.
    args = parse_args()

    # Configure logging.
    configure_runtime_logs(verbose=args.verbose)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Discover input documents.
    document_files = discover_documents(input_path)
    if not document_files:
        raise FileNotFoundError(f"No supported file (.pdf/.txt) found at: {input_path}")

    print(f"Found {len(document_files)} document file(s).")

    # Create chunks.
    chunks = chunk_documents(
        document_files=document_files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_ocr_fallback=not args.disable_ocr_fallback,
        ocr_dpi=args.ocr_dpi,
        ocr_lang=args.ocr_lang,
    )
    print(f"Created {len(chunks)} chunk(s).")

    # Generate embeddings.
    vectors = create_embeddings(
        chunks=chunks,
        model_name=args.model,
        show_progress=args.show_progress,
    )
    print(f"Generated embeddings with shape: {vectors.shape}")

    # Build index and save artifacts.
    index = build_faiss_index(vectors=vectors)
    save_artifacts(output_dir=output_dir, index=index, chunks=chunks, vectors=vectors)

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
        import importlib

        pdf2image_module = importlib.import_module("pdf2image")
        pytesseract_module = importlib.import_module("pytesseract")
        convert_from_path = pdf2image_module.convert_from_path
    except Exception:
        return ""

    try:
        page_images = convert_from_path(
            str(pdf_path),
            dpi=ocr_dpi,
            first_page=page_number,
            last_page=page_number,
            fmt="png",
        )
        if not page_images:
            return ""

        return (pytesseract_module.image_to_string(page_images[0], lang=ocr_lang) or "").strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()