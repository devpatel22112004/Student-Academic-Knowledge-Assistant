from __future__ import annotations

# Phase 2 — Document Processing Pipeline
# Kaam: PDFs ko chunk karo, embeddings banao, aur FAISS vector DB me store karo.
# Ye script indexing step complete karta hai jo retrieval ke liye base banata hai.

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# FAISS: fast vector search index ke liye
import faiss
# NumPy: embeddings ko float32 matrix me convert/store karne ke liye
import numpy as np
# pypdf: PDF se text read karne ke liye
from pypdf import PdfReader
# sentence-transformers: text -> embedding vector conversion
from sentence_transformers import SentenceTransformer

# Text splitter ke liye dedicated lightweight package use kar rahe hain.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== ADDITIONAL (NOT REQUIRED): OCR Fallback for Scanned PDFs =====
# Ye section humne add kiya hai — original project me required nahi tha.
# Scanned/image-based PDFs ke liye automatic OCR fallback provide karta hai.
try:
    from pdf2image import convert_from_path
    import pytesseract
except Exception:
    convert_from_path = None
    pytesseract = None
# =======================================================================


@dataclass
class ChunkRecord:
    # Har chunk ke saath yeh metadata store hoga.
    # Isse baad me answer ka source page/file trace kar paoge.
    chunk_id: int
    source_file: str
    page_number: int
    text: str


def discover_documents(input_path: Path) -> list[Path]:
    # Agar input direct supported file hai to usko list me return karo.
    if input_path.is_file() and input_path.suffix.lower() in {".pdf", ".txt"}:
        return [input_path]
    # Agar input folder hai to us folder (aur subfolders) me sab PDFs/TXTs dhundo.
    if input_path.is_dir():
        pdf_files = list(input_path.rglob("*.pdf"))
        txt_files = list(input_path.rglob("*.txt"))
        return sorted(pdf_files + txt_files)
    # Invalid path case me empty list return.
    return []


# ADDITIONAL (NOT REQUIRED): OCR Fallback Function
# Humne ye function add kiya hai scanned PDFs handle karne ke liye.
def run_ocr_on_page(pdf_path: Path, page_number: int, ocr_dpi: int, ocr_lang: str) -> str:
    """ADDITIONAL: Extract text from scanned PDF page using Tesseract OCR."""
    if convert_from_path is None or pytesseract is None:
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

        ocr_text = (pytesseract.image_to_string(page_images[0], lang=ocr_lang) or "").strip()
        return ocr_text
    except Exception:
        return ""


def load_pdf_pages(
    pdf_path: Path,
    use_ocr_fallback: bool,
    ocr_dpi: int,
    ocr_lang: str,
) -> list[tuple[int, str]]:
    # PDF open karke page-wise text nikaalte hain.
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        # Empty page text ko skip karne ke liye strip + check.
        page_text = (page.extract_text() or "").strip()
        # ADDITIONAL (NOT REQUIRED): OCR fallback for pages without selectable text
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
    # TXT ke liye ek pseudo page (1) treat karte hain.
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
    # Recursive splitter semantic boundary preserve karte hue chunks banata hai.
    # chunk_overlap context continuity maintain karta hai.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunk_records: list[ChunkRecord] = []
    chunk_id = 0

    # Har document -> page/text block -> multiple chunks.
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

                # Chunk ke saath source metadata save karte hain.
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
    # Agar chunking se kuch output hi nahi aaya to indexing possible nahi.
    if not chunks:
        raise ValueError(
            "No chunks were created. Input files may be empty or non-extractable. "
            "(HINT: Use OCR fallback — install pytesseract + pdf2image + system tesseract/poppler) "
            "or provide valid text-based PDF/TXT files."
        )

    # Embedding model load karo (default: all-MiniLM-L6-v2).
    model = SentenceTransformer(model_name)
    # Sirf text list pass karte hain embedding generation ke liye.
    texts = [chunk.text for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
    # FAISS ke liye float32 recommended format.
    return np.asarray(vectors, dtype=np.float32)


def configure_runtime_logs(verbose: bool) -> None:
    # Quiet mode: unnecessary terminal warnings/info ko suppress karta hai.
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
    # Embedding dimension detect karke L2 distance index banate hain.
    embedding_dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    # Saare vectors index me add karo.
    index.add(vectors)
    return index


def save_artifacts(output_dir: Path, index: faiss.IndexFlatL2, chunks: list[ChunkRecord], vectors: np.ndarray) -> None:
    # Output folder create/ensure.
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Vector index save (binary FAISS format)
    faiss.write_index(index, str(output_dir / "index.faiss"))

    # 2) Metadata save (chunk-level traceability)
    (output_dir / "metadata.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 3) Quick summary save (sanity check ke liye)
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
    # CLI arguments define karte hain taki script reusable ho.
    parser = argparse.ArgumentParser(
        description="Phase 2 pipeline: chunk text, generate embeddings, and index in FAISS."
    )
    # Input: single document file ya directory.
    parser.add_argument("--input", required=True, help="PDF/TXT file path or directory containing documents.")
    # Output: index + metadata kaha save karna hai.
    parser.add_argument(
        "--output",
        default="outputs/vector_store",
        help="Output folder where index and metadata files are saved.",
    )
    # Chunk tuning params.
    parser.add_argument("--chunk-size", type=int, default=600, help="Max characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap characters between chunks.")
    # Embedding model selection.
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
    # ADDITIONAL (NOT REQUIRED): OCR-related CLI arguments added by us
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
    # 1) Args parse
    args = parse_args()

    # 1.1) Runtime log behavior set karo (default: quiet/clean output)
    configure_runtime_logs(verbose=args.verbose)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # 2) Input document discovery
    document_files = discover_documents(input_path)
    if not document_files:
        raise FileNotFoundError(f"No supported file (.pdf/.txt) found at: {input_path}")

    print(f"Found {len(document_files)} document file(s).")

    # 3) Chunking
    chunks = chunk_documents(
        document_files=document_files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_ocr_fallback=not args.disable_ocr_fallback,
        ocr_dpi=args.ocr_dpi,
        ocr_lang=args.ocr_lang,
    )
    print(f"Created {len(chunks)} chunk(s).")

    # 4) Embedding generation
    vectors = create_embeddings(
        chunks=chunks,
        model_name=args.model,
        show_progress=args.show_progress,
    )
    print(f"Generated embeddings with shape: {vectors.shape}")

    # 5) Index build + artifact save
    index = build_faiss_index(vectors=vectors)
    save_artifacts(output_dir=output_dir, index=index, chunks=chunks, vectors=vectors)

    print(f"Indexing completed. Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    # Script entry point
    main()