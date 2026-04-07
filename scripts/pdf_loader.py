#!/usr/bin/env python3

"""
Phase 1 pipeline: discover PDF/TXT documents and normalize them into text files.

PDF is extracted page-wise with markers.
TXT is read directly with Python built-ins.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
from pathlib import Path

# PDF parser (used only for .pdf files).
from pypdf import PdfReader

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF and keep page markers for traceability."""
    reader = PdfReader(str(pdf_path))
    page_sections: list[str] = []

    # Extract page by page for clean source mapping.
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_sections.append(f"--- Page {page_number} ---\n{page_text.strip()}\n")

    return "\n".join(page_sections).strip()


def extract_txt_text(txt_path: Path) -> str:
    """Read UTF-8 text from TXT file."""
    return txt_path.read_text(encoding="utf-8").strip()


def extract_docx_text(docx_path: Path) -> str:
    """Extract plain text from DOCX file."""
    try:
        docx_module = importlib.import_module("docx")
        docx_document = docx_module.Document
    except Exception:
        raise RuntimeError("DOCX support missing. Install dependency: python-docx")

    doc = docx_document(str(docx_path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts).strip()


def extract_rtf_text(rtf_path: Path) -> str:
    """Extract plain text from RTF file."""
    try:
        striprtf_module = importlib.import_module("striprtf.striprtf")
        convert_rtf = striprtf_module.rtf_to_text
    except Exception:
        raise RuntimeError("RTF support missing. Install dependency: striprtf")

    raw = rtf_path.read_text(encoding="utf-8", errors="ignore")
    return convert_rtf(raw).strip()


def extract_doc_text(doc_path: Path) -> str:
    """Extract text from legacy DOC file using antiword binary."""
    if shutil.which("antiword") is None:
        raise RuntimeError(
            "DOC support needs system tool 'antiword'. Install with: sudo apt-get install -y antiword"
        )

    result = subprocess.run(
        ["antiword", str(doc_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return (result.stdout or "").strip()


def discover_documents(input_path: Path) -> list[Path]:
    """Return sorted list of supported input files."""
    supported = {".pdf", ".txt", ".rtf", ".doc", ".docx"}

    if input_path.is_file() and input_path.suffix.lower() in supported:
        return [input_path]

    # Recursive discovery for folder input.
    if input_path.is_dir():
        matched: list[Path] = []
        for ext in supported:
            matched.extend(input_path.rglob(f"*{ext}"))
        return sorted(matched)

    return []


def ensure_output_dir(output_dir: Path) -> None:
    """Create output directory when missing."""
    output_dir.mkdir(parents=True, exist_ok=True)


def write_extracted_text(output_dir: Path, source_path: Path, text: str) -> Path:
    """Write extracted text to `<source_stem>.txt` in output folder."""
    output_file = output_dir / f"{source_path.stem}.txt"
    output_file.write_text(text, encoding="utf-8")
    return output_file


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and normalize text from PDF/TXT/RTF/DOC/DOCX documents."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to PDF/TXT/RTF/DOC/DOCX file or folder containing documents.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Destination folder for extracted .txt files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Phase 1 extraction flow."""
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    documents = discover_documents(input_path)
    if not documents:
        raise FileNotFoundError(f"No supported files (.pdf/.txt/.rtf/.doc/.docx) found at: {input_path}")

    ensure_output_dir(output_dir)

    # Route by file extension.
    for source_file in documents:
        suffix = source_file.suffix.lower()
        if suffix == ".pdf":
            text = extract_pdf_text(source_file)
        elif suffix == ".txt":
            text = extract_txt_text(source_file)
        elif suffix == ".rtf":
            text = extract_rtf_text(source_file)
        elif suffix == ".docx":
            text = extract_docx_text(source_file)
        else:
            text = extract_doc_text(source_file)

        output_file = write_extracted_text(output_dir, source_file, text)
        print(f"Processed: {source_file} -> {output_file}")


if __name__ == "__main__":
    main()
