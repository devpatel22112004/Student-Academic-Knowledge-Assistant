#!/usr/bin/env python3

"""
Phase 1 loader.
This script reads supported document files and writes normalized .txt outputs.
"""

# Keep modern typing behavior consistent across Python versions.
from __future__ import annotations

# CLI argument parsing.
import argparse
# Lazy import for optional parsers.
import importlib
# Check if system command exists (used for .doc via antiword).
import shutil
# Run external command for legacy .doc extraction.
import subprocess
# Path-safe file/folder handling.
from pathlib import Path

# PDF parser library.
from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF and keep page markers for traceability."""
    # Open PDF file.
    reader = PdfReader(str(pdf_path))
    # Store per-page extracted blocks.
    page_sections: list[str] = []

    # Read each page with page number.
    for page_number, page in enumerate(reader.pages, start=1):
        # Extract page text; fallback to empty string.
        page_text = page.extract_text() or ""
        # Add page marker so source page stays visible in output.
        page_sections.append(f"--- Page {page_number} ---\n{page_text.strip()}\n")

    # Merge all page blocks into one final string.
    return "\n".join(page_sections).strip()


def extract_txt_text(txt_path: Path) -> str:
    """Read UTF-8 text from TXT file."""
    # Read plain text directly.
    return txt_path.read_text(encoding="utf-8").strip()


def extract_docx_text(docx_path: Path) -> str:
    """Extract plain text from DOCX file."""
    try:
        # Import python-docx only when DOCX file is used.
        docx_module = importlib.import_module("docx")
        # Get Document class from module.
        docx_document = docx_module.Document
    except Exception:
        # Give clear install instruction if dependency missing.
        raise RuntimeError("DOCX support missing. Install dependency: python-docx")

    # Open DOCX document.
    doc = docx_document(str(docx_path))
    # Collect non-empty paragraph texts.
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    # Return combined text.
    return "\n".join(parts).strip()


def extract_rtf_text(rtf_path: Path) -> str:
    """Extract plain text from RTF file."""
    try:
        # Import RTF converter only when RTF file is used.
        striprtf_module = importlib.import_module("striprtf.striprtf")
        # Load converter function.
        convert_rtf = striprtf_module.rtf_to_text
    except Exception:
        # Give clear install instruction if dependency missing.
        raise RuntimeError("RTF support missing. Install dependency: striprtf")

    # Read raw RTF content.
    raw = rtf_path.read_text(encoding="utf-8", errors="ignore")
    # Convert RTF markup to plain text.
    # Remove null bytes that may appear in some RTF exports.
    return convert_rtf(raw).replace("\x00", "").strip()


def extract_doc_text(doc_path: Path) -> str:
    """Extract text from legacy DOC file using antiword binary."""
    # Validate antiword availability.
    if shutil.which("antiword") is None:
        raise RuntimeError(
            "DOC support needs system tool 'antiword'. Install with: sudo apt-get install -y antiword"
        )

    # Run antiword command and capture output text.
    result = subprocess.run(
        ["antiword", str(doc_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    # Return stdout as extracted text.
    return (result.stdout or "").strip()


def discover_documents(input_path: Path) -> list[Path]:
    """Return sorted list of supported input files."""
    # Allowed file extensions.
    supported = {".pdf", ".txt", ".rtf", ".doc", ".docx"}

    # If input is one supported file, return it directly.
    if input_path.is_file() and input_path.suffix.lower() in supported:
        return [input_path]

    # If input is folder, search recursively for all supported files.
    if input_path.is_dir():
        # Collect matches from each extension.
        matched: list[Path] = []
        for ext in supported:
            matched.extend(input_path.rglob(f"*{ext}"))
        # Keep output order stable.
        return sorted(matched)

    # Return empty if path is unsupported or missing.
    return []


def ensure_output_dir(output_dir: Path) -> None:
    """Create output directory when missing."""
    # Create directory tree; do nothing if already exists.
    output_dir.mkdir(parents=True, exist_ok=True)


def write_extracted_text(output_dir: Path, source_path: Path, text: str) -> Path:
    """Write extracted text to `<source_stem>.txt` in output folder."""
    # Build output file name using source file stem.
    output_file = output_dir / f"{source_path.stem}.txt"
    # Save UTF-8 text to disk.
    output_file.write_text(text, encoding="utf-8")
    # Return output path for logging.
    return output_file


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    # Create CLI parser.
    parser = argparse.ArgumentParser(
        description="Extract and normalize text from PDF/TXT/RTF/DOC/DOCX documents."
    )
    # Required input path argument.
    parser.add_argument(
        "--input",
        required=True,
        help="Path to PDF/TXT/RTF/DOC/DOCX file or folder containing documents.",
    )
    # Optional output directory argument.
    parser.add_argument(
        "--output",
        default="outputs/extracted",
        help="Destination folder for extracted .txt files.",
    )
    # Return parsed CLI values.
    return parser.parse_args()


def main() -> None:
    """Run Phase 1 extraction flow."""
    # Read CLI values.
    args = parse_args()
    # Convert input string to Path object.
    input_path = Path(args.input)
    # Convert output string to Path object.
    output_dir = Path(args.output)

    # Discover all supported documents from input path.
    documents = discover_documents(input_path)
    # Stop with clear error if nothing found.
    if not documents:
        raise FileNotFoundError(f"No supported files (.pdf/.txt/.rtf/.doc/.docx) found at: {input_path}")

    # Ensure output folder exists before writing files.
    ensure_output_dir(output_dir)

    # Process each discovered file using extension-based extractor.
    for source_file in documents:
        # Normalize extension to lowercase.
        suffix = source_file.suffix.lower()
        # PDF extraction path.
        if suffix == ".pdf":
            text = extract_pdf_text(source_file)
        # TXT extraction path.
        elif suffix == ".txt":
            text = extract_txt_text(source_file)
        # RTF extraction path.
        elif suffix == ".rtf":
            text = extract_rtf_text(source_file)
        # DOCX extraction path.
        elif suffix == ".docx":
            text = extract_docx_text(source_file)
        # DOC extraction path.
        else:
            text = extract_doc_text(source_file)

        # Save extracted text file.
        output_file = write_extracted_text(output_dir, source_file, text)
        # Print processing status.
        print(f"Processed: {source_file} -> {output_file}")


# Run main only when file is executed directly.
if __name__ == "__main__":
    main()
