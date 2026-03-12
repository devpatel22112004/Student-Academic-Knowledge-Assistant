from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    page_sections: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_sections.append(f"--- Page {page_number} ---\n{page_text.strip()}\n")

    return "\n".join(page_sections).strip()


def discover_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]

    if input_path.is_dir():
        return sorted(input_path.rglob("*.pdf"))

    return []


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def write_extracted_text(output_dir: Path, pdf_path: Path, text: str) -> Path:
    output_file = output_dir / f"{pdf_path.stem}.txt"
    output_file.write_text(text, encoding="utf-8")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load PDF files and extract page-wise text."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a PDF file or a directory containing PDFs.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory where extracted text files will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    pdf_files = discover_pdfs(input_path)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found at: {input_path}")

    ensure_output_dir(output_dir)

    for pdf_file in pdf_files:
        text = extract_pdf_text(pdf_file)
        output_file = write_extracted_text(output_dir, pdf_file, text)
        print(f"Processed: {pdf_file} -> {output_file}")


if __name__ == "__main__":
    main()
