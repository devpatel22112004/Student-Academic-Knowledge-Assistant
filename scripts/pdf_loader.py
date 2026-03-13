# Phase 1 — PDF Loader Script
# Kaam: PDF file(s) se page-wise text nikalna aur .txt file me save karna
# Ye RAG pipeline ka pehla step hai — pehle document load karo, tabhi kuch aage ho sakta hai

from __future__ import annotations

import argparse
from pathlib import Path

# pypdf library use hoti hai PDF ke andar ka text padhne ke liye
from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    # PDF file ko open karo aur reader object banao
    reader = PdfReader(str(pdf_path))
    page_sections: list[str] = []

    # Har page par ek ek karke loop chalao
    for page_number, page in enumerate(reader.pages, start=1):
        # Page ka text nikalo — agar kuch na mile to empty string lo
        page_text = page.extract_text() or ""
        # Har page ka text "--- Page X ---" header ke saath list me daalo
        page_sections.append(f"--- Page {page_number} ---\n{page_text.strip()}\n")

    # Sab pages ka text ek saath jodke return karo
    return "\n".join(page_sections).strip()


def discover_pdfs(input_path: Path) -> list[Path]:
    # Agar seedha ek PDF file di hai to uski list return karo
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]

    # Agar folder diya hai to usme se sab .pdf files dhundo (sub-folders bhi)
    if input_path.is_dir():
        return sorted(input_path.rglob("*.pdf"))

    # Kuch nahi mila to empty list
    return []


def ensure_output_dir(output_dir: Path) -> None:
    # Output folder nahi hai to bana do — exist kare to koi error mat do
    output_dir.mkdir(parents=True, exist_ok=True)


def write_extracted_text(output_dir: Path, pdf_path: Path, text: str) -> Path:
    # Output file ka naam PDF ke naam jaisa rakho, sirf extension .txt kar do
    output_file = output_dir / f"{pdf_path.stem}.txt"
    # Extracted text ko file me UTF-8 encoding me likho
    output_file.write_text(text, encoding="utf-8")
    return output_file


def parse_args() -> argparse.Namespace:
    # Command line se --input aur --output arguments lene ka setup
    parser = argparse.ArgumentParser(
        description="PDF files load karo aur unka text extract karo."
    )
    # --input: PDF file ya folder ka path (zaruri argument)
    parser.add_argument(
        "--input",
        required=True,
        help="PDF file ya folder ka path jisme PDFs rakhi hain.",
    )
    # --output: text files kahan save karni hain (default: outputs folder)
    parser.add_argument(
        "--output",
        default="outputs",
        help="Folder jahan extracted text files save hongi.",
    )
    return parser.parse_args()


def main() -> None:
    # Command line arguments parse karo
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Input path se sab PDF files dhundo
    pdf_files = discover_pdfs(input_path)

    # Agar koi PDF nahi mili to error do
    if not pdf_files:
        raise FileNotFoundError(f"Koi PDF nahi mili yahan: {input_path}")

    # Output folder banao agar pehle se nahi hai
    ensure_output_dir(output_dir)

    # Har PDF ke liye: text nikalo aur file me save karo
    for pdf_file in pdf_files:
        text = extract_pdf_text(pdf_file)
        output_file = write_extracted_text(output_dir, pdf_file, text)
        # Console me batao ki kaunsi file process hui aur output kahan gayi
        print(f"Processed: {pdf_file} -> {output_file}")


# Ye block tabhi chalta hai jab hum seedha is file ko run karte hain
# (kisi aur file ne import kiya ho tab nahi chalta)
if __name__ == "__main__":
    main()
