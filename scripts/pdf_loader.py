#!/usr/bin/env python3

"""
PHASE 1 — Document Loader (PDF + TXT Support)

क्या करता है:
1) Documents discover (PDF + TXT दोनों)
2) Text extract:
   - PDF: pypdf से page-wise
   - TXT: Native Python (कोई extra library नहीं)
3) Output: .txt files

Dependencies:
REQUIRED: pypdf (PDF के लिए)
NOT NEEDED FOR TXT: TXT plain text है, pathlib.read_text() काफी है (built-in)

Usage: bash run_phase1.sh data outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

# pypdf: PDF text extraction के लिए (TXT के लिए नहीं चाहिए)
from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    # Short: PDF ko read karo.
    # Concept: Hum page-wise extraction rakhte hain taaki downstream metadata me
    # page context preserve rahe (later citation/retrieval me useful).
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


def extract_txt_text(txt_path: Path) -> str:
    # Short: TXT ko direct read karo.
    # Concept: TXT plain-text hota hai, isliye yahan kisi external parser ki
    # zarurat nahi padti (zero extra dependency path).
    return txt_path.read_text(encoding="utf-8").strip()


def discover_documents(input_path: Path) -> list[Path]:
    # Short: Supported inputs identify karo.
    # Concept: File-mode aur folder-mode dono support karke script ko
    # automation-friendly banaya gaya hai.
    # Agar seedha ek supported file di hai to uski list return karo.
    if input_path.is_file() and input_path.suffix.lower() in {".pdf", ".txt"}:
        return [input_path]

    # Agar folder diya hai to usme se sab .pdf/.txt files dhundo (sub-folders bhi).
    if input_path.is_dir():
        pdf_files = list(input_path.rglob("*.pdf"))
        txt_files = list(input_path.rglob("*.txt"))
        return sorted(pdf_files + txt_files)

    # Kuch nahi mila to empty list
    return []


def ensure_output_dir(output_dir: Path) -> None:
    # Output folder nahi hai to bana do — exist kare to koi error mat do
    output_dir.mkdir(parents=True, exist_ok=True)


def write_extracted_text(output_dir: Path, source_path: Path, text: str) -> Path:
    # Short: Output file name normalize karo.
    # Concept: Har input ko normalized .txt me convert karne se Phase 2 ke liye
    # ingestion format consistent rehta hai.
    # Output file ka naam source file ke stem jaisa rakho, extension .txt rahega.
    output_file = output_dir / f"{source_path.stem}.txt"
    # Extracted text ko file me UTF-8 encoding me likho
    output_file.write_text(text, encoding="utf-8")
    return output_file


def parse_args() -> argparse.Namespace:
    # Command line se --input aur --output arguments lene ka setup
    parser = argparse.ArgumentParser(
        description="PDF/TXT files load karo aur unka text extract karo."
    )
    # --input: PDF/TXT file ya folder ka path (zaruri argument)
    parser.add_argument(
        "--input",
        required=True,
        help="PDF/TXT file ya folder ka path jisme documents rakhe hain.",
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

    # Input path se sab supported documents dhundo
    documents = discover_documents(input_path)

    # Agar koi supported file nahi mili to error do
    if not documents:
        raise FileNotFoundError(f"Koi supported file (.pdf/.txt) nahi mili yahan: {input_path}")

    # Output folder banao agar pehle se nahi hai
    ensure_output_dir(output_dir)

    # Har document ke liye text extraction strategy extension-based choose hoti hai.
    # PDF -> structured page extraction, TXT -> direct raw text read.
    for source_file in documents:
        if source_file.suffix.lower() == ".pdf":
            text = extract_pdf_text(source_file)
        else:
            text = extract_txt_text(source_file)

        output_file = write_extracted_text(output_dir, source_file, text)
        # Console me batao ki kaunsi file process hui aur output kahan gayi
        print(f"Processed: {source_file} -> {output_file}")


# Ye block tabhi chalta hai jab hum seedha is file ko run karte hain
# (kisi aur file ne import kiya ho tab nahi chalta)
if __name__ == "__main__":
    main()
