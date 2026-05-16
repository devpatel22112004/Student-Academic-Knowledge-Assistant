from pathlib import Path
import io
import os

from pypdf import PdfReader

# This module provides functions to find and read documents from the local filesystem and uploaded files. It supports PDF and TXT formats, returning a list of (source, text) pairs for downstream processing.
def find_all_documents():
    data_path = Path("data")
    documents = []

    # Search only the document types this app supports.
    documents.extend(data_path.rglob("*.pdf"))
    documents.extend(data_path.rglob("*.txt"))

    return sorted(documents)


def read_document_content(file_path):
    """Read a local PDF or TXT file and return source-text pairs."""
    content = []
    file_name = file_path.name

    if file_path.suffix.lower() == ".pdf":
        # Read PDF file page by page so source tracking stays accurate.
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                content.append((f"{file_name} - Page {page_num}", text))
    elif file_path.suffix.lower() == ".txt":
        text = file_path.read_text(encoding="utf-8")
        if text.strip():
            content.append((file_name, text))

    return content

# This function reads uploaded files (PDF or TXT) from Streamlit's file uploader, extracts their text content, and returns a list of (source, text) pairs. For PDFs, it processes each page separately to maintain source granularity.
def read_uploaded_documents(uploaded_files):
    """Read uploaded PDF or TXT files and return source-text pairs."""
    all_documents = []

    for uploaded in uploaded_files:
        name = uploaded.name
        suffix = os.path.splitext(name)[1].lower()

        if suffix == ".pdf":
            # Streamlit uploads stay in memory, so PdfReader can read bytes directly.
            pdf_reader = PdfReader(io.BytesIO(uploaded.getvalue()))
            for i, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_documents.append((f"{name} - Page {i}", page_text))
        elif suffix == ".txt":
            text = uploaded.getvalue().decode("utf-8", errors="ignore")
            if text.strip():
                all_documents.append((name, text))

    return all_documents