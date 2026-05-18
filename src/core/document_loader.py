from pathlib import Path
import io
import os

from pypdf import PdfReader


def find_all_documents():
    data_path = Path("data")
    documents = []

    documents.extend(data_path.rglob("*.pdf"))
    documents.extend(data_path.rglob("*.txt"))

    return sorted(documents)


def read_document_content(file_path):
    """Read a local PDF or TXT file and return source-text pairs."""
    content = []
    file_name = file_path.name

    if file_path.suffix.lower() == ".pdf":
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



def read_uploaded_documents(uploaded_files):
    """Read uploaded PDF or TXT files and return source-text pairs."""
    all_documents = []

    for uploaded in uploaded_files:
        name = uploaded.name
        suffix = os.path.splitext(name)[1].lower()

        if suffix == ".pdf":
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