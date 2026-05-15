from src.core.chunking import chunk_text
from src.core.document_loader import read_uploaded_documents
from src.core.embeddings import create_embeddings
from src.core.retrieval import build_search_index


def build_knowledge_base(uploaded_files):
    """Convert uploaded files into chunks, embeddings, and a FAISS index."""
    documents = read_uploaded_documents(uploaded_files)
    if not documents:
        return None

    chunks = chunk_text(documents)
    embeddings, model = create_embeddings(chunks)
    index = build_search_index(embeddings)

    return {
        "chunks": chunks,
        "model": model,
        "index": index,
    }