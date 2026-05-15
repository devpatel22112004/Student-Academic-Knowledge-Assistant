import numpy as np
from sentence_transformers import SentenceTransformer


def create_embeddings(chunks):
    """Convert chunks into vector embeddings with a sentence transformer model."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings, model