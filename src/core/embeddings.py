import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# This module provides a function to create vector embeddings from text chunks using a pre-trained sentence transformer model. The embeddings can be used for efficient similarity search during retrieval.
@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the sentence transformer model for reuse."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks, model=None):
    """Convert chunks into vector embeddings with a sentence transformer model."""
    model = model or get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings, model