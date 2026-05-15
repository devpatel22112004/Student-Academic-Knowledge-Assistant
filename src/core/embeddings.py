import numpy as np
from sentence_transformers import SentenceTransformer

# This module provides a function to create vector embeddings from text chunks using a pre-trained sentence transformer model. The embeddings can be used for efficient similarity search during retrieval.
def create_embeddings(chunks):
    """Convert chunks into vector embeddings with a sentence transformer model."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings, model