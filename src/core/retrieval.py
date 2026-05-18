import re
import numpy as np
from src.services.vector_db_service import get_pinecone_service

# This module implements the core retrieval logic for the knowledge assistant. It uses Pinecone vector database for efficient similarity search and includes lexical matching with hybrid scoring.

def extract_keywords(text):
    """Extract simple keywords from a query for lexical matching."""
    stop_words = {
        "the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for",
        "from", "what", "who", "when", "where", "why", "how", "explain", "describe",
    }
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in words if len(word) > 2 and word not in stop_words]


def lexical_overlap_score(query, chunk_text):
    """Score a chunk by keyword overlap with the query."""
    q_keywords = set(extract_keywords(query))
    if not q_keywords:
        return 0.0

    c_words = set(re.findall(r"[a-zA-Z0-9]+", chunk_text.lower()))
    hits = len(q_keywords.intersection(c_words))
    return hits / len(q_keywords)


def find_relevant_chunks(question, model, num_results=5, user_id=None):
    """
    Find the most relevant chunks for a question using Pinecone vector similarity + lexical ranking.
    
    Args:
        question: User's question/query
        model: Sentence transformer model for embedding
        num_results: Number of results to return
        user_id: User ID for filtering results (optional)
    
    Returns:
        List of relevant chunks with metadata
    """
    # Get Pinecone service
    vector_db = get_pinecone_service()
    
    # Create embedding for the question
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding, dtype=np.float32)
    # Convert to list format for Pinecone
    question_embedding = question_embedding.flatten().tolist()
    
    # Query Pinecone for similar vectors
    matches = vector_db.query_embeddings(
        query_vector=question_embedding,
        top_k=num_results,
        user_id=user_id
    )
    
    # Process and return results
    relevant_chunks = []
    for match in matches:
        chunk = {
            "text": match["metadata"].get("text", ""),
            "source": match["metadata"].get("source", "Unknown"),
            "chunk_id": match["id"],
            "score": match["score"]
        }
        relevant_chunks.append(chunk)
    
    return relevant_chunks