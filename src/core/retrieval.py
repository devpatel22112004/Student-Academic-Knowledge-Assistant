import re
import faiss
import numpy as np

# This module implements the core retrieval logic for the knowledge assistant. It includes functions to create vector embeddings from text chunks, build a FAISS index for efficient similarity search, extract keywords for lexical matching, and find relevant chunks based on a hybrid dense + lexical scoring approach.
def build_search_index(embeddings):
    """Build a FAISS index over normalized embeddings."""
    emb = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


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

# This function finds the most relevant chunks for a given question by combining dense vector similarity from the FAISS index with a lexical overlap score. It retrieves candidate chunks based on dense similarity, then re-ranks them using a weighted hybrid score to return the top relevant chunks.
def find_relevant_chunks(question, index, chunks, model, num_results=5):
    """Find the most relevant chunks for a question using dense + lexical ranking."""
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding, dtype=np.float32)
    faiss.normalize_L2(question_embedding)

    candidate_k = min(max(num_results * 3, 10), len(chunks))
    dense_scores, indices = index.search(question_embedding, candidate_k)

    candidates = []
    for rank, chunk_idx in enumerate(indices[0]):
        dense = float(dense_scores[0][rank])
        lexical = lexical_overlap_score(question, chunks[chunk_idx]["text"])
        hybrid = (0.7 * dense) + (0.3 * lexical)
        candidates.append((hybrid, chunk_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [chunks[idx] for _, idx in candidates[:num_results]]