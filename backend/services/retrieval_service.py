"""Retrieval service for searching across files and getting relevant chunks."""

from backend.models import FileChunk, ChatMessage
from backend.config import RETRIEVAL_TOP_K


def search_chunks(query_text: str, file_hashes: list, top_k: int = RETRIEVAL_TOP_K) -> list:
    """Search for relevant chunks.
    
    For now: Return first top_k chunks from specified files.
    Later: Use vector similarity with Qdrant.
    
    Args:
        query_text: Search query (will be used for vector similarity)
        file_hashes: List of file hashes to search in
        top_k: Number of results to return
        
    Returns:
        list of relevant chunks with metadata
    """
    # TODO: Replace with vector similarity search using Qdrant
    # For now: retrieve top_k chunks from each file
    
    relevant_chunks = []
    
    for file_hash in file_hashes:
        chunks = FileChunk.find_by_file_hash(file_hash)
        relevant_chunks.extend(chunks[:top_k])
    
    return [
        {
            "chunk_id": str(c["_id"]),
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "source": c["source"],
            "similarity": 0.95,  # TODO: Compute from vector similarity
        }
        for c in relevant_chunks[:top_k]
    ]


def save_chat(user_id: str, question: str, answer: str, sources: list) -> dict:
    """Save a Q&A pair to chat history."""
    message = ChatMessage.create(
        user_id=user_id,
        question=question,
        answer=answer,
        sources=sources,
    )
    
    return {
        "message_id": str(message["_id"]),
        "created_at": message["created_at"].isoformat(),
    }


def get_chat_history(user_id: str, limit: int = 50) -> list:
    """Get user's chat history."""
    messages = ChatMessage.find_by_user(user_id, limit=limit)
    
    return [
        {
            "message_id": str(m["_id"]),
            "question": m["question"],
            "answer": m["answer"],
            "sources": m["sources"],
            "created_at": m["created_at"].isoformat(),
        }
        for m in reversed(messages)  # Oldest first
    ]


def clear_chat_history(user_id: str) -> dict:
    """Clear all chat history for user."""
    result = ChatMessage.delete_by_user(user_id)
    
    return {
        "cleared": True,
        "messages_deleted": result.deleted_count,
    }
