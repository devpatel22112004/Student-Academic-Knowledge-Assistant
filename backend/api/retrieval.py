"""Retrieval and chat API endpoints."""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List
from backend.services import retrieval_service, auth_service

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


class SearchRequest(BaseModel):
    """Search request for document chunks."""
    query: str
    file_hashes: List[str]
    top_k: int = 5


class ChunkResult(BaseModel):
    """A single chunk result."""
    chunk_id: str
    chunk_index: int
    text: str
    source: str
    similarity: float


class ChatRequest(BaseModel):
    """Chat/answer request."""
    question: str
    answer: str
    file_hashes: List[str]


class ChatMessage(BaseModel):
    """Chat message."""
    message_id: str
    question: str
    answer: str
    sources: List[str]
    created_at: str


@router.post("/search", response_model=List[ChunkResult])
async def search_chunks(
    req: SearchRequest,
    authorization: str = Header(None),
):
    """Search for relevant chunks across files."""
    try:
        # Verify token
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        
        token = authorization.split(" ")[1]
        payload = auth_service.verify_token(token)
        user_id = payload["user_id"]
        
        # Search
        results = retrieval_service.search_chunks(
            query_text=req.query,
            file_hashes=req.file_hashes,
            top_k=req.top_k,
        )
        return results
        
    except ValueError as e:
        if "Token" in str(e):
            raise HTTPException(status_code=401, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/chat")
async def save_chat(
    req: ChatRequest,
    authorization: str = Header(None),
):
    """Save a Q&A pair to chat history."""
    try:
        # Verify token
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        
        token = authorization.split(" ")[1]
        payload = auth_service.verify_token(token)
        user_id = payload["user_id"]
        
        # Save chat
        result = retrieval_service.save_chat(
            user_id=user_id,
            question=req.question,
            answer=req.answer,
            sources=req.file_hashes,
        )
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.get("/history", response_model=List[ChatMessage])
async def get_history(
    limit: int = 50,
    authorization: str = Header(None),
):
    """Get user's chat history."""
    try:
        # Verify token
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        
        token = authorization.split(" ")[1]
        payload = auth_service.verify_token(token)
        user_id = payload["user_id"]
        
        # Get history
        history = retrieval_service.get_chat_history(user_id, limit=limit)
        return history
        
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.delete("/history")
async def clear_history(authorization: str = Header(None)):
    """Clear user's chat history."""
    try:
        # Verify token
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        
        token = authorization.split(" ")[1]
        payload = auth_service.verify_token(token)
        user_id = payload["user_id"]
        
        # Clear history
        result = retrieval_service.clear_chat_history(user_id)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
