"""File management API endpoints."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Header
from pydantic import BaseModel
from typing import List
from backend.services import file_service, auth_service

router = APIRouter(prefix="/api/files", tags=["files"])


class UploadResponse(BaseModel):
    """File upload response."""
    file_id: str
    filename: str
    file_hash: str
    is_new: bool
    chunks_count: int
    message: str


class FileInfo(BaseModel):
    """File information."""
    file_id: str
    filename: str
    file_hash: str
    file_size: int
    uploaded_at: str
    processed: bool


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    authorization: str = Header(None),
):
    """Upload a file for processing."""
    try:
        # Extract token from Authorization header
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        
        token = authorization.split(" ")[1]
        payload = auth_service.verify_token(token)
        user_id = payload["user_id"]
        
        # Read file content
        content = await file.read()
        
        # Upload and process
        result = file_service.upload_file(
            user_id=user_id,
            filename=file.filename,
            file_content=content,
        )
        return result
        
    except ValueError as e:
        if "Token" in str(e):
            raise HTTPException(status_code=401, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/list", response_model=List[FileInfo])
async def list_files(authorization: str = Header(None)):
    """List all files uploaded by user."""
    try:
        # Extract token
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        
        token = authorization.split(" ")[1]
        payload = auth_service.verify_token(token)
        user_id = payload["user_id"]
        
        # Get files
        files = file_service.get_user_files(user_id)
        return files
        
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    authorization: str = Header(None),
):
    """Delete a file."""
    try:
        # Extract token
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        
        token = authorization.split(" ")[1]
        payload = auth_service.verify_token(token)
        user_id = payload["user_id"]
        
        # Delete file
        result = file_service.delete_file(user_id, file_id)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/chunks/{file_hash}")
async def get_file_chunks(
    file_hash: str,
    limit: int = None,
):
    """Get chunks for a specific file."""
    try:
        chunks = file_service.get_file_chunks(file_hash, limit=limit)
        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
