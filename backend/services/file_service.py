"""File management service with deduplication and chunk extraction."""

import hashlib
from io import BytesIO
from backend.models import UploadedFile, FileChunk
from src.core.chunking import chunk_text
from src.core.document_loader import extract_text_from_pdf


def compute_file_hash(file_content: bytes) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()


def upload_file(user_id: str, filename: str, file_content: bytes) -> dict:
    """Upload file with deduplication check.
    
    If file hash exists:
    - Reuse existing chunks and embeddings
    
    If file hash is new:
    - Extract chunks
    - Save to database
    - Mark as processed
    
    Args:
        user_id: Authenticated user ID
        filename: Original file name
        file_content: File bytes
        
    Returns:
        dict with file_id, filename, file_hash, is_new, chunks_count
    """
    # Compute hash
    file_hash = compute_file_hash(file_content)
    file_size = len(file_content)
    
    # Check if file already processed
    existing_file = UploadedFile.find_by_hash(file_hash)
    
    if existing_file and existing_file["processed"]:
        # File exists and is processed - reuse chunks
        chunks = FileChunk.find_by_file_hash(file_hash)
        
        # Create record for this user if not exists
        user_file = UploadedFile.create(
            user_id=user_id,
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            gridfs_id=existing_file.get("gridfs_id"),
        )
        
        return {
            "file_id": str(user_file["_id"]),
            "filename": filename,
            "file_hash": file_hash,
            "is_new": False,
            "chunks_count": len(chunks),
            "message": "File already processed, reusing chunks",
        }
    
    # NEW FILE - Extract and process chunks
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(BytesIO(file_content))
        
        # Split into chunks
        chunks_data = chunk_text(text, filename)
        
        # Create file record
        file_record = UploadedFile.create(
            user_id=user_id,
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
        )
        
        # Save chunks
        FileChunk.bulk_create(file_hash, chunks_data)
        
        # Mark as processed
        UploadedFile.mark_processed(file_hash)
        
        return {
            "file_id": str(file_record["_id"]),
            "filename": filename,
            "file_hash": file_hash,
            "is_new": True,
            "chunks_count": len(chunks_data),
            "message": f"File processed, {len(chunks_data)} chunks created",
        }
        
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")


def get_user_files(user_id: str) -> list:
    """Get all uploaded files for a user."""
    files = UploadedFile.find_by_user(user_id)
    
    return [
        {
            "file_id": str(f["_id"]),
            "filename": f["filename"],
            "file_hash": f["file_hash"],
            "file_size": f["file_size"],
            "uploaded_at": f["uploaded_at"].isoformat(),
            "processed": f["processed"],
        }
        for f in files
    ]


def delete_file(user_id: str, file_id: str) -> dict:
    """Delete a file for user.
    
    Note: Chunks are NOT deleted (may be used by other users).
    Only the user's file reference is deleted.
    """
    file_record = UploadedFile.find_by_hash(file_id)
    
    if not file_record:
        raise ValueError("File not found")
    
    # Verify ownership
    if str(file_record["user_id"]) != str(user_id):
        raise ValueError("Unauthorized")
    
    # Delete file record
    result = UploadedFile.delete(file_id)
    
    return {
        "deleted": True,
        "file_id": file_id,
        "message": "File removed from your library (chunks kept for reuse)",
    }


def get_file_chunks(file_hash: str, limit: int = None) -> list:
    """Get chunks for a file."""
    chunks = FileChunk.find_by_file_hash(file_hash)
    
    if limit:
        chunks = chunks[:limit]
    
    return [
        {
            "chunk_id": str(c["_id"]),
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "source": c["source"],
        }
        for c in chunks
    ]
