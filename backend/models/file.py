"""File and chunk models for uploaded documents."""

from datetime import datetime
from bson import ObjectId
from backend.db import get_db


class UploadedFile:
    """Metadata for uploaded files."""

    @staticmethod
    def create(user_id, filename: str, file_hash: str, file_size: int, gridfs_id=None) -> dict:
        """Create uploaded file record."""
        db = get_db()
        file_doc = {
            "_id": ObjectId(),
            "user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id,
            "filename": filename,
            "file_hash": file_hash,
            "file_size": file_size,
            "gridfs_id": gridfs_id,  # Reference to GridFS file
            "uploaded_at": datetime.utcnow(),
            "processed": False,  # Flag if chunks/embeddings are extracted
            "processed_at": None,
        }
        result = db.uploaded_files.insert_one(file_doc)
        file_doc["_id"] = result.inserted_id
        return file_doc

    @staticmethod
    def find_by_hash(file_hash: str) -> dict:
        """Find file record by hash."""
        db = get_db()
        return db.uploaded_files.find_one({"file_hash": file_hash})

    @staticmethod
    def find_by_user(user_id) -> list:
        """Get all uploaded files for a user."""
        db = get_db()
        return list(
            db.uploaded_files.find(
                {"user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id}
            )
            .sort("uploaded_at", -1)
        )

    @staticmethod
    def mark_processed(file_hash: str):
        """Mark file as processed (chunks extracted)."""
        db = get_db()
        db.uploaded_files.update_many(
            {"file_hash": file_hash},
            {"$set": {"processed": True, "processed_at": datetime.utcnow()}},
        )

    @staticmethod
    def delete(file_id):
        """Delete file record."""
        db = get_db()
        return db.uploaded_files.delete_one(
            {"_id": ObjectId(file_id) if isinstance(file_id, str) else file_id}
        )


class FileChunk:
    """Extracted text chunks from files."""

    @staticmethod
    def create(file_hash: str, chunk_index: int, text: str, source: str) -> dict:
        """Create a chunk record."""
        db = get_db()
        chunk_doc = {
            "_id": ObjectId(),
            "file_hash": file_hash,
            "chunk_index": chunk_index,
            "text": text,
            "source": source,  # e.g., "file.pdf page 5"
            "created_at": datetime.utcnow(),
        }
        result = db.file_chunks.insert_one(chunk_doc)
        chunk_doc["_id"] = result.inserted_id
        return chunk_doc

    @staticmethod
    def find_by_file_hash(file_hash: str) -> list:
        """Get all chunks for a file."""
        db = get_db()
        return list(
            db.file_chunks.find({"file_hash": file_hash}).sort("chunk_index", 1)
        )

    @staticmethod
    def bulk_create(file_hash: str, chunks: list) -> int:
        """Insert multiple chunks at once.
        
        Args:
            file_hash: unique file identifier
            chunks: list of dicts with keys: chunk_index, text, source
        """
        db = get_db()
        docs = [
            {
                "file_hash": file_hash,
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "source": c["source"],
                "created_at": datetime.utcnow(),
            }
            for c in chunks
        ]
        result = db.file_chunks.insert_many(docs, ordered=False)
        return len(result.inserted_ids)

    @staticmethod
    def delete_by_hash(file_hash: str):
        """Delete all chunks for a file."""
        db = get_db()
        return db.file_chunks.delete_many({"file_hash": file_hash})
