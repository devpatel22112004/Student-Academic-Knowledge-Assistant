"""
Pinecone Vector Database Service
Handles all vector storage and retrieval operations
"""

import hashlib
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

class PineconeService:
    def __init__(self):
        """Initialize Pinecone connection"""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.host = os.getenv("PINECONE_HOST")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "student-assistant")
        
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(host=self.host)
        
    def compute_file_hash(self, file_content: bytes) -> str:
        """Create MD5 hash of file for duplicate detection"""
        return hashlib.md5(file_content).hexdigest()
    
    def file_exists(self, file_hash: str, user_id: str) -> bool:
        """Check if file already processed using hash - checks Pinecone metadata"""
        try:
            results = self.index.query(
                vector=[0] * 384,
                top_k=100,
                filter={
                    "$and": [
                        {"file_hash": {"$eq": file_hash}},
                        {"user_id": {"$eq": user_id}}
                    ]
                },
                include_metadata=True
            )
            file_found = len(results.get("matches", [])) > 0
            return file_found
        except Exception as e:
            print(f"⚠️ Warning checking file existence: {e}")
            return False
    
    def get_user_file_hashes(self, user_id: str) -> set:
        """Get all file hashes for a user from Pinecone"""
        try:
            results = self.index.query(
                vector=[0] * 384,
                top_k=1000,
                filter={"user_id": user_id},
                include_metadata=True
            )
            file_hashes = set()
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                if "file_hash" in metadata:
                    file_hashes.add(metadata["file_hash"])
            return file_hashes
        except:
            return set()
    
    def upsert_embeddings(self, 
                         chunk_id: str,
                         embedding: list,
                         metadata: dict) -> None:
        """Store embeddings in Pinecone with metadata"""
        try:
            self.index.upsert(
                vectors=[
                    (
                        chunk_id,
                        embedding,
                        metadata
                    )
                ]
            )
        except Exception as e:
            print(f"Error upserting embedding: {e}")
            raise
    
    def upsert_batch(self, vectors_batch: list) -> None:
        """Upsert multiple embeddings at once"""
        try:
            self.index.upsert(vectors=vectors_batch)
        except Exception as e:
            print(f"Error upserting batch: {e}")
            raise
    
    def query_embeddings(self, 
                        query_vector: list,
                        top_k: int = 5,
                        user_id: str = None) -> list:
        """Query similar embeddings from Pinecone"""
        try:
            filter_dict = {}
            if user_id:
                filter_dict["user_id"] = user_id
            
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            return results.get("matches", [])
        except Exception as e:
            print(f"Error querying embeddings: {e}")
            raise
    
    def delete_user_vectors(self, user_id: str) -> None:
        """Delete all vectors for a specific user"""
        try:
            self.index.delete(filter={"user_id": user_id})
        except Exception as e:
            print(f"Error deleting user vectors: {e}")
            raise
    
    def delete_file_vectors(self, file_hash: str, user_id: str) -> None:
        """Delete all vectors for a specific file"""
        try:
            self.index.delete(
                filter={
                    "file_hash": file_hash,
                    "user_id": user_id
                }
            )
        except Exception as e:
            print(f"Error deleting file vectors: {e}")
            raise
    
    def get_index_stats(self):
        """Get Pinecone index statistics"""
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            print(f"Error getting index stats: {e}")
            raise


_pinecone_service = None

def get_pinecone_service():
    """Get or create Pinecone service instance"""
    global _pinecone_service
    if _pinecone_service is None:
        _pinecone_service = PineconeService()
    return _pinecone_service
