"""Backend configuration and constants."""

import os
from datetime import timedelta

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "student_knowledge_assistant")

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = timedelta(days=30)

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768  # all-mpnet-base-v2 output dimension

# File Processing Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Vector DB Configuration (Qdrant)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "academic_documents"
QDRANT_VECTOR_SIZE = EMBEDDING_DIMENSION

# Retrieval Configuration
RETRIEVAL_TOP_K = 5  # Default number of chunks to retrieve per query
MIN_SIMILARITY_SCORE = 0.5
