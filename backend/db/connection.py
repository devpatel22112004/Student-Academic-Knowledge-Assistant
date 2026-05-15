"""MongoDB connection and initialization."""

import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# MongoDB connection URI from environment
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:27017"
)
DATABASE_NAME = os.getenv("MONGODB_DB_NAME", "student_knowledge_assistant")

# Global client and db
_client = None
_db = None


def get_db():
    """Get MongoDB database instance."""
    global _db
    if _db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db


def init_db():
    """Initialize MongoDB connection and create collections."""
    global _client, _db

    try:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        _client.admin.command("ping")
        _db = _client[DATABASE_NAME]

        # Create indexes for common queries
        _create_indexes()
        print(f"✓ Connected to MongoDB: {DATABASE_NAME}")
        return _db

    except ServerSelectionTimeoutError:
        print("✗ Failed to connect to MongoDB. Check MONGODB_URI and that MongoDB is running.")
        raise
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        raise


def _create_indexes():
    """Create MongoDB indexes for efficient querying."""
    db = _db

    # Users collection
    db.users.create_index("email", unique=True)

    # Uploaded files collection
    db.uploaded_files.create_index([("user_id", 1), ("file_hash", 1)], unique=True)
    db.uploaded_files.create_index([("user_id", 1), ("uploaded_at", -1)])
    db.uploaded_files.create_index("file_hash")

    # File chunks collection
    db.file_chunks.create_index([("file_hash", 1), ("chunk_index", 1)], unique=True)
    db.file_chunks.create_index("file_hash")

    # Chat history collection
    db.chat_history.create_index([("user_id", 1), ("created_at", -1)])

    print("✓ MongoDB indexes created")


def close_db():
    """Close MongoDB connection."""
    global _client
    if _client:
        _client.close()
        print("✓ MongoDB connection closed")
