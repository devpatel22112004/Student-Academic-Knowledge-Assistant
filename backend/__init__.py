"""Backend package for MongoDB-based authentication and file management."""

from backend.db import init_db, get_db, close_db

__all__ = ["init_db", "get_db", "close_db"]
