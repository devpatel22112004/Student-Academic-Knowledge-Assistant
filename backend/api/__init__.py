"""REST API endpoints."""

from backend.api.auth import router as auth_router
from backend.api.files import router as files_router
from backend.api.retrieval import router as retrieval_router

__all__ = ["auth_router", "files_router", "retrieval_router"]
