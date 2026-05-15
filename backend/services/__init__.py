"""Backend services package."""

from backend.services.auth_service import (
    register_user,
    login_user,
    get_user_profile,
    hash_password,
    verify_password,
    generate_token,
    verify_token,
)
from backend.services.file_service import (
    upload_file,
    get_user_files,
    delete_file,
    get_file_chunks,
    compute_file_hash,
)
from backend.services.retrieval_service import (
    search_chunks,
    save_chat,
    get_chat_history,
    clear_chat_history,
)

__all__ = [
    # Auth
    "register_user",
    "login_user",
    "get_user_profile",
    "hash_password",
    "verify_password",
    "generate_token",
    "verify_token",
    # Files
    "upload_file",
    "get_user_files",
    "delete_file",
    "get_file_chunks",
    "compute_file_hash",
    # Retrieval
    "search_chunks",
    "save_chat",
    "get_chat_history",
    "clear_chat_history",
]
