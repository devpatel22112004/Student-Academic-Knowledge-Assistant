"""HTTP client for backend API calls from Streamlit frontend."""

import requests
from typing import Optional, List
import json

# Backend API base URL
API_BASE_URL = "http://localhost:8000"


class BackendClient:
    """Client for backend API."""

    def __init__(self, token: Optional[str] = None):
        """Initialize with optional JWT token."""
        self.token = token
        self.headers = {}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def set_token(self, token: str):
        """Set JWT token for subsequent requests."""
        self.token = token
        self.headers["Authorization"] = f"Bearer {token}"

    # ===== AUTH ENDPOINTS =====

    def register(self, name: str, email: str, password: str) -> dict:
        """Register a new user."""
        url = f"{API_BASE_URL}/api/auth/register"
        data = {"name": name, "email": email, "password": password}
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def login(self, email: str, password: str) -> dict:
        """Login user."""
        url = f"{API_BASE_URL}/api/auth/login"
        data = {"email": email, "password": password}
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        self.set_token(result["token"])
        return result

    def get_profile(self, user_id: str) -> dict:
        """Get user profile."""
        url = f"{API_BASE_URL}/api/auth/profile"
        params = {"user_id": user_id}
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        return response.json()

    # ===== FILE ENDPOINTS =====

    def upload_file(self, filename: str, file_bytes: bytes) -> dict:
        """Upload a file."""
        url = f"{API_BASE_URL}/api/files/upload"
        files = {"file": (filename, file_bytes)}
        response = requests.post(url, files=files, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def list_files(self) -> List[dict]:
        """List user's uploaded files."""
        url = f"{API_BASE_URL}/api/files/list"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def delete_file(self, file_id: str) -> dict:
        """Delete a file."""
        url = f"{API_BASE_URL}/api/files/{file_id}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_file_chunks(self, file_hash: str, limit: Optional[int] = None) -> dict:
        """Get chunks for a file."""
        url = f"{API_BASE_URL}/api/files/chunks/{file_hash}"
        params = {}
        if limit:
            params["limit"] = limit
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        return response.json()

    # ===== RETRIEVAL ENDPOINTS =====

    def search_chunks(
        self, query: str, file_hashes: List[str], top_k: int = 5
    ) -> List[dict]:
        """Search for relevant chunks."""
        url = f"{API_BASE_URL}/api/retrieval/search"
        data = {"query": query, "file_hashes": file_hashes, "top_k": top_k}
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def save_chat(self, question: str, answer: str, file_hashes: List[str]) -> dict:
        """Save Q&A to chat history."""
        url = f"{API_BASE_URL}/api/retrieval/chat"
        data = {"question": question, "answer": answer, "file_hashes": file_hashes}
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_chat_history(self, limit: int = 50) -> List[dict]:
        """Get chat history."""
        url = f"{API_BASE_URL}/api/retrieval/history"
        params = {"limit": limit}
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def clear_chat_history(self) -> dict:
        """Clear chat history."""
        url = f"{API_BASE_URL}/api/retrieval/history"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    # ===== HELPER METHODS =====

    @staticmethod
    def health_check() -> bool:
        """Check if backend is running."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
