"""
User Files Manager - Tracks uploaded files per user persistently
Stores metadata about files each user has uploaded
"""

import json
import os
from datetime import datetime
from pathlib import Path

USER_FILES_DB = Path("data/user_files.json")

def ensure_db_exists():
    """Ensure the user files database exists"""
    USER_FILES_DB.parent.mkdir(parents=True, exist_ok=True)
    if not USER_FILES_DB.exists():
        USER_FILES_DB.write_text(json.dumps({}))

def get_user_files(user_id: str) -> list:
    """Get list of files uploaded by a user"""
    ensure_db_exists()
    try:
        data = json.loads(USER_FILES_DB.read_text())
        return data.get(user_id, [])
    except:
        return []

def add_user_file(user_id: str, filename: str, file_hash: str) -> None:
    """Add a file to user's uploaded files list"""
    ensure_db_exists()
    try:
        data = json.loads(USER_FILES_DB.read_text())
        if user_id not in data:
            data[user_id] = []
        
        # Check if file already in list
        existing = next((f for f in data[user_id] if f["hash"] == file_hash), None)
        if not existing:
            data[user_id].append({
                "name": filename,
                "hash": file_hash,
                "date": datetime.now().strftime("%d %b %Y"),
                "timestamp": datetime.now().isoformat()
            })
            USER_FILES_DB.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error adding user file: {e}")

def remove_user_file(user_id: str, file_hash: str) -> None:
    """Remove a file from user's uploaded files list"""
    ensure_db_exists()
    try:
        data = json.loads(USER_FILES_DB.read_text())
        if user_id in data:
            data[user_id] = [f for f in data[user_id] if f["hash"] != file_hash]
            USER_FILES_DB.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error removing user file: {e}")

def clear_user_files(user_id: str) -> None:
    """Clear all files for a user"""
    ensure_db_exists()
    try:
        data = json.loads(USER_FILES_DB.read_text())
        if user_id in data:
            del data[user_id]
            USER_FILES_DB.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error clearing user files: {e}")

def file_hash_exists(user_id: str, file_hash: str) -> bool:
    """Check if a file hash already exists for a user"""
    ensure_db_exists()
    try:
        user_files = get_user_files(user_id)
        return any(f.get("hash") == file_hash for f in user_files)
    except:
        return False

def get_file_names_by_hashes(user_id: str, file_hashes: list) -> dict:
    """Get filenames for given file hashes"""
    ensure_db_exists()
    try:
        user_files = get_user_files(user_id)
        return {f.get("hash"): f.get("name") for f in user_files if f.get("hash") in file_hashes}
    except:
        return {}

def get_all_file_hashes(user_id: str) -> set:
    """Get all file hashes for a user"""
    ensure_db_exists()
    try:
        user_files = get_user_files(user_id)
        return {f.get("hash") for f in user_files if f.get("hash")}
    except:
        return set()
