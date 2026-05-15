"""User model for authentication."""

from datetime import datetime
from bson import ObjectId
from backend.db import get_db


class User:
    """User document in MongoDB."""

    @staticmethod
    def create(name: str, email: str, hashed_password: str) -> dict:
        """Create a new user."""
        db = get_db()
        user_doc = {
            "_id": ObjectId(),
            "name": name,
            "email": email.lower().strip(),
            "password": hashed_password,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        result = db.users.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        return user_doc

    @staticmethod
    def find_by_email(email: str) -> dict:
        """Find user by email."""
        db = get_db()
        return db.users.find_one({"email": email.lower().strip()})

    @staticmethod
    def find_by_id(user_id) -> dict:
        """Find user by ID."""
        db = get_db()
        return db.users.find_one({"_id": ObjectId(user_id) if isinstance(user_id, str) else user_id})

    @staticmethod
    def update(user_id, updates: dict) -> dict:
        """Update user document."""
        db = get_db()
        updates["updated_at"] = datetime.utcnow()
        result = db.users.find_one_and_update(
            {"_id": ObjectId(user_id) if isinstance(user_id, str) else user_id},
            {"$set": updates},
            return_document=True,
        )
        return result
