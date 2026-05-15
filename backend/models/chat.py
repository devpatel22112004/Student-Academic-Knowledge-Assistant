"""Chat history model."""

from datetime import datetime
from bson import ObjectId
from backend.db import get_db


class ChatMessage:
    """User question/answer pairs stored per user."""

    @staticmethod
    def create(user_id, question: str, answer: str, sources: list) -> dict:
        """Create a chat message record."""
        db = get_db()
        message_doc = {
            "_id": ObjectId(),
            "user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id,
            "question": question,
            "answer": answer,
            "sources": sources,  # List of file names or chunk sources
            "created_at": datetime.utcnow(),
        }
        result = db.chat_history.insert_one(message_doc)
        message_doc["_id"] = result.inserted_id
        return message_doc

    @staticmethod
    def find_by_user(user_id, limit: int = 50) -> list:
        """Get chat history for a user."""
        db = get_db()
        return list(
            db.chat_history.find(
                {"user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id}
            )
            .sort("created_at", -1)
            .limit(limit)
        )

    @staticmethod
    def delete_by_user(user_id):
        """Clear chat history for a user."""
        db = get_db()
        return db.chat_history.delete_many(
            {"user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id}
        )
