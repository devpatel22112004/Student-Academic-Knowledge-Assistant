# Backend Architecture & MongoDB Schema

## Overview

Backend structure for MongoDB-based authentication, file tracking, and persistent storage of processed embeddings.

```
backend/
├── db/                      # MongoDB connection & initialization
│   ├── __init__.py
│   └── connection.py        # Connection, init_db(), indexes
├── models/                  # MongoDB document models
│   ├── __init__.py
│   ├── user.py             # User authentication & profile
│   ├── file.py             # UploadedFile & FileChunk
│   └── chat.py             # ChatMessage history
├── services/                # Business logic (auth, file processing)
├── api/                      # REST API endpoints
└── __init__.py
```

## MongoDB Collections

### 1. `users`
```javascript
{
  _id: ObjectId,
  name: String,
  email: String (unique, lowercase),
  password: String (SHA-256 hash),
  created_at: DateTime,
  updated_at: DateTime
}
```
**Indexes:**
- `email` (unique)

---

### 2. `uploaded_files`
Tracks files uploaded by each user.

```javascript
{
  _id: ObjectId,
  user_id: ObjectId,
  filename: String,
  file_hash: String (SHA-256 of file content),
  file_size: Int,
  gridfs_id: ObjectId (reference to GridFS file),
  uploaded_at: DateTime,
  processed: Boolean (chunks extracted?),
  processed_at: DateTime
}
```
**Indexes:**
- `(user_id, file_hash)` (unique)
- `(user_id, uploaded_at)` (for listing user's files)
- `file_hash` (for duplicate detection)

**Flow:**
1. User uploads file → create record with `processed: false`.
2. Chunks extracted → update `processed: true, processed_at: now()`.
3. Next upload of same file → check `file_hash`, reuse chunks/embeddings.

---

### 3. `file_chunks`
Extracted text chunks from processed files.

```javascript
{
  _id: ObjectId,
  file_hash: String,
  chunk_index: Int,
  text: String,
  source: String (e.g., "file.pdf page 5"),
  created_at: DateTime
}
```
**Indexes:**
- `(file_hash, chunk_index)` (unique)
- `file_hash`

**Notes:**
- Keyed by `file_hash` so same file across different users shares chunks.
- If embedding model is versioned, store model_name too.

---

### 4. `chat_history`
User question/answer pairs.

```javascript
{
  _id: ObjectId,
  user_id: ObjectId,
  question: String,
  answer: String,
  sources: Array<String>,
  created_at: DateTime
}
```
**Indexes:**
- `(user_id, created_at)` (for listing user's chats)

---

### 5. `file_embeddings` (Optional, if storing vectors in MongoDB)
For smaller projects; otherwise use dedicated vector DB (Qdrant, etc).

```javascript
{
  _id: ObjectId,
  file_hash: String,
  chunk_index: Int,
  model: String (e.g., "sentence-transformers/all-mpnet-base-v2"),
  model_version: String,
  embedding_dimension: Int,
  vector: Array<Float> (e.g., [0.1, 0.2, ...]),
  text_preview: String (first 200 chars),
  created_at: DateTime
}
```
**Indexes:**
- `(file_hash, chunk_index, model)` (unique)
- `file_hash`

---

## Initialization

```python
from backend.db import init_db

# On application startup
init_db()
```

This will:
1. Connect to MongoDB.
2. Create all collections if missing.
3. Create indexes.

## Environment Variables

`.env.example` or `.streamlit/secrets.toml`:
```
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=student_knowledge_assistant
```

## Models Usage

```python
from backend.models import User, UploadedFile, FileChunk, ChatMessage

# Create user
user = User.create(name="John", email="john@example.com", hashed_password=hash)

# Upload file
file_record = UploadedFile.create(
    user_id=user["_id"],
    filename="notes.pdf",
    file_hash="abc123def...",
    file_size=1024000,
    gridfs_id=gridfs_file_id
)

# Create chunks
FileChunk.bulk_create(
    file_hash="abc123def...",
    chunks=[
        {"chunk_index": 0, "text": "Chapter 1...", "source": "notes.pdf page 1"},
        {"chunk_index": 1, "text": "Chapter 2...", "source": "notes.pdf page 2"},
    ]
)

# Mark file as processed
UploadedFile.mark_processed("abc123def...")

# Get user's files
files = UploadedFile.find_by_user(user["_id"])

# Chat history
ChatMessage.create(
    user_id=user["_id"],
    question="What is...",
    answer="Answer is...",
    sources=["notes.pdf page 1", "notes.pdf page 3"]
)
```

## Next Steps

1. Implement auth service (register/login with JWT).
2. Implement file service (upload, hash, chunk extraction).
3. Create REST API endpoints.
4. Connect Streamlit frontend to backend API.
5. Add vector DB (Qdrant) for embeddings storage.
