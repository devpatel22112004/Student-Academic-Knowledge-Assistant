# API Documentation

## Base URL

```
http://localhost:8000
```

## Endpoints Overview

### Authentication (`/api/auth`)

#### POST `/api/auth/register`
Register a new user.

**Request:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "user_id": "507f1f77bcf86cd799439011",
  "name": "John Doe",
  "email": "john@example.com",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

#### POST `/api/auth/login`
Login user and get JWT token.

**Request:**
```json
{
  "email": "john@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "user_id": "507f1f77bcf86cd799439011",
  "name": "John Doe",
  "email": "john@example.com",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

#### GET `/api/auth/profile`
Get user profile information.

**Query Parameters:**
- `user_id` (required): User's ID

**Headers:**
```
Authorization: Bearer {token}
```

**Response:**
```json
{
  "user_id": "507f1f77bcf86cd799439011",
  "name": "John Doe",
  "email": "john@example.com",
  "created_at": "2024-01-15T10:30:00"
}
```

---

### Files (`/api/files`)

#### POST `/api/files/upload`
Upload a PDF file for processing.

**Headers:**
```
Authorization: Bearer {token}
Content-Type: multipart/form-data
```

**Form Data:**
- `file` (required): PDF file

**Response:**
```json
{
  "file_id": "507f1f77bcf86cd799439012",
  "filename": "notes.pdf",
  "file_hash": "abc123def456...",
  "is_new": true,
  "chunks_count": 25,
  "message": "File processed, 25 chunks created"
}
```

#### GET `/api/files/list`
List all files uploaded by user.

**Headers:**
```
Authorization: Bearer {token}
```

**Response:**
```json
[
  {
    "file_id": "507f1f77bcf86cd799439012",
    "filename": "notes.pdf",
    "file_hash": "abc123def456...",
    "file_size": 1024000,
    "uploaded_at": "2024-01-15T10:30:00",
    "processed": true
  },
  {
    "file_id": "507f1f77bcf86cd799439013",
    "filename": "lecture.pdf",
    "file_hash": "xyz789...",
    "file_size": 2048000,
    "uploaded_at": "2024-01-16T14:15:00",
    "processed": true
  }
]
```

#### DELETE `/api/files/{file_id}`
Delete a file from user's library.

**Headers:**
```
Authorization: Bearer {token}
```

**Response:**
```json
{
  "deleted": true,
  "file_id": "507f1f77bcf86cd799439012",
  "message": "File removed from your library (chunks kept for reuse)"
}
```

#### GET `/api/files/chunks/{file_hash}`
Get text chunks for a specific file.

**Query Parameters:**
- `limit` (optional): Max number of chunks to return

**Response:**
```json
{
  "chunks": [
    {
      "chunk_id": "507f1f77bcf86cd799439014",
      "chunk_index": 0,
      "text": "Chapter 1: Introduction to Machine Learning...",
      "source": "notes.pdf page 1"
    },
    {
      "chunk_id": "507f1f77bcf86cd799439015",
      "chunk_index": 1,
      "text": "1.1 What is Machine Learning...",
      "source": "notes.pdf page 2"
    }
  ]
}
```

---

### Retrieval (`/api/retrieval`)

#### POST `/api/retrieval/search`
Search for relevant chunks across files.

**Headers:**
```
Authorization: Bearer {token}
```

**Request:**
```json
{
  "query": "What is machine learning?",
  "file_hashes": ["abc123def456", "xyz789..."],
  "top_k": 5
}
```

**Response:**
```json
[
  {
    "chunk_id": "507f1f77bcf86cd799439014",
    "chunk_index": 0,
    "text": "Machine learning is a subset of artificial intelligence...",
    "source": "notes.pdf page 5",
    "similarity": 0.95
  },
  {
    "chunk_id": "507f1f77bcf86cd799439015",
    "chunk_index": 2,
    "text": "ML algorithms learn patterns from data...",
    "source": "notes.pdf page 6",
    "similarity": 0.92
  }
]
```

#### POST `/api/retrieval/chat`
Save a question-answer pair to chat history.

**Headers:**
```
Authorization: Bearer {token}
```

**Request:**
```json
{
  "question": "What is machine learning?",
  "answer": "Machine learning is...",
  "file_hashes": ["abc123def456"]
}
```

**Response:**
```json
{
  "message_id": "507f1f77bcf86cd799439020",
  "created_at": "2024-01-16T15:45:00"
}
```

#### GET `/api/retrieval/history`
Get user's chat history.

**Headers:**
```
Authorization: Bearer {token}
```

**Query Parameters:**
- `limit` (optional, default: 50): Max messages to return

**Response:**
```json
[
  {
    "message_id": "507f1f77bcf86cd799439020",
    "question": "What is machine learning?",
    "answer": "Machine learning is...",
    "sources": ["notes.pdf page 5"],
    "created_at": "2024-01-16T15:45:00"
  }
]
```

#### DELETE `/api/retrieval/history`
Clear all chat history for user.

**Headers:**
```
Authorization: Bearer {token}
```

**Response:**
```json
{
  "cleared": true,
  "messages_deleted": 15
}
```

---

## Authentication

### JWT Token
All protected endpoints require a Bearer token in the Authorization header:

```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

### Token Expiration
Tokens expire after 30 days. Users must re-login to get a new token.

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message description"
}
```

**Common Status Codes:**
- `200`: Success
- `201`: Created
- `400`: Bad request (validation error)
- `401`: Unauthorized (missing/invalid token)
- `404`: Not found
- `500`: Server error

---

## File Upload Deduplication

When uploading a file:

1. File hash (SHA-256) is computed
2. If file hash exists and is processed:
   - Response: `is_new: false`
   - Chunks are reused
   - No reprocessing
3. If file hash is new:
   - Response: `is_new: true`
   - File is processed
   - Chunks extracted and stored

This prevents duplicate processing and saves compute/storage.

---

## Using the API Client

From `frontend/api_client.py`:

```python
from frontend.api_client import BackendClient

# Check if backend is running
if BackendClient.health_check():
    print("✓ Backend is running")
else:
    print("✗ Backend is not running")

# Register user
client = BackendClient()
result = client.register(
    name="John Doe",
    email="john@example.com",
    password="password123"
)
token = result["token"]

# Create new client with token
client = BackendClient(token=token)

# Upload file
with open("notes.pdf", "rb") as f:
    upload_result = client.upload_file("notes.pdf", f.read())
    print(f"Uploaded: {upload_result['filename']}")

# List files
files = client.list_files()
print(f"You have {len(files)} files")

# Search chunks
results = client.search_chunks(
    query="What is machine learning?",
    file_hashes=[upload_result["file_hash"]],
    top_k=5
)
print(f"Found {len(results)} relevant chunks")

# Save to chat history
client.save_chat(
    question="What is machine learning?",
    answer="Machine learning is...",
    file_hashes=[upload_result["file_hash"]]
)
```

---

## Running the Backend

```bash
# Start MongoDB
docker run -d --name mongodb -p 27017:27017 mongo:latest

# Install dependencies
pip install -r requirements.txt

# Start server
python server.py
```

Server will be available at:
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
