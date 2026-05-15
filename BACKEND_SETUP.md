# Backend Setup Guide

## Prerequisites

### 1. MongoDB Installation

**Using Docker (Recommended):**
```bash
docker run -d --name mongodb -p 27017:27017 mongo:latest
```

**Using Homebrew (macOS):**
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

**Using APT (Ubuntu/Debian):**
```bash
sudo apt-get install -y mongodb
sudo systemctl start mongodb
```

### 2. Environment Variables

Create `.streamlit/secrets.toml`:
```
# MongoDB Connection
MONGODB_URI = "mongodb://localhost:27017"
MONGODB_DB_NAME = "student_knowledge_assistant"

# Gemini API
GEMINI_API_KEY = "your-api-key-here"

# JWT Secret (for authentication tokens)
JWT_SECRET = "your-super-secret-key-change-this"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Database Initialization

```python
from backend.db import init_db

# Initialize on app startup
init_db()
```

This will:
- Connect to MongoDB
- Create all collections
- Create all necessary indexes
- Verify connection

## Testing the Backend

### Quick Connection Test
```python
from backend.db import init_db, get_db

init_db()
db = get_db()
print(db.list_collection_names())
```

### Create a Test User
```python
from backend.models import User
import hashlib

hashed = hashlib.sha256("password123".encode()).hexdigest()
user = User.create(
    name="Test User",
    email="test@example.com",
    hashed_password=hashed
)
print(user)
```

### Upload a Test File
```python
from backend.models import UploadedFile

file_record = UploadedFile.create(
    user_id=user["_id"],
    filename="test.pdf",
    file_hash="abc123def456",
    file_size=1024000
)
print(file_record)
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  (auth_page.py, workspace_page.py, components)              │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP Requests (FastAPI)
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Backend API (FastAPI)                           │
│  backend/api/                                               │
│  ├── auth.py          (register, login → JWT)              │
│  ├── files.py         (upload, list, delete)               │
│  ├── retrieval.py     (query vectors, get sources)        │
│  └── chat.py          (save/fetch conversation)           │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴──────────────┐
        │                           │
    ┌───▼──────────────┐    ┌──────▼──────────────┐
    │ MongoDB          │    │ Vector DB (Qdrant)  │
    │ (Persistent)     │    │ (Similarity Search) │
    │ - users          │    │ - vectors           │
    │ - uploaded_files │    │ - metadata          │
    │ - file_chunks    │    │ - fast retrieval    │
    │ - chat_history   │    │                     │
    └──────────────────┘    └─────────────────────┘
```

## Workflow: File Upload & Reuse

```
User Uploads File
    ↓
Frontend: Compute SHA-256 hash
    ↓
Backend: Check if file_hash exists in MongoDB
    ├─ YES (file seen before)
    │   ├─ Get chunks from file_chunks collection
    │   ├─ Get embeddings from vector DB
    │   └─ Return existing vectors for RAG
    │
    └─ NO (new file)
        ├─ Extract chunks using chunking.py
        ├─ Generate embeddings using embeddings.py
        ├─ Save chunks to MongoDB
        ├─ Save vectors to Qdrant
        ├─ Mark file as processed
        └─ Return vectors for RAG
```

## API Endpoints (To be implemented)

### Authentication
```
POST /api/auth/register
  body: {name, email, password}
  response: {user_id, token}

POST /api/auth/login
  body: {email, password}
  response: {user_id, token}
```

### File Management
```
POST /api/files/upload
  headers: Authorization: Bearer {token}
  body: multipart/form-data {file}
  response: {file_id, filename, file_hash}

GET /api/files
  headers: Authorization: Bearer {token}
  response: [{file_id, filename, uploaded_at}, ...]

DELETE /api/files/{file_id}
  headers: Authorization: Bearer {token}
  response: {deleted: true}
```

### Retrieval & Chat
```
POST /api/retrieval/search
  headers: Authorization: Bearer {token}
  body: {query, top_k: 5}
  response: [{chunk, source, similarity}, ...]

POST /api/chat/ask
  headers: Authorization: Bearer {token}
  body: {question, selected_files: [file_ids]}
  response: {answer, sources}
```

## Security Considerations

✓ Passwords hashed with SHA-256 (or bcrypt)
✓ JWT tokens for API authentication
✓ User ID in request headers validated against token
✓ File operations scoped to authenticated user only
✓ Environment variables in `.streamlit/secrets.toml` (gitignored)

## Troubleshooting

### MongoDB Connection Failed
```
Check: mongosh
  mongosh > db.version()
```

### Port 27017 Already in Use
```
sudo lsof -i :27017
sudo kill -9 <PID>
```

### Indexes Not Created
```python
from backend.db.connection import _create_indexes, get_db
init_db()
_create_indexes()
```
