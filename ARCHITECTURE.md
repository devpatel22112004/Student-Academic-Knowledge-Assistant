# Database Connection Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                         │
│  (Runs on port 8501)                                        │
│  - auth_page.py (Login/Register)                           │
│  - workspace_page.py (Main app)                            │
│  - Components (sidebar, navbar, chat)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ HTTP Requests
                     │ (frontend/api_client.py)
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   FastAPI Backend                           │
│  (Runs on port 8000)                                       │
│  server.py - Main application                             │
│                                                             │
│  backend/api/                                              │
│  ├── auth.py                                              │
│  │   POST /api/auth/register                             │
│  │   POST /api/auth/login                                │
│  │   GET  /api/auth/profile                              │
│  │                                                        │
│  ├── files.py                                            │
│  │   POST /api/files/upload                              │
│  │   GET  /api/files/list                                │
│  │   DELETE /api/files/{file_id}                         │
│  │   GET  /api/files/chunks/{file_hash}                  │
│  │                                                        │
│  └── retrieval.py                                        │
│      POST /api/retrieval/search                          │
│      POST /api/retrieval/chat                            │
│      GET  /api/retrieval/history                         │
│      DELETE /api/retrieval/history                       │
│                                                             │
│  backend/services/                                         │
│  ├── auth_service.py                                     │
│  │   - register_user()                                   │
│  │   - login_user()                                      │
│  │   - generate_token() [JWT]                            │
│  │   - verify_token()                                    │
│  │                                                        │
│  ├── file_service.py                                     │
│  │   - upload_file()                                     │
│  │   - compute_file_hash() [SHA-256]                    │
│  │   - get_user_files()                                 │
│  │   - delete_file()                                    │
│  │                                                        │
│  └── retrieval_service.py                                │
│      - search_chunks()                                   │
│      - save_chat()                                       │
│      - get_chat_history()                               │
│      - clear_chat_history()                             │
│                                                             │
│  backend/db/                                               │
│  └── connection.py                                        │
│      - init_db() [MongoDB initialization]                │
│      - get_db()                                          │
│      - _create_indexes()                                 │
│                                                             │
│  backend/models/                                           │
│  ├── user.py                                             │
│  │   - User.create()                                    │
│  │   - User.find_by_email()                             │
│  │   - User.find_by_id()                                │
│  │                                                        │
│  ├── file.py                                            │
│  │   - UploadedFile.create()                            │
│  │   - UploadedFile.find_by_hash()                      │
│  │   - FileChunk.create()                               │
│  │   - FileChunk.bulk_create()                          │
│  │                                                        │
│  └── chat.py                                            │
│      - ChatMessage.create()                             │
│      - ChatMessage.find_by_user()                       │
│                                                             │
│  backend/config.py                                        │
│  - Configuration constants                              │
│  - MONGODB_URI, JWT_SECRET, etc.                       │
└────────────────────┬───────────────────┬──────────────────┘
                     │                   │
        ┌────────────▼─────┐    ┌───────▼─────────┐
        │   MongoDB        │    │   Qdrant        │
        │   (Port 27017)   │    │   (Optional)    │
        │                  │    │                 │
        │ Collections:     │    │ Vector storage  │
        │ - users          │    │ (Future)        │
        │ - uploaded_files │    │                 │
        │ - file_chunks    │    │                 │
        │ - chat_history   │    │                 │
        │                  │    │                 │
        │ Indexes:         │    │                 │
        │ - email (unique) │    │                 │
        │ - user_id       │    │                 │
        │ - file_hash     │    │                 │
        │ - timestamps    │    │                 │
        └──────────────────┘    └─────────────────┘
```

---

## Data Flow Examples

### 1. User Registration Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User fills form in auth_page.py                         │
│    name: "John Doe"                                         │
│    email: "john@example.com"                               │
│    password: "password123"                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ POST /api/auth/register
                       │ {name, email, password}
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. FastAPI auth endpoint (backend/api/auth.py)             │
│    - Validates email format                                │
│    - Passes to auth_service.register_user()              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Auth Service (backend/services/auth_service.py)         │
│    - Normalize email (lowercase)                           │
│    - Check if user exists in MongoDB                       │
│    - Hash password (SHA-256)                              │
│    - Call User.create()                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. User Model (backend/models/user.py)                     │
│    - Insert document into 'users' collection              │
│    MongoDB: {                                              │
│      _id: ObjectId(...),                                  │
│      name: "John Doe",                                    │
│      email: "john@example.com",                          │
│      password: "abc123... (hash)",                       │
│      created_at: DateTime,                               │
│      updated_at: DateTime                                │
│    }                                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. Generate JWT Token                                       │
│    - Generate token with user_id                          │
│    - Expiration: 30 days                                  │
│    - Return to frontend                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Response: {user_id, token}
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. Frontend stores token                                    │
│    st.session_state.token = token                         │
│    st.session_state.user_id = user_id                    │
│    Redirect to workspace_page.py                          │
└──────────────────────────────────────────────────────────────┘
```

### 2. File Upload & Deduplication Flow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. User uploads file from sidebar.py                        │
│    File: notes.pdf (1 MB)                                  │
│    Token: Bearer abc123...                                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ POST /api/files/upload
                       │ multipart/form-data {file}
                       │ Authorization: Bearer token
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. FastAPI file endpoint (backend/api/files.py)            │
│    - Extract token from header                            │
│    - Verify token → get user_id                          │
│    - Read file bytes                                      │
│    - Call file_service.upload_file()                     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. File Service (backend/services/file_service.py)         │
│    - Compute SHA-256 hash of file                        │
│    - file_hash = "abc123def456..."                       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Check Deduplication                                      │
│    - UploadedFile.find_by_hash("abc123...")              │
│    - Query MongoDB: uploaded_files collection            │
│                                                            │
│    CASE A: File exists and is processed                  │
│    - Reuse chunks from FileChunk collection             │
│    - Create file reference for this user                 │
│    - Response: is_new=false, chunks_count=25            │
│                                                            │
│    CASE B: File is new                                  │
│    - Continue to processing                            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼ (CASE B only)
┌──────────────────────────────────────────────────────────────┐
│ 5. Extract Text & Create Chunks                            │
│    - extract_text_from_pdf() → "Chapter 1..."           │
│    - chunk_text() → 25 chunks (size=1000, overlap=200)  │
│    Each chunk: {                                          │
│      chunk_index: 0,                                      │
│      text: "Chapter 1 content...",                       │
│      source: "notes.pdf page 1"                         │
│    }                                                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. Save to MongoDB                                          │
│                                                             │
│    uploaded_files collection:                             │
│    {                                                       │
│      _id: ObjectId(...),                                 │
│      user_id: ObjectId(user_id),                        │
│      filename: "notes.pdf",                             │
│      file_hash: "abc123...",                            │
│      file_size: 1000000,                                │
│      processed: true,                                   │
│      processed_at: DateTime,                            │
│      uploaded_at: DateTime                              │
│    }                                                      │
│                                                             │
│    file_chunks collection (25 documents):                │
│    {                                                       │
│      _id: ObjectId(...),                                 │
│      file_hash: "abc123...",                            │
│      chunk_index: 0,                                     │
│      text: "Chapter 1...",                              │
│      source: "notes.pdf page 1",                       │
│      created_at: DateTime                               │
│    }                                                      │
│    ... (24 more chunks)                                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ Response: {file_id, chunks_count=25}
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. Frontend shows success                                   │
│    ✓ File uploaded: notes.pdf                             │
│    ✓ 25 chunks created (or reused)                       │
│    File added to user's file list                         │
└──────────────────────────────────────────────────────────────┘
```

### 3. Search & Chat Flow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. User types question in chat_panel.py                    │
│    "What is machine learning?"                             │
│    Selected files: ["abc123...", "xyz789..."]            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ POST /api/retrieval/search
                       │ {query, file_hashes, top_k=5}
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Retrieval endpoint (backend/api/retrieval.py)           │
│    - Verify token → get user_id                          │
│    - Call retrieval_service.search_chunks()             │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Search Service (backend/services/retrieval_service.py)  │
│    - For each file_hash:                                  │
│      - FileChunk.find_by_file_hash()                     │
│      - Query MongoDB for matching chunks                │
│    - Return top 5 most relevant                         │
│      (Currently: first 5 by index)                       │
│      (Future: use vector similarity with Qdrant)         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ Response: [chunks with similarity scores]
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Frontend sends chunks to Gemini API                     │
│    - Chunks + question → src/services/gemini_service.py  │
│    - Gemini generates answer                             │
│    - Display sources in source cards                     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ Save Q&A pair
                       │ POST /api/retrieval/chat
                       │ {question, answer, file_hashes}
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. Save to Chat History (MongoDB)                          │
│    chat_history collection: {                             │
│      _id: ObjectId(...),                                 │
│      user_id: ObjectId(user_id),                        │
│      question: "What is machine learning?",             │
│      answer: "Machine learning is...",                  │
│      sources: ["notes.pdf page 5", ...],               │
│      created_at: DateTime                               │
│    }                                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## Database Schema at a Glance

### users
```javascript
{
  _id: ObjectId,
  name: String,
  email: String (unique),
  password: String (SHA-256 hash),
  created_at: DateTime,
  updated_at: DateTime
}
```

### uploaded_files
```javascript
{
  _id: ObjectId,
  user_id: ObjectId,
  filename: String,
  file_hash: String,
  file_size: Int,
  processed: Boolean,
  uploaded_at: DateTime,
  processed_at: DateTime
}
```

### file_chunks
```javascript
{
  _id: ObjectId,
  file_hash: String,
  chunk_index: Int,
  text: String,
  source: String,
  created_at: DateTime
}
```

### chat_history
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

---

## Key Advantages

✅ **Persistent State** - No more Streamlit session memory loss on refresh  
✅ **User Isolation** - Each user's data is separate  
✅ **File Deduplication** - Same file across users reuses chunks/embeddings  
✅ **API-Based** - Can be called from any client (web, mobile, CLI)  
✅ **Scalable** - MongoDB handles millions of documents  
✅ **JWT Auth** - Secure token-based authentication  
✅ **Chat History** - Persistent conversation logs  

---

## Ready for Production?

- ✅ MongoDB schema with indexes
- ✅ User authentication with JWT
- ✅ File upload with deduplication
- ✅ API endpoints
- ⏳ Vector DB integration (Qdrant)
- ⏳ Embedding caching
- ⏳ Error handling & logging
- ⏳ Rate limiting & API keys
