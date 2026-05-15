# Quick Start Guide - Backend Setup

## Step 1: Setup Secrets File

Create `.streamlit/secrets.toml` in the project root:

```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
# MongoDB Connection
MONGODB_URI = "mongodb://localhost:27017"
MONGODB_DB_NAME = "student_knowledge_assistant"

# Gemini API
GEMINI_API_KEY = "your-actual-gemini-api-key"

# JWT Authentication
JWT_SECRET = "your-super-secret-key-change-this-in-production"
EOF
```

**Note:** Never commit `.streamlit/secrets.toml` (it's in .gitignore)

## Step 2: Install Dependencies

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt
```

## Step 3: Start MongoDB

### Option A: Using Docker (Recommended)
```bash
docker run -d --name mongodb -p 27017:27017 mongo:latest
```

### Option B: Using Homebrew (macOS)
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

### Option C: Using APT (Ubuntu/Debian)
```bash
sudo apt-get install -y mongodb
sudo systemctl start mongodb
```

## Step 4: Start Backend Server

### On Linux/Mac:
```bash
chmod +x start_backend.sh
./start_backend.sh
```

### On Windows:
```bash
start_backend.bat
```

### Or manually:
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## Step 5: Access the Backend

Open in browser:
- **API Documentation:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

## Step 6: Test with Frontend

Update `frontend/pages/auth_page.py` to use the new API:

```python
from frontend.api_client import BackendClient

# In your login handler:
client = BackendClient()
result = client.login(email, password)
st.session_state.token = result["token"]
```

---

## Folder Structure Summary

```
backend/
├── config.py                    # Configuration constants
├── __init__.py
├── db/
│   ├── connection.py           # MongoDB connection
│   └── __init__.py
├── models/                      # MongoDB document models
│   ├── user.py
│   ├── file.py
│   ├── chat.py
│   └── __init__.py
├── services/                    # Business logic
│   ├── auth_service.py         # User authentication
│   ├── file_service.py         # File upload, chunks
│   ├── retrieval_service.py    # Search, chat history
│   └── __init__.py
└── api/                         # FastAPI endpoints
    ├── auth.py                 # /api/auth/*
    ├── files.py                # /api/files/*
    ├── retrieval.py            # /api/retrieval/*
    └── __init__.py

frontend/
└── api_client.py               # Client for backend API
```

---

## Backend Features

✅ **User Authentication**
- Register with email/password
- Login with JWT token
- Token-based API auth

✅ **File Upload & Processing**
- SHA-256 hash-based deduplication
- PDF text extraction
- Automatic chunking
- Metadata tracking in MongoDB

✅ **Smart Reuse**
- Check if file hash exists
- If yes → reuse chunks/embeddings
- If no → process once, store forever

✅ **Chat History**
- Save Q&A pairs per user
- Retrieve history
- Clear history

✅ **Data Persistence**
- All user data in MongoDB
- GridFS for large files (ready for PDFs)
- Chunk storage with source tracking

---

## API Examples

### Register User
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "password": "password123"
  }'
```

### Login
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com",
    "password": "password123"
  }'
```

### Upload File
```bash
curl -X POST http://localhost:8000/api/files/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@notes.pdf"
```

### List Files
```bash
curl -X GET http://localhost:8000/api/files/list \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## Troubleshooting

### MongoDB Connection Error
```bash
# Check if MongoDB is running
docker ps | grep mongodb

# Check connection
mongosh
> db.version()
```

### Port 27017 Already in Use
```bash
# Find and kill the process
lsof -i :27017
kill -9 <PID>
```

### FastAPI Server Won't Start
```bash
# Check if port 8000 is in use
lsof -i :8000

# If needed, use a different port
python -m uvicorn server:app --port 8001
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Next Steps

1. ✅ Backend structure created
2. ✅ MongoDB models implemented
3. ✅ API endpoints ready
4. ⏳ **Update Streamlit frontend to use backend APIs**
5. ⏳ Integrate Qdrant for vector similarity search
6. ⏳ Add embedding generation and storage

---

**Questions?** Check [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for detailed endpoint specs.
