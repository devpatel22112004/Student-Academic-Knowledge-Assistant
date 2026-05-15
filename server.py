"""Main FastAPI application with MongoDB and backend services."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.db import init_db, close_db
from backend.api import auth_router, files_router, retrieval_router

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print("🚀 Starting backend server...")
    try:
        init_db()
        print("✓ Database initialized")
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        raise
    
    yield
    
    # Shutdown
    print("🛑 Shutting down backend server...")
    close_db()


# Create FastAPI app
app = FastAPI(
    title="Student Academic Knowledge Assistant",
    description="Backend API for file upload, processing, and RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (configure as needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(files_router)
app.include_router(retrieval_router)

# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Student Academic Knowledge Assistant",
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Student Academic Knowledge Assistant API",
        "docs": "/docs",
        "endpoints": {
            "auth": "/api/auth",
            "files": "/api/files",
            "retrieval": "/api/retrieval",
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
