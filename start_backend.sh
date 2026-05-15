#!/bin/bash

# Startup script for Student Academic Knowledge Assistant
# Starts MongoDB and FastAPI backend server

set -e

echo "=================================================="
echo "  Student Academic Knowledge Assistant"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if MongoDB is already running
echo -e "${BLUE}[1/4]${NC} Checking MongoDB..."
if docker ps --format '{{.Names}}' | grep -q '^mongodb$'; then
    echo -e "${GREEN}✓${NC} MongoDB is already running"
else
    echo -e "${YELLOW}⚡${NC} Starting MongoDB in Docker..."
    docker run -d --name mongodb -p 27017:27017 mongo:latest > /dev/null 2>&1 || true
    sleep 2
    echo -e "${GREEN}✓${NC} MongoDB started"
fi

# Check Python dependencies
echo ""
echo -e "${BLUE}[2/4]${NC} Checking Python dependencies..."
pip install -q -r requirements.txt 2>/dev/null || {
    echo -e "${YELLOW}⚠${NC} Some dependencies may not have installed"
}
echo -e "${GREEN}✓${NC} Dependencies ready"

# Show environment setup
echo ""
echo -e "${BLUE}[3/4]${NC} Configuration:"
echo "  MongoDB: mongodb://localhost:27017"
echo "  API URL: http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"

# Make sure .streamlit/secrets.toml exists
echo ""
echo -e "${BLUE}[4/4]${NC} Checking secrets configuration..."
if [ ! -f .streamlit/secrets.toml ]; then
    echo -e "${YELLOW}⚠${NC} .streamlit/secrets.toml not found"
    echo ""
    echo "Please create .streamlit/secrets.toml with:"
    echo ""
    echo "MONGODB_URI = \"mongodb://localhost:27017\""
    echo "MONGODB_DB_NAME = \"student_knowledge_assistant\""
    echo "GEMINI_API_KEY = \"your-api-key-here\""
    echo "JWT_SECRET = \"your-super-secret-key-change-this\""
    echo ""
    read -p "Press Enter to continue..."
else
    echo -e "${GREEN}✓${NC} Secrets configuration found"
fi

# Start FastAPI server
echo ""
echo -e "${GREEN}=================================================="
echo "  Starting FastAPI Backend Server"
echo "==================================================${NC}"
echo ""
echo "Backend will be available at:"
echo "  - API: http://localhost:8000"
echo "  - Interactive Docs: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
