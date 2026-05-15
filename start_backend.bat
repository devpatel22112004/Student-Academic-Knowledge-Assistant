@echo off
REM Startup script for Student Academic Knowledge Assistant (Windows)
REM Starts MongoDB and FastAPI backend server

setlocal enabledelayedexpansion

echo ==================================================
echo   Student Academic Knowledge Assistant
echo ==================================================
echo.

REM Colors would be more complex in batch, so we'll skip them

echo [1/4] Checking MongoDB...
docker ps --format "{{.Names}}" | findstr "mongodb" > nul
if errorlevel 1 (
    echo Starting MongoDB in Docker...
    docker run -d --name mongodb -p 27017:27017 mongo:latest > nul 2>&1
    timeout /t 2 /nobreak > nul
    echo MongoDB started
) else (
    echo MongoDB is already running
)

echo.
echo [2/4] Checking Python dependencies...
pip install -q -r requirements.txt
echo Dependencies ready

echo.
echo [3/4] Configuration:
echo   MongoDB: mongodb://localhost:27017
echo   API URL: http://localhost:8000
echo   Docs: http://localhost:8000/docs

echo.
echo [4/4] Checking secrets configuration...
if not exist ".streamlit\secrets.toml" (
    echo .streamlit\secrets.toml not found
    echo.
    echo Please create .streamlit\secrets.toml with:
    echo.
    echo MONGODB_URI = "mongodb://localhost:27017"
    echo MONGODB_DB_NAME = "student_knowledge_assistant"
    echo GEMINI_API_KEY = "your-api-key-here"
    echo JWT_SECRET = "your-super-secret-key-change-this"
    echo.
    pause
) else (
    echo Secrets configuration found
)

echo.
echo ==================================================
echo   Starting FastAPI Backend Server
echo ==================================================
echo.
echo Backend will be available at:
echo   - API: http://localhost:8000
echo   - Interactive Docs: http://localhost:8000/docs
echo   - ReDoc: http://localhost:8000/redoc
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
