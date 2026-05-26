# Student Academic Knowledge Assistant

An end-to-end RAG application that lets students upload study files and ask grounded questions from their own notes.

## Overview
This project provides a Streamlit-based academic assistant with:
- PDF/TXT ingestion
- chunking and embeddings
- Pinecone-based semantic retrieval
- Gemini Flash grounded generation
- extractive fallback answers
- per-user file history with duplicate detection
- file-scoped retrieval selection for faster focused queries

## Key Features
- Authentication UI (register/login) for workspace access
- Multi-file upload and processing pipeline
- Duplicate upload detection using content hash + Pinecone metadata
- Persistent vector storage in Pinecone
- Source-attributed answers with source preview cards
- Sidebar file history and multiselect scope filtering

## Tech Stack
- Python
- Streamlit
- sentence-transformers (`all-MiniLM-L6-v2`)
- Pinecone
- langchain-text-splitters
- pypdf
- google-generativeai (Gemini Flash)
- numpy
- python-dotenv

## Architecture
```
Student-Academic-Knowledge-Assistant/
├── app.py
├── main.py
├── frontend/
│   ├── components/
│   ├── pages/
│   └── ui/
├── src/
│   ├── core/
│   ├── services/
│   └── utils/
├── styles/
├── data/
├── requirements.txt
├── README.md
└── PROJECT_REPORT.md
```

## Environment Setup
1. Create and activate virtual environment.
2. Install dependencies.
3. Configure environment variables.

```bash
pip install -r requirements.txt
```

Create a `.env` file using `.env.example` and set:
- `GEMINI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_HOST`
- `PINECONE_INDEX_NAME`

## Run

Primary web app:
```bash
python app.py
```

Alternative launch:
```bash
python -m streamlit run app.py
```

CLI note:
- `main.py` is a legacy compatibility entrypoint and currently out of sync with the latest retrieval module.
- Use the Streamlit app as the official runtime path.

## User Workflow
1. Login or register.
2. Upload PDF/TXT files.
3. Click `Process Documents`.
4. App chunks text, creates embeddings, and stores vectors in Pinecone.
5. Ask questions in chat.
6. Optionally narrow retrieval by selecting specific files from `Your File History`.
7. View answer plus source details.

## Retrieval Flow
1. Question embedding generation
2. Pinecone query filtered by `user_id`
3. Optional filter by selected `file_hash` list
4. Top chunks returned as context
5. Gemini Flash grounded response
6. Fallback to local extractive answer if model call fails

## Data Notes
- User file history is stored in `data/user_files.json`.
- Vector metadata stores source, user_id, file_hash, chunk text, and timestamps.

## Project Report
Comprehensive final project documentation is available in [PROJECT_REPORT.md](PROJECT_REPORT.md).

## Current Status
Project is complete for the current scope:
- implemented upload-to-query RAG pipeline
- implemented duplicate-safe ingestion
- implemented file-scoped retrieval optimization
- implemented source-grounded answer UX