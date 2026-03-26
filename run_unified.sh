#!/bin/bash

###############################################################################
# UNIFIED SCRIPT: Phase 1 + Phase 2 in ONE command
# Handles BOTH PDF and TXT files automatically
###############################################################################
# Short: Ek command me extraction + indexing complete.
# Concept: Operational complexity kam hoti hai - user ko phase-order ya
# intermediate wiring manually manage nahi karni padti.

set -e

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# Input/Output paths
INPUT_FOLDER="${1:-data}"
EXTRACTED_FOLDER="${2:-outputs/extracted}"
VECTOR_STORE_FOLDER="${3:-outputs/vector_store}"

# Chunking parameters
CHUNK_SIZE="${4:-600}"
CHUNK_OVERLAP="${5:-100}"
MODEL_NAME="${6:-sentence-transformers/all-MiniLM-L6-v2}"

# Ye parameters directly Phase 2 semantic retrieval behavior influence karte hain.

echo "[UNIFIED] Running complete pipeline..."
echo "[UNIFIED] Input: $INPUT_FOLDER"
echo "[UNIFIED] Output: $VECTOR_STORE_FOLDER"
echo ""

# STEP 1: Phase 1 - Extract text from all documents (PDF + TXT)
echo "[STEP 1/2] Running Phase 1: Document Extraction..."
# Input folder ke andar mixed sources (PDF/TXT) auto-discover hote hain.
python3 scripts/pdf_loader.py --input "$INPUT_FOLDER" --output "$EXTRACTED_FOLDER"

# STEP 2: Phase 2 - Create vector store from extracted text
echo "[STEP 2/2] Running Phase 2: Embeddings + Vector Store..."
# Phase 1 ka normalized text pass karne se pipeline deterministic rehti hai.
python3 scripts/document_pipeline.py --input "$EXTRACTED_FOLDER" --output "$VECTOR_STORE_FOLDER" --chunk-size "$CHUNK_SIZE" --chunk-overlap "$CHUNK_OVERLAP" --model "$MODEL_NAME"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "[DONE] ✅ COMPLETE PIPELINE SUCCESS!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📁 Extracted documents: $EXTRACTED_FOLDER"
echo "📊 Vector store index: $VECTOR_STORE_FOLDER"
echo ""
echo "Usage:"
echo "  bash run_unified.sh [input_folder] [extracted_folder] [vector_folder]"
echo "  bash run_unified.sh data outputs/extracted outputs/vector_store (custom)"
echo "  bash run_unified.sh (uses defaults)"
echo ""
echo "What was created:"
echo "  ✓ Extracted text files (Phase 1 output)"
echo "  ✓ FAISS vector index (Phase 2 output)"
echo "  ✓ Metadata file with chunk references"
echo ""
