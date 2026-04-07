#!/usr/bin/env bash
# Stop script on first command failure.
set -e
# Fail pipeline if any command in pipe fails.
set -o pipefail

# =====================================================
# Phase 2 Direct Runner (Document Processing Pipeline)
# =====================================================
# Purpose: Run Phase 2 indexing in one command.
# Flow: activate env -> optional deps install -> run `document_pipeline.py`.
# Default mode keeps logs clean; `--verbose` keeps full logs.
#
# Default usage:
#   bash run_phase2.sh
#
# First-time setup or dependency issue ke time:
#   bash run_phase2.sh --install-deps
#
# Custom usage (manual parameters):
#   bash run_phase2.sh data/pdfs outputs/vector_store 600 100 sentence-transformers/all-MiniLM-L6-v2
#
# Custom usage + dependency install:
#   bash run_phase2.sh --install-deps data/pdfs outputs/vector_store 600 100 sentence-transformers/all-MiniLM-L6-v2
#
# Parameters:
#   --install-deps   Install dependencies from requirements.txt
#   --verbose        Show detailed logs
#   --show-progress  Show embedding progress bar
#   $1               Input file/folder (default: data)
#   $2               Output folder (default: outputs/vector_store)
#   $3               Chunk size (default: 600)
#   $4               Chunk overlap (default: 100)
#   $5               Embedding model name

INSTALL_DEPS=false
VERBOSE=false
SHOW_PROGRESS=false

# Parse optional flags first.
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    # Install dependencies mode.
    --install-deps)
      INSTALL_DEPS=true
      shift
      ;;
    # Verbose logs mode.
    --verbose)
      VERBOSE=true
      shift
      ;;
    # Embedding progress bar mode.
    --show-progress)
      SHOW_PROGRESS=true
      shift
      ;;
    # Remaining args are positional input/output/chunk params.
    *)
      break
      ;;
  esac
done

# Input defaults to full data folder.
INPUT_PATH="${1:-data}"
# Output defaults to vector store folder.
OUTPUT_PATH="${2:-outputs/vector_store}"
# Chunk size default.
CHUNK_SIZE="${3:-600}"
# Chunk overlap default.
CHUNK_OVERLAP="${4:-100}"
# Default embedding model.
MODEL_NAME="${5:-sentence-transformers/all-MiniLM-L6-v2}"

# Chunking trade-off:
# - Larger chunk size keeps more context.
# - Larger overlap improves continuity.

if [[ -d ".venv" ]]; then
  echo "[INFO] Activating virtual environment (.venv)"
  # shellcheck disable=SC1091
  # Load virtual environment in current shell.
  source .venv/bin/activate
else
  # Continue with system Python if .venv does not exist.
  echo "[WARN] .venv not found. Running with current Python environment."
fi

if [[ "$INSTALL_DEPS" == "true" ]]; then
  echo "[INFO] Installing dependencies from requirements.txt"
  # Keep environment reproducible.
  # Install/update project dependencies.
  pip install --no-cache-dir -r requirements.txt
else
  # Skip install for faster repeated runs.
  echo "[INFO] Skipping dependency install (already installed assumed)."
  echo "[INFO] If needed, run: bash run_phase2.sh --install-deps"
  # Manual fallback:
  # pip install -r requirements.txt
fi

echo "[INFO] Running Phase 2 pipeline"
# Build python command as array to keep argument quoting safe.
CMD=(python scripts/document_pipeline.py \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --chunk-size "$CHUNK_SIZE" \
  --chunk-overlap "$CHUNK_OVERLAP" \
  --model "$MODEL_NAME")

if [[ "$VERBOSE" == "true" ]]; then
  # Add verbose flag when requested.
  CMD+=(--verbose)
fi

if [[ "$SHOW_PROGRESS" == "true" ]]; then
  # Add embedding progress flag when requested.
  CMD+=(--show-progress)
fi

if [[ "$VERBOSE" == "true" ]]; then
  # Print full runtime logs.
  # Run command directly in verbose mode.
  "${CMD[@]}"
else
  # Hide known noisy lines while keeping real errors visible.
  # Run command and filter expected non-critical noise lines.
  "${CMD[@]}" 2>&1 \
    | grep -vF 'Warning: You are sending unauthenticated requests to the HF Hub' \
    | grep -vF 'Loading weights:' \
    | grep -vF 'BertModel LOAD REPORT from:' \
    | grep -vF 'embeddings.position_ids | UNEXPECTED' \
    | grep -vF 'UNEXPECTED    :can be ignored' \
    | grep -Ev '^Key[[:space:]]+\|[[:space:]]+Status|^-+$'
fi

# Success summary.
echo "[DONE] Phase 2 indexing complete."
echo "[DONE] Output folder: $OUTPUT_PATH"
