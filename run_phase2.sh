#!/usr/bin/env bash
set -e
set -o pipefail

# =====================================================
# Phase 2 Direct Runner (Document Processing Pipeline)
# =====================================================
# Is script ko direct run karke aap Phase 2 indexing kar sakte ho.
# Short: Text ko searchable vectors me convert karta hai.
# Concept: Ye retrieval backbone banata hai - yahi data later semantic
# question answering me nearest relevant context return karta hai.
# Ye script ye steps karta hai:
# 1) Virtual environment activate karta hai (agar .venv present ho)
# 2) (Optional) Required packages install/update karta hai
# 3) document_pipeline.py run karta hai
# 4) Default me clean output ke liye quiet mode use hota hai
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
#   --install-deps = pip install -r requirements.txt run karega (optional)
#   --verbose      = extra logs/warnings dekhna ho to
#   --show-progress= embedding progress bar dekhna ho to
#   $1 = input path (PDF/TXT file ya folder) [default: data/pdfs]
#   $2 = output path [default: outputs/vector_store]
#   $3 = chunk size [default: 600]
#   $4 = chunk overlap [default: 100]
#   $5 = embedding model [default: sentence-transformers/all-MiniLM-L6-v2]

INSTALL_DEPS=false
VERBOSE=false
SHOW_PROGRESS=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --install-deps)
      INSTALL_DEPS=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --show-progress)
      SHOW_PROGRESS=true
      shift
      ;;
    *)
      break
      ;;
  esac
done

INPUT_PATH="${1:-data/pdfs}"
OUTPUT_PATH="${2:-outputs/vector_store}"
CHUNK_SIZE="${3:-600}"
CHUNK_OVERLAP="${4:-100}"
MODEL_NAME="${5:-sentence-transformers/all-MiniLM-L6-v2}"

# Chunking defaults trade-off:
# - chunk size bada -> context richer, but granularity coarse.
# - overlap bada -> continuity better, but duplicate tokens increase.

if [[ -d ".venv" ]]; then
  echo "[INFO] Activating virtual environment (.venv)"
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[WARN] .venv not found. Running with current Python environment."
fi

if [[ "$INSTALL_DEPS" == "true" ]]; then
  echo "[INFO] Installing dependencies from requirements.txt"
  # Dependency sync ensures local run and CI behavior consistent rahe.
  pip install --no-cache-dir -r requirements.txt
else
  echo "[INFO] Skipping dependency install (already installed assumed)."
  echo "[INFO] If needed, run: bash run_phase2.sh --install-deps"
  # Manual fallback command (sirf issue aaye tab):
  # pip install -r requirements.txt
fi

echo "[INFO] Running Phase 2 pipeline"
CMD=(python scripts/document_pipeline.py \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --chunk-size "$CHUNK_SIZE" \
  --chunk-overlap "$CHUNK_OVERLAP" \
  --model "$MODEL_NAME")

if [[ "$VERBOSE" == "true" ]]; then
  CMD+=(--verbose)
fi

if [[ "$SHOW_PROGRESS" == "true" ]]; then
  CMD+=(--show-progress)
fi

if [[ "$VERBOSE" == "true" ]]; then
  # Verbose mode debugging/troubleshooting ke liye best hai.
  "${CMD[@]}"
else
  # Quiet mode me sirf known harmless noise hide karte hain:
  # - HF unauthenticated warning (token na ho tab aata hai)
  # - model loading progress/report lines
  # Real errors/traceback filter nahi kiye jaate.
  "${CMD[@]}" 2>&1 \
    | grep -vF 'Warning: You are sending unauthenticated requests to the HF Hub' \
    | grep -vF 'Loading weights:' \
    | grep -vF 'BertModel LOAD REPORT from:' \
    | grep -vF 'embeddings.position_ids | UNEXPECTED' \
    | grep -vF 'UNEXPECTED    :can be ignored' \
    | grep -Ev '^Key[[:space:]]+\|[[:space:]]+Status|^-+$'
fi

echo "[DONE] Phase 2 indexing complete."
echo "[DONE] Output folder: $OUTPUT_PATH"
