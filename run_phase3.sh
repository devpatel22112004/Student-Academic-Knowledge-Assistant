#!/usr/bin/env bash
set -e
set -o pipefail

# ==========================================
# Phase 3 Runner (Retrieval + Evaluation)
# ==========================================
# Purpose: Run semantic search and retrieval-quality checks.
#
# Examples:
#   bash run_phase3.sh query --query "What is Tesla?"
#   bash run_phase3.sh query --query "electric vehicles" --top-k 3 --json
#   bash run_phase3.sh eval --top-k 5 --sample-size 30
#
# Optional setup:
#   bash run_phase3.sh --install-deps query --query "..."

INSTALL_DEPS=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --install-deps)
      INSTALL_DEPS=true
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ -d ".venv" ]]; then
  echo "[INFO] Activating virtual environment (.venv)"
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[WARN] .venv not found. Running with current Python environment."
fi

if [[ "$INSTALL_DEPS" == "true" ]]; then
  echo "[INFO] Installing dependencies from requirements.txt"
  pip install --no-cache-dir -r requirements.txt
fi

if [[ "$#" -eq 0 ]]; then
  echo "[ERROR] Missing command. Use: query or eval"
  echo "[HINT] Example: bash run_phase3.sh query --query \"What is Tesla?\""
  exit 1
fi

echo "[INFO] Running Phase 3 retrieval system"
python scripts/retrieval_system.py "$@"

echo "[DONE] Phase 3 completed."
