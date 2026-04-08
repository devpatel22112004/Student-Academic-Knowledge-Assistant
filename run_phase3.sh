#!/usr/bin/env bash
# Stop script on first command failure.
set -e
# Fail pipeline if any command in pipe fails.
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

# Parse optional flags first.
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    # Install dependencies mode.
    --install-deps)
      INSTALL_DEPS=true
      shift
      ;;
    # Remaining args belong to query/eval subcommand.
    *)
      break
      ;;
  esac
done

# Activate local virtual environment if present.
if [[ -d ".venv" ]]; then
  echo "[INFO] Activating virtual environment (.venv)"
  # shellcheck disable=SC1091
  # Load venv into current shell.
  source .venv/bin/activate
else
  # Continue with system Python when venv is missing.
  echo "[WARN] .venv not found. Running with current Python environment."
fi

# Install dependencies only when requested.
if [[ "$INSTALL_DEPS" == "true" ]]; then
  echo "[INFO] Installing dependencies from requirements.txt"
  # Install/update project dependencies.
  pip install --no-cache-dir -r requirements.txt
fi

# Require at least one subcommand (query/eval).
if [[ "$#" -eq 0 ]]; then
  echo "[ERROR] Missing command. Use: query or eval"
  echo "[HINT] Example: bash run_phase3.sh query --query \"What is Tesla?\""
  exit 1
fi

# Run Phase 3 Python script with all remaining user arguments.
echo "[INFO] Running Phase 3 retrieval system"
python scripts/retrieval_system.py "$@"

# Success summary.
echo "[DONE] Phase 3 completed."
