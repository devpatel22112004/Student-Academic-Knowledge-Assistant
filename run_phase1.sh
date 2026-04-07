#!/usr/bin/env bash
# Stop script on first command failure.
set -e
# Fail pipeline if any command in pipe fails.
set -o pipefail

# ==============================================
# Phase 1 Direct Runner (PDF/TXT/RTF/DOC/DOCX Text Extraction)
# ==============================================
# Purpose: Run Phase 1 extraction in a consistent way.
# Flow: activate env -> optional deps install -> run `pdf_loader.py`.
#
# Default usage:
#   bash run_phase1.sh
#
# First-time setup or dependency issue ke time:
#   bash run_phase1.sh --install-deps
#
# Custom usage:
#   bash run_phase1.sh data outputs/extracted
#
# Parameters:
#   --install-deps  Install dependencies from requirements.txt
#   $1              Input file/folder (default: data)
#   $2              Output folder (default: outputs/extracted)

INSTALL_DEPS=false

# Parse optional flags first.
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    # Install dependencies mode.
    --install-deps)
      INSTALL_DEPS=true
      shift
      ;;
    # Remaining args are positional input/output.
    *)
      break
      ;;
  esac
done

# Input defaults to full data folder.
INPUT_PATH="${1:-data}"
# Output defaults to extracted folder.
OUTPUT_PATH="${2:-outputs/extracted}"

# Default input is `data` so root + subfolder documents are discovered.
# NOTE: `.doc` support requires system tool `antiword`.

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
  echo "[INFO] If needed, run: bash run_phase1.sh --install-deps"
fi

echo "[INFO] Running Phase 1 PDF loader"
# Extraction routing is handled inside the Python script.
# Run Phase 1 loader with resolved input/output paths.
python scripts/pdf_loader.py --input "$INPUT_PATH" --output "$OUTPUT_PATH"

# Success summary.
echo "[DONE] Phase 1 extraction complete."
echo "[DONE] Output folder: $OUTPUT_PATH"
