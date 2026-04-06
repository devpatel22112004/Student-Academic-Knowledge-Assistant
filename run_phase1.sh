#!/usr/bin/env bash
set -e
set -o pipefail

# ==============================================
# Phase 1 Direct Runner (PDF/TXT Text Extraction)
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
#   bash run_phase1.sh data outputs
#
# Parameters:
#   --install-deps  Install dependencies from requirements.txt
#   $1              Input file/folder (default: data)
#   $2              Output folder (default: outputs)

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

INPUT_PATH="${1:-data}"
OUTPUT_PATH="${2:-outputs}"

# Default input is `data` so both `data/tesla.txt` and `data/pdfs/*.pdf` work.

if [[ -d ".venv" ]]; then
  echo "[INFO] Activating virtual environment (.venv)"
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[WARN] .venv not found. Running with current Python environment."
fi

if [[ "$INSTALL_DEPS" == "true" ]]; then
  echo "[INFO] Installing dependencies from requirements.txt"
  # Keep environment reproducible.
  pip install --no-cache-dir -r requirements.txt
else
  echo "[INFO] Skipping dependency install (already installed assumed)."
  echo "[INFO] If needed, run: bash run_phase1.sh --install-deps"
fi

echo "[INFO] Running Phase 1 PDF loader"
# Extraction routing is handled inside the Python script.
python scripts/pdf_loader.py --input "$INPUT_PATH" --output "$OUTPUT_PATH"

echo "[DONE] Phase 1 extraction complete."
echo "[DONE] Output folder: $OUTPUT_PATH"
