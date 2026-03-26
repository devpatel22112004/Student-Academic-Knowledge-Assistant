#!/usr/bin/env bash
set -e
set -o pipefail

# ==============================================
# Phase 1 Direct Runner (PDF/TXT Text Extraction)
# ==============================================
# Is script se aap Phase 1 directly run kar sakte ho.
# Short: Raw documents ko machine-readable text me convert karta hai.
# Concept: Ye ingestion gateway hai - agar extraction clean hoga to
# downstream chunking, embeddings, aur retrieval quality better hogi.
# Ye script ye steps karta hai:
# 1) Virtual environment activate karta hai (agar .venv present ho)
# 2) (Optional) Dependencies install/update karta hai
# 3) pdf_loader.py run karta hai
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
#   --install-deps = pip install -r requirements.txt run karega (optional)
#   $1 = input path (PDF/TXT file ya folder) [default: data/pdfs]
#   $2 = output path [default: outputs]

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

INPUT_PATH="${1:-data/pdfs}"
OUTPUT_PATH="${2:-outputs}"

# Default `data/pdfs` backward-compatible rakha gaya hai.
# Agar mixed input (PDF + TXT) ho to input path `data` pass karein.

if [[ -d ".venv" ]]; then
  echo "[INFO] Activating virtual environment (.venv)"
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[WARN] .venv not found. Running with current Python environment."
fi

if [[ "$INSTALL_DEPS" == "true" ]]; then
  echo "[INFO] Installing dependencies from requirements.txt"
  # Reproducible setup: same dependency list se environment align hota hai.
  pip install --no-cache-dir -r requirements.txt
else
  echo "[INFO] Skipping dependency install (already installed assumed)."
  echo "[INFO] If needed, run: bash run_phase1.sh --install-deps"
fi

echo "[INFO] Running Phase 1 PDF loader"
# Python script extension-based routing karta hai:
# PDF -> page-wise extract, TXT -> direct read.
python scripts/pdf_loader.py --input "$INPUT_PATH" --output "$OUTPUT_PATH"

echo "[DONE] Phase 1 extraction complete."
echo "[DONE] Output folder: $OUTPUT_PATH"
