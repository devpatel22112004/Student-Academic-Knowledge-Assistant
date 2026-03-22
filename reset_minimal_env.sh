#!/usr/bin/env bash
set -e

# =====================================================
# Reset Minimal Environment (Phase 1 + Phase 2)
# =====================================================
# Purpose:
# - Heavy / broken virtual env ko clean reset karna
# - Sirf required minimal dependencies install karna
# - Future low-disk issues reduce karna
#
# Usage:
#   bash reset_minimal_env.sh
#
# What this does:
# 1) Existing .venv remove karta hai
# 2) Fresh .venv banata hai
# 3) pip upgrade karta hai
# 4) requirements.txt se minimal deps install karta hai (no cache)

PROJECT_ROOT="/workspaces/Student-Academic-Knowledge-Assistant"
cd "$PROJECT_ROOT"

echo "[INFO] Removing old virtual environment (.venv)"
rm -rf .venv

echo "[INFO] Creating fresh virtual environment"
python3 -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[INFO] Upgrading pip"
python -m pip install --upgrade pip

echo "[INFO] Installing minimal dependencies (no cache)"
pip install --no-cache-dir -r requirements.txt

echo "[INFO] Final disk usage"
df -h /workspaces || true

echo "[DONE] Minimal environment is ready."
