#!/usr/bin/env bash
set -e

# =====================================================
# Codespace Space Cleanup Helper
# =====================================================
# Purpose:
# - Low disk warning aaye to quickly safe cleanup karna
# - Working environment break kiye bina free space badhana
#
# Usage:
#   bash cleanup_codespace_space.sh
#
# What it cleans:
# 1) pip cache  (~GBs ho sakta hai)
# 2) huggingface cache (~GBs ho sakta hai)
# 3) python __pycache__ folders in project
#
# Note:
# - Ye .venv delete nahi karta (safe default)
# - Agar still low space ho, manual heavy cleanup ke liye README me notes follow karein

WORKSPACE_DIR="/workspaces/Student-Academic-Knowledge-Assistant"

echo "[INFO] Disk before cleanup"
df -h /workspaces || true

echo "[INFO] Removing pip and huggingface cache"
rm -rf /home/codespace/.cache/pip /home/codespace/.cache/huggingface

echo "[INFO] Removing project __pycache__ folders"
find "$WORKSPACE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

echo "[INFO] Disk after cleanup"
df -h /workspaces || true

echo "[DONE] Safe cleanup completed."
echo "[TIP] If still low space, recreate a smaller .venv (CPU-only torch) as next step."
