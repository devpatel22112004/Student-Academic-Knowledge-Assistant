#!/usr/bin/env bash

# Stop script on first command failure.
set -e
# Fail pipeline if any command inside a pipe fails.
set -o pipefail

# ==========================================================
# Unified Runner: Phase 1 / Phase 2 / Phase 3
# ==========================================================
# Default behavior:
#   Runs Phase 1 + Phase 2.
# Simple modes:
#   phase1, phase2, phase3/query, eval, ask

# Default runtime settings.
RUN_EVAL=false
QUERY_TEXT=""
QUERY_TOP_K="5"
EVAL_TOP_K="5"
EVAL_SAMPLE_SIZE="20"
PHASE3_ONLY=false
RUN_PHASE1=true
RUN_PHASE2=true

# ---------------------------------------------
# Simple command mode (easy to remember/use)
# ---------------------------------------------
# Supported simple starts:
#   bash run_unified.sh build
#   bash run_unified.sh phase1
#   bash run_unified.sh phase2
#   bash run_unified.sh phase3
#   bash run_unified.sh query "Who is CSK captain?"
#   bash run_unified.sh eval
#   bash run_unified.sh ask
#   bash run_unified.sh query:"Who is CSK captain?"
if [[ "${1:-}" == "build" ]]; then
    # Explicitly request full build flow (Phase 1 + 2).
    shift
elif [[ "${1:-}" == "phase1" ]]; then
    # Run only extraction step.
    RUN_PHASE1=true
    RUN_PHASE2=false
    PHASE3_ONLY=false
    shift
elif [[ "${1:-}" == "phase2" ]]; then
    # Run only indexing step.
    RUN_PHASE1=false
    RUN_PHASE2=true
    PHASE3_ONLY=false
    shift
elif [[ "${1:-}" == "phase3" ]]; then
    # Run only query retrieval in interactive style.
    PHASE3_ONLY=true
    RUN_PHASE1=false
    RUN_PHASE2=false
    shift
elif [[ "${1:-}" == "query" ]]; then
    # Query mode: skip build and run only Phase 3 query.
    PHASE3_ONLY=true
    RUN_PHASE1=false
    RUN_PHASE2=false
    shift

    # Accept first non-flag token as query text.
    # Extra options like --query-top-k are still parsed later.
    if [[ "$#" -gt 0 && "${1:-}" != --* ]]; then
        QUERY_TEXT="$1"
        shift
    fi
elif [[ "${1:-}" == "eval" ]]; then
    # Eval mode: skip build and run only Phase 3 evaluation.
    PHASE3_ONLY=true
    RUN_PHASE1=false
    RUN_PHASE2=false
    RUN_EVAL=true
    shift
elif [[ "${1:-}" == query:* ]]; then
    # One-token query shortcut: query:Your question here
    PHASE3_ONLY=true
    RUN_PHASE1=false
    RUN_PHASE2=false
    QUERY_TEXT="${1#query:}"
    shift
elif [[ "${1:-}" == "ask" ]]; then
    # Prompt-driven mode selection for non-flag users.
    echo "Choose run mode:"
    echo "  1) phase1"
    echo "  2) phase2"
    echo "  3) phase3-query"
    echo "  4) phase3-eval"
    echo "  5) full (phase1+phase2)"
    echo "  6) full + phase3-query"
    read -rp "Enter choice (1-6): " ASK_CHOICE

    case "$ASK_CHOICE" in
        1)
            RUN_PHASE1=true
            RUN_PHASE2=false
            ;;
        2)
            RUN_PHASE1=false
            RUN_PHASE2=true
            ;;
        3)
            PHASE3_ONLY=true
            RUN_PHASE1=false
            RUN_PHASE2=false
            ;;
        4)
            PHASE3_ONLY=true
            RUN_PHASE1=false
            RUN_PHASE2=false
            RUN_EVAL=true
            ;;
        5)
            RUN_PHASE1=true
            RUN_PHASE2=true
            ;;
        6)
            RUN_PHASE1=true
            RUN_PHASE2=true
            ;;
        *)
            echo "[ERROR] Invalid choice. Use 1-6."
            exit 1
            ;;
    esac

    if [[ "$ASK_CHOICE" == "3" || "$ASK_CHOICE" == "6" ]]; then
        read -rp "Enter query: " QUERY_TEXT
    fi
    shift
fi

# Parse optional flags first.
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --query)
            # Phase 3 query text.
            QUERY_TEXT="$2"
            shift 2
            ;;
        --query-top-k)
            # Top-k for query mode.
            QUERY_TOP_K="$2"
            shift 2
            ;;
        --run-eval)
            # Enable Phase 3 eval mode.
            RUN_EVAL=true
            shift
            ;;
        --eval-top-k)
            # Top-k for eval mode.
            EVAL_TOP_K="$2"
            shift 2
            ;;
        --eval-sample-size)
            # Probe count for eval mode.
            EVAL_SAMPLE_SIZE="$2"
            shift 2
            ;;
        --phase3-only)
            # Fast mode: skip Phase 1 and 2, run only Phase 3.
            PHASE3_ONLY=true
            RUN_PHASE1=false
            RUN_PHASE2=false
            shift
            ;;
        *)
            # Remaining args are positional phase1/phase2 args.
            break
            ;;
    esac
done

# Positional paths and chunk settings.
INPUT_FOLDER="${1:-data}"
EXTRACTED_FOLDER="${2:-outputs/extracted}"
VECTOR_STORE_FOLDER="${3:-outputs/vector_store}"
CHUNK_SIZE="${4:-600}"
CHUNK_OVERLAP="${5:-100}"
MODEL_NAME="${6:-sentence-transformers/all-MiniLM-L6-v2}"

# Activate local virtual environment when available.
if [[ -d ".venv" ]]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "[UNIFIED] Running pipeline..."
echo "[UNIFIED] Input folder: $INPUT_FOLDER"
echo "[UNIFIED] Extracted output: $EXTRACTED_FOLDER"
echo "[UNIFIED] Vector store: $VECTOR_STORE_FOLDER"
if [[ "$PHASE3_ONLY" == "true" ]]; then
    echo "[UNIFIED] Mode: phase3-only (fast mode)"
fi
echo ""

# In query mode, allow empty CLI query and ask interactively.
if [[ "$PHASE3_ONLY" == "true" && -z "$QUERY_TEXT" && "$RUN_EVAL" != "true" ]]; then
    read -rp "Enter your query: " QUERY_TEXT
fi

if [[ "$RUN_PHASE1" == "true" ]]; then
    # -------------------------
    # STEP 1: Phase 1 extraction
    # -------------------------
    echo "[STEP 1] Phase 1: Document extraction"
    python scripts/pdf_loader.py \
        --input "$INPUT_FOLDER" \
        --output "$EXTRACTED_FOLDER"
fi

if [[ "$RUN_PHASE2" == "true" ]]; then
    # ----------------------
    # STEP 2: Phase 2 indexing
    # ----------------------
    echo "[STEP 2] Phase 2: Chunking + Embedding + Indexing"
    python scripts/document_pipeline.py \
        --input "$EXTRACTED_FOLDER" \
        --output "$VECTOR_STORE_FOLDER" \
        --chunk-size "$CHUNK_SIZE" \
        --chunk-overlap "$CHUNK_OVERLAP" \
        --model "$MODEL_NAME"
fi

if [[ "$RUN_PHASE1" == "false" && "$RUN_PHASE2" == "false" && ( -n "$QUERY_TEXT" || "$RUN_EVAL" == "true" ) ]]; then
    # In fast mode, reuse existing index artifacts from disk.
    if [[ ! -f "$VECTOR_STORE_FOLDER/index.faiss" || ! -f "$VECTOR_STORE_FOLDER/metadata.json" ]]; then
        echo "[ERROR] phase3-only mode requires existing index files in: $VECTOR_STORE_FOLDER"
        echo "[ERROR] Missing index.faiss or metadata.json. Run full pipeline first."
        exit 1
    fi
fi

# -----------------------------------------
# STEP 3 (optional): Phase 3 query retrieval
# -----------------------------------------
if [[ -n "$QUERY_TEXT" ]]; then
    echo "[STEP 3A] Phase 3: Query retrieval"
    python scripts/retrieval_system.py query \
        --query "$QUERY_TEXT" \
        --vector-store "$VECTOR_STORE_FOLDER" \
        --top-k "$QUERY_TOP_K" \
        --model "$MODEL_NAME"
fi

# ------------------------------------------
# STEP 3 (optional): Phase 3 eval retrieval
# ------------------------------------------
if [[ "$RUN_EVAL" == "true" ]]; then
    echo "[STEP 3B] Phase 3: Retrieval quality evaluation"
    python scripts/retrieval_system.py eval \
        --vector-store "$VECTOR_STORE_FOLDER" \
        --top-k "$EVAL_TOP_K" \
        --sample-size "$EVAL_SAMPLE_SIZE" \
        --model "$MODEL_NAME"
fi

echo ""
echo "=========================================================="
echo "[DONE] Unified run completed successfully"
echo "=========================================================="
echo ""
# echo "Usage examples:"
# echo "  bash run_unified.sh"
# echo "  bash run_unified.sh build"
# echo "  bash run_unified.sh phase1"
# echo "  bash run_unified.sh phase2"
# echo "  bash run_unified.sh phase3"
# echo "  bash run_unified.sh query \"Who is CSK captain?\""
# echo "  bash run_unified.sh query:\"Who is CSK captain?\""
# echo "  bash run_unified.sh eval"
# echo "  bash run_unified.sh ask"
# echo "  bash run_unified.sh data outputs/extracted outputs/vector_store"
# echo "  bash run_unified.sh --query \"Who is CSK captain?\""
# echo "  bash run_unified.sh --run-eval --eval-sample-size 20"
# echo "  bash run_unified.sh --query \"MI titles\" --query-top-k 3 --run-eval"
# echo "  bash run_unified.sh --phase3-only --query \"Who is CSK captain?\""
