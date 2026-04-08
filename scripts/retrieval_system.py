#!/usr/bin/env python3

"""
Phase 3 retrieval system.
This script runs semantic search and retrieval-quality evaluation on vector store.
"""

# Keep modern typing behavior consistent across Python versions.
from __future__ import annotations

# Parse CLI arguments.
import argparse
# Load/save JSON data.
import json
# Path-safe folder/file handling.
from pathlib import Path
# Type hints for flexible metadata fields.
from typing import Any

# FAISS index read/search.
import faiss
# Numpy arrays for embeddings.
import numpy as np
# Embedding model wrapper.
from sentence_transformers import SentenceTransformer


def load_vector_store(vector_store_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]], dict[str, int] | None]:
    """Load FAISS index and metadata from vector store folder."""
    # Build expected artifact paths.
    index_path = vector_store_dir / "index.faiss"
    metadata_path = vector_store_dir / "metadata.json"
    shape_path = vector_store_dir / "vectors_shape.json"

    # Stop if index file is missing.
    if not index_path.exists():
        raise FileNotFoundError(f"Missing file: {index_path}")
    # Stop if metadata file is missing.
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing file: {metadata_path}")

    # Load FAISS index from disk.
    index = faiss.read_index(str(index_path))
    # Load chunk metadata list from JSON.
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Optional shape summary.
    shape_info = None
    if shape_path.exists():
        shape_info = json.loads(shape_path.read_text(encoding="utf-8"))

    # Validate index rows and metadata rows are aligned.
    if len(metadata) != index.ntotal:
        raise ValueError(
            "Metadata count and index vector count mismatch. "
            f"metadata={len(metadata)}, index={index.ntotal}"
        )

    # Return loaded artifacts.
    return index, metadata, shape_info


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Convert query text into FAISS-compatible embedding."""
    # Encode single query into vector matrix shape (1, dim).
    vector = model.encode([query], convert_to_numpy=True)
    # Ensure FAISS-compatible dtype.
    return np.asarray(vector, dtype=np.float32)


def search_top_k(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    model: SentenceTransformer,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run similarity search and map results to chunk metadata."""
    # Convert text query to vector.
    query_vector = embed_query(model, query)
    # Search nearest neighbors from FAISS.
    distances, indices = index.search(query_vector, top_k)

    # Build final response rows.
    results: list[dict[str, Any]] = []
    # Iterate ranked search outputs.
    for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        # Skip invalid FAISS placeholders.
        if idx < 0:
            continue
        # Fetch metadata row by index id.
        row = metadata[idx]
        # Store normalized result fields.
        results.append(
            {
                "rank": rank,
                "score_l2": float(distance),
                "chunk_id": row.get("chunk_id", idx),
                "source_file": row.get("source_file", "unknown"),
                "page_number": row.get("page_number", -1),
                "text": row.get("text", ""),
            }
        )
    # Return ranked results.
    return results


def summarize_result(result: dict[str, Any], max_chars: int) -> str:
    """Format a single retrieval result for CLI output."""
    # Normalize text to single-line preview.
    text = (result["text"] or "").replace("\n", " ").strip()
    # Truncate long preview safely.
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."

    # Return printable result block.
    return (
        f"[{result['rank']}] score={result['score_l2']:.4f} "
        f"file={result['source_file']} page={result['page_number']} chunk={result['chunk_id']}\n"
        f"    {text}"
    )


def build_eval_query(text: str, max_chars: int) -> str:
    """Create a stable synthetic query from chunk text."""
    # Collapse repeated whitespace.
    clean = " ".join(text.split())
    # Return full text if already short.
    if len(clean) <= max_chars:
        return clean
    # Trim to requested max length.
    return clean[:max_chars].rstrip()


def evaluate_retrieval(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    model: SentenceTransformer,
    top_k: int,
    sample_size: int,
    query_chars: int,
) -> dict[str, Any]:
    """Estimate retrieval quality using self-retrieval probes."""
    # Ensure metadata exists.
    if not metadata:
        raise ValueError("No metadata available for evaluation")

    # Clamp sample size to available rows.
    sample_count = min(sample_size, len(metadata))
    if sample_count <= 0:
        raise ValueError("Sample size must be greater than 0")

    # Pick evenly distributed probes.
    stride = max(len(metadata) // sample_count, 1)
    probe_rows = [metadata[i] for i in range(0, len(metadata), stride)][:sample_count]

    # Metric counters.
    exact_hit = 0
    source_hit = 0
    mrr_total = 0.0

    # Evaluate each probe row.
    for row in probe_rows:
        # Expected labels for this probe.
        expected_chunk_id = row.get("chunk_id")
        expected_source = row.get("source_file")
        # Build synthetic query from probe chunk text.
        query = build_eval_query(row.get("text", ""), query_chars)
        # Skip empty probe queries.
        if not query:
            continue

        # Run search for this probe query.
        ranked = search_top_k(index=index, metadata=metadata, model=model, query=query, top_k=top_k)

        # Track first exact rank and source presence.
        exact_rank = None
        source_found = False
        for item in ranked:
            if item.get("source_file") == expected_source:
                source_found = True
            if item.get("chunk_id") == expected_chunk_id and exact_rank is None:
                exact_rank = item["rank"]

        # Update exact hit + MRR.
        if exact_rank is not None:
            exact_hit += 1
            mrr_total += 1.0 / exact_rank
        # Update source hit.
        if source_found:
            source_hit += 1

    # Final evaluated probe count.
    evaluated = len(probe_rows)
    if evaluated == 0:
        raise ValueError("No probes were evaluated")

    # Return rounded metrics report.
    return {
        "probes": evaluated,
        "top_k": top_k,
        "exact_hit_rate": round(exact_hit / evaluated, 4),
        "source_hit_rate": round(source_hit / evaluated, 4),
        "mrr": round(mrr_total / evaluated, 4),
        "notes": (
            "Self-retrieval evaluation: each probe is generated from indexed chunk text. "
            "Use this as a quick quality signal, not a final benchmark."
        ),
    }


def run_query_mode(args: argparse.Namespace) -> None:
    """Execute query mode and print top-k retrieval results."""
    # Resolve vector-store folder path.
    vector_store_dir = Path(args.vector_store)
    # Load index + metadata artifacts.
    index, metadata, shape_info = load_vector_store(vector_store_dir)
    # Load embedding model for query.
    model = SentenceTransformer(args.model)

    # Execute similarity search.
    results = search_top_k(
        index=index,
        metadata=metadata,
        model=model,
        query=args.query,
        top_k=args.top_k,
    )

    # Print output header.
    print("=" * 72)
    print("PHASE 3: RETRIEVAL RESULTS")
    print("=" * 72)
    print(f"Query: {args.query}")
    print(f"Vector store: {vector_store_dir}")
    print(f"Top-k: {args.top_k}")
    if shape_info:
        print(
            "Index stats: "
            f"vectors={shape_info.get('total_vectors', index.ntotal)}, "
            f"dim={shape_info.get('embedding_dimension', 'unknown')}"
        )
    print("-" * 72)

    # Handle empty result set.
    if not results:
        print("No results found.")
        return

    # Print each ranked result.
    for result in results:
        print(summarize_result(result=result, max_chars=args.preview_chars))

    # Optional full JSON print.
    if args.json:
        print("-" * 72)
        print(json.dumps(results, ensure_ascii=False, indent=2))


def run_eval_mode(args: argparse.Namespace) -> None:
    """Execute quick retrieval-quality evaluation."""
    # Resolve vector-store folder path.
    vector_store_dir = Path(args.vector_store)
    # Load index + metadata artifacts.
    index, metadata, shape_info = load_vector_store(vector_store_dir)
    # Load embedding model for probe queries.
    model = SentenceTransformer(args.model)

    # Compute evaluation metrics.
    metrics = evaluate_retrieval(
        index=index,
        metadata=metadata,
        model=model,
        top_k=args.top_k,
        sample_size=args.sample_size,
        query_chars=args.query_chars,
    )

    # Print report header.
    print("=" * 72)
    print("PHASE 3: RETRIEVAL QUALITY")
    print("=" * 72)
    print(f"Vector store: {vector_store_dir}")
    if shape_info:
        print(
            "Index stats: "
            f"vectors={shape_info.get('total_vectors', index.ntotal)}, "
            f"dim={shape_info.get('embedding_dimension', 'unknown')}"
        )
    # Print metrics JSON block.
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for retrieval and evaluation modes."""
    # Create top-level parser.
    parser = argparse.ArgumentParser(description="Phase 3 retrieval system for indexed chunks.")
    # Create required subcommand parser.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Query subcommand parser.
    query_parser = subparsers.add_parser("query", help="Run semantic similarity search.")
    # User question text.
    query_parser.add_argument("--query", required=True, help="Natural language query text.")
    # Vector store path.
    query_parser.add_argument("--vector-store", default="outputs/vector_store", help="Vector store folder path.")
    # Number of results to return.
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of top results to retrieve.")
    # Query embedding model.
    query_parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for query encoding.",
    )
    # Max preview length per result.
    query_parser.add_argument("--preview-chars", type=int, default=220, help="Result text preview length.")
    # Optional full JSON output.
    query_parser.add_argument("--json", action="store_true", help="Also print full JSON results.")
    # Attach query handler.
    query_parser.set_defaults(handler=run_query_mode)

    # Eval subcommand parser.
    eval_parser = subparsers.add_parser("eval", help="Evaluate retrieval quality quickly.")
    # Vector store path.
    eval_parser.add_argument("--vector-store", default="outputs/vector_store", help="Vector store folder path.")
    # Top-k used during metric computation.
    eval_parser.add_argument("--top-k", type=int, default=5, help="Top-k used for metric calculation.")
    # Number of probe rows.
    eval_parser.add_argument("--sample-size", type=int, default=50, help="Number of probes for evaluation.")
    # Max chars for synthetic probe query.
    eval_parser.add_argument("--query-chars", type=int, default=140, help="Probe query length from chunk text.")
    # Probe embedding model.
    eval_parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for probe encoding.",
    )
    # Attach eval handler.
    eval_parser.set_defaults(handler=run_eval_mode)

    # Return configured parser.
    return parser


def main() -> None:
    # Build parser with subcommands.
    parser = build_parser()
    # Parse CLI arguments.
    args = parser.parse_args()
    # Execute selected mode handler.
    args.handler(args)


# Run main only when executed directly.
if __name__ == "__main__":
    main()
