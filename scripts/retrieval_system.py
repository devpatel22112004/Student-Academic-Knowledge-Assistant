#!/usr/bin/env python3

"""
Phase 3 retrieval system: semantic search on FAISS index.

Features:
- Similarity search for user query
- Top-k relevant chunk retrieval
- Lightweight retrieval quality evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_vector_store(vector_store_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]], dict[str, int] | None]:
    """Load FAISS index and metadata from vector store folder."""
    index_path = vector_store_dir / "index.faiss"
    metadata_path = vector_store_dir / "metadata.json"
    shape_path = vector_store_dir / "vectors_shape.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing file: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing file: {metadata_path}")

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    shape_info = None
    if shape_path.exists():
        shape_info = json.loads(shape_path.read_text(encoding="utf-8"))

    if len(metadata) != index.ntotal:
        raise ValueError(
            "Metadata count and index vector count mismatch. "
            f"metadata={len(metadata)}, index={index.ntotal}"
        )

    return index, metadata, shape_info


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Convert query text into FAISS-compatible embedding."""
    vector = model.encode([query], convert_to_numpy=True)
    return np.asarray(vector, dtype=np.float32)


def search_top_k(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    model: SentenceTransformer,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run similarity search and map results to chunk metadata."""
    query_vector = embed_query(model, query)
    distances, indices = index.search(query_vector, top_k)

    results: list[dict[str, Any]] = []
    for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx < 0:
            continue
        row = metadata[idx]
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
    return results


def summarize_result(result: dict[str, Any], max_chars: int) -> str:
    """Format a single retrieval result for CLI output."""
    text = (result["text"] or "").replace("\n", " ").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."

    return (
        f"[{result['rank']}] score={result['score_l2']:.4f} "
        f"file={result['source_file']} page={result['page_number']} chunk={result['chunk_id']}\n"
        f"    {text}"
    )


def build_eval_query(text: str, max_chars: int) -> str:
    """Create a stable synthetic query from chunk text."""
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
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
    if not metadata:
        raise ValueError("No metadata available for evaluation")

    sample_count = min(sample_size, len(metadata))
    if sample_count <= 0:
        raise ValueError("Sample size must be greater than 0")

    stride = max(len(metadata) // sample_count, 1)
    probe_rows = [metadata[i] for i in range(0, len(metadata), stride)][:sample_count]

    exact_hit = 0
    source_hit = 0
    mrr_total = 0.0

    for row in probe_rows:
        expected_chunk_id = row.get("chunk_id")
        expected_source = row.get("source_file")
        query = build_eval_query(row.get("text", ""), query_chars)
        if not query:
            continue

        ranked = search_top_k(index=index, metadata=metadata, model=model, query=query, top_k=top_k)

        exact_rank = None
        source_found = False
        for item in ranked:
            if item.get("source_file") == expected_source:
                source_found = True
            if item.get("chunk_id") == expected_chunk_id and exact_rank is None:
                exact_rank = item["rank"]

        if exact_rank is not None:
            exact_hit += 1
            mrr_total += 1.0 / exact_rank
        if source_found:
            source_hit += 1

    evaluated = len(probe_rows)
    if evaluated == 0:
        raise ValueError("No probes were evaluated")

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
    vector_store_dir = Path(args.vector_store)
    index, metadata, shape_info = load_vector_store(vector_store_dir)
    model = SentenceTransformer(args.model)

    results = search_top_k(
        index=index,
        metadata=metadata,
        model=model,
        query=args.query,
        top_k=args.top_k,
    )

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

    if not results:
        print("No results found.")
        return

    for result in results:
        print(summarize_result(result=result, max_chars=args.preview_chars))

    if args.json:
        print("-" * 72)
        print(json.dumps(results, ensure_ascii=False, indent=2))


def run_eval_mode(args: argparse.Namespace) -> None:
    """Execute quick retrieval-quality evaluation."""
    vector_store_dir = Path(args.vector_store)
    index, metadata, shape_info = load_vector_store(vector_store_dir)
    model = SentenceTransformer(args.model)

    metrics = evaluate_retrieval(
        index=index,
        metadata=metadata,
        model=model,
        top_k=args.top_k,
        sample_size=args.sample_size,
        query_chars=args.query_chars,
    )

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
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for retrieval and evaluation modes."""
    parser = argparse.ArgumentParser(description="Phase 3 retrieval system for indexed chunks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Run semantic similarity search.")
    query_parser.add_argument("--query", required=True, help="Natural language query text.")
    query_parser.add_argument("--vector-store", default="outputs/vector_store", help="Vector store folder path.")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of top results to retrieve.")
    query_parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for query encoding.",
    )
    query_parser.add_argument("--preview-chars", type=int, default=220, help="Result text preview length.")
    query_parser.add_argument("--json", action="store_true", help="Also print full JSON results.")
    query_parser.set_defaults(handler=run_query_mode)

    eval_parser = subparsers.add_parser("eval", help="Evaluate retrieval quality quickly.")
    eval_parser.add_argument("--vector-store", default="outputs/vector_store", help="Vector store folder path.")
    eval_parser.add_argument("--top-k", type=int, default=5, help="Top-k used for metric calculation.")
    eval_parser.add_argument("--sample-size", type=int, default=50, help="Number of probes for evaluation.")
    eval_parser.add_argument("--query-chars", type=int, default=140, help="Probe query length from chunk text.")
    eval_parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for probe encoding.",
    )
    eval_parser.set_defaults(handler=run_eval_mode)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
