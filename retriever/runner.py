"""
CLI runner for the GraphRAG agentic retrieval pipeline.

Usage:
    python -m retriever.runner --query "How to replace the battery?"
    python -m retriever.runner --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from retriever.pipeline import RetrievalPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_query(pipeline: RetrievalPipeline, query: str) -> None:
    """Execute a single query and print structured results."""
    result = pipeline.retrieve(query)

    print("\n" + "=" * 60)
    print("  STRUCTURED OUTPUT PACKAGE")
    print("=" * 60)

    # Convert RetrievalResult to the requested JSON format dictionary
    output_package = {
        "answer_context": result.answer_context,
        "evidence_chunks": result.evidence_chunks,
        "graph_paths": result.graph_paths,
        "source_entities": result.source_entities,
        "retrieval_metadata": result.retrieval_metadata,
    }

    # Print a beautiful JSON representation of the output
    print(json.dumps(output_package, indent=2))
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG Agentic Retrieval Pipeline"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="A single query to retrieve context for.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (loop prompting for queries).",
    )
    parser.add_argument(
        "--chunks",
        default="output/chunks.json",
        help="Path to chunks.json (default: output/chunks.json)",
    )
    parser.add_argument(
        "--entities",
        default="output/entities.json",
        help="Path to entities.json (default: output/entities.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of results after re-ranking (default: 15)",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=4000,
        help="Token budget for final context (default: 4000)",
    )
    args = parser.parse_args()

    # Create pipeline
    pipeline = RetrievalPipeline(
        chunks_path=args.chunks,
        entities_path=args.entities,
        rerank_top_k=args.top_k,
        token_budget=args.token_budget,
    )

    try:
        if args.query:
            run_query(pipeline, args.query)

        elif args.interactive:
            print("\n" + "=" * 40)
            print("  GraphRAG Agentic Retrieval - REPL")
            print("  Type 'quit' or 'exit' to stop")
            print("=" * 40 + "\n")

            while True:
                try:
                    query = input("Query > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break

                if not query:
                    continue
                if query.lower() in ("quit", "exit", "q"):
                    print("Exiting.")
                    break

                run_query(pipeline, query)
                print()

        else:
            # Default demo query
            demo_query = "How to replace the battery?"
            print(f"\nRunning default query: \"{demo_query}\"")
            print("Use --query or --interactive for custom queries.\n")
            run_query(pipeline, demo_query)

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
