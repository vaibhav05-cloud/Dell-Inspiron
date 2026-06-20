"""
CLI runner for the Answer Synthesis pipeline.

Runs the full retrieval pipeline (7 agents), then feeds the result
into the Answer Synthesis Chain to produce a grounded, attributed answer.

Usage:
    python -m synthesizer.runner --query "What is the battery warranty?"
    python -m synthesizer.runner --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from retriever.pipeline import RetrievalPipeline
from synthesizer.synthesis_chain import AnswerSynthesisChain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_query(
    pipeline: RetrievalPipeline,
    synth_chain: AnswerSynthesisChain,
    query: str,
) -> None:
    """Execute retrieval + synthesis for a single query and print results."""
    # Phase 1: Retrieval
    retrieval_result = pipeline.retrieve(query)

    # Phase 2: Answer Synthesis
    synthesis_result = synth_chain.synthesize(query, retrieval_result)

    # Print final output
    print("\n" + "=" * 60)
    print("  SYNTHESIZED ANSWER")
    print("=" * 60)
    print(json.dumps(synthesis_result.model_dump(), indent=2))
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG Answer Synthesis Pipeline"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="A single query to answer.",
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
        help="Token budget for retrieval context (default: 4000)",
    )
    args = parser.parse_args()

    # Initialize pipelines
    pipeline = RetrievalPipeline(
        chunks_path=args.chunks,
        entities_path=args.entities,
        rerank_top_k=args.top_k,
        token_budget=args.token_budget,
    )
    synth_chain = AnswerSynthesisChain()

    try:
        if args.query:
            run_query(pipeline, synth_chain, args.query)

        elif args.interactive:
            print("\n" + "=" * 40)
            print("  GraphRAG Answer Synthesis - REPL")
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

                run_query(pipeline, synth_chain, query)
                print()

        else:
            # Default demo query
            demo_query = "What is the battery warranty?"
            print(f'\nRunning default query: "{demo_query}"')
            print("Use --query or --interactive for custom queries.\n")
            run_query(pipeline, synth_chain, demo_query)

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
