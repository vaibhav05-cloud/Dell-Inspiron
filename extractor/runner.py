"""
CLI runner for entity extraction.

Usage:
    python -m extractor.runner
    python -m extractor.runner --input output/chunks.json --output output/entities.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from extractor.extractor import EntityExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_chunks(path: str) -> list:
    """Load chunks from a JSON file."""
    p = Path(path)
    if not p.exists():
        logger.error(f"Input file not found: {p}")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {p}")
    return chunks


def save_entities(result, path: str) -> None:
    """Serialize ExtractionResult to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Convert to a JSON-serializable dict
    output = {
        "total_chunks_processed":  result.total_chunks_processed,
        "total_entities_extracted": result.total_entities_extracted,
        "entities_by_type":        result.entities_by_type,
        "entities": [
            ent.model_dump(mode="json") for ent in result.entities
        ],
    }

    with open(p, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    logger.info(f"Entities saved to {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract entities from chunks.json"
    )
    parser.add_argument(
        "--input",
        default="output/chunks.json",
        help="Path to chunks.json (default: output/chunks.json)",
    )
    parser.add_argument(
        "--output",
        default="output/entities.json",
        help="Path to write entities.json (default: output/entities.json)",
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────
    chunks = load_chunks(args.input)

    # ── Extract ───────────────────────────────────────────────────────────
    extractor = EntityExtractor()
    result = extractor.extract_all(chunks)

    # ── Save ──────────────────────────────────────────────────────────────
    save_entities(result, args.output)

    print(f"\n[OK]  Entity extraction complete!")
    print(f"   Chunks : {result.total_chunks_processed}")
    print(f"   Entities: {result.total_entities_extracted}")
    print(f"   Output : {args.output}")


if __name__ == "__main__":
    main()
