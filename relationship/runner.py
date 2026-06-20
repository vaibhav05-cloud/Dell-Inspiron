"""
CLI runner for relationship extraction.

Usage:
    python -m relationship.runner
    python -m relationship.runner --chunks output/chunks.json \
                                  --entities output/entities.json \
                                  --output output/relationships.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from relationship.extractor import RelationshipExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_json(path: str, label: str) -> Any:
    """Load a JSON file, exiting on failure."""
    from typing import Any

    p = Path(path)
    if not p.exists():
        logger.error(f"{label} not found: {p}")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {label} from {p}")
    return data


def save_relationships(result, path: str) -> None:
    """Serialize RelationshipExtractionResult to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "total_chunks_processed":      result.total_chunks_processed,
        "total_relationships_extracted": result.total_relationships_extracted,
        "relationships_by_type":        result.relationships_by_type,
        "relationships": [
            rel.model_dump(mode="json") for rel in result.relationships
        ],
    }

    with open(p, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    logger.info(f"Relationships saved to {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract relationships from chunks + entities"
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
        "--output",
        default="output/relationships.json",
        help="Path to write relationships.json "
             "(default: output/relationships.json)",
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────
    chunks = load_json(args.chunks, "chunks")

    entities_data = load_json(args.entities, "entities")
    # entities.json has a top-level wrapper; extract the list
    if isinstance(entities_data, dict) and "entities" in entities_data:
        entities = entities_data["entities"]
    else:
        entities = entities_data

    logger.info(f"  {len(chunks)} chunks, {len(entities)} entities")

    # ── Extract ───────────────────────────────────────────────────────────
    extractor = RelationshipExtractor()
    result = extractor.extract_all(chunks, entities)

    # ── Save ──────────────────────────────────────────────────────────────
    save_relationships(result, args.output)

    print(f"\n[OK]  Relationship extraction complete!")
    print(f"   Chunks        : {result.total_chunks_processed}")
    print(f"   Relationships : {result.total_relationships_extracted}")
    print(f"   Output        : {args.output}")


if __name__ == "__main__":
    main()
