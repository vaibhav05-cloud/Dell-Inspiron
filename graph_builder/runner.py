"""
CLI runner for Neo4j graph construction.

Usage:
    python -m graph_builder.runner
    python -m graph_builder.runner --entities output/entities.json \
                                   --relationships output/relationships.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from graph_builder.connection import Neo4jConnection
from graph_builder.ingestion import GraphIngestionService
from graph_builder.validator import validate_graph, validate_inputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_json(path: str, label: str):
    """Load a JSON file, exiting on failure."""
    p = Path(path)
    if not p.exists():
        logger.error(f"{label} not found: {p}")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {label} from {p}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Build Neo4j graph from entities + relationships"
    )
    parser.add_argument(
        "--entities",
        default="output/entities.json",
        help="Path to entities.json (default: output/entities.json)",
    )
    parser.add_argument(
        "--relationships",
        default="output/relationships.json",
        help="Path to relationships.json "
             "(default: output/relationships.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for MERGE operations (default: 100)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip pre-ingestion validation",
    )
    args = parser.parse_args()

    # ── Load inputs ───────────────────────────────────────────────────────
    entities_data = load_json(args.entities, "entities")
    rels_data     = load_json(args.relationships, "relationships")

    # Handle wrapper format (top-level dict with "entities" / "relationships" key)
    if isinstance(entities_data, dict) and "entities" in entities_data:
        entities = entities_data["entities"]
    else:
        entities = entities_data

    if isinstance(rels_data, dict) and "relationships" in rels_data:
        relationships = rels_data["relationships"]
    else:
        relationships = rels_data

    logger.info(
        f"  {len(entities)} entities, "
        f"{len(relationships)} relationships"
    )

    # ── Pre-ingestion validation ──────────────────────────────────────────
    if not args.skip_validation:
        logger.info("\n" + "─" * 50)
        logger.info("  PRE-INGESTION VALIDATION")
        logger.info("─" * 50)

        report = validate_inputs(entities, relationships)

        if not report["valid"]:
            logger.error(
                "Validation failed — aborting ingestion. "
                "Use --skip-validation to force."
            )
            sys.exit(1)

    # ── Connect & ingest ──────────────────────────────────────────────────
    with Neo4jConnection() as conn:
        service = GraphIngestionService(conn)
        stats = service.ingest_all(
            entities,
            relationships,
            batch_size=args.batch_size,
        )

        # ── Post-ingestion health check ───────────────────────────────────
        logger.info("\n" + "─" * 50)
        logger.info("  POST-INGESTION HEALTH CHECK")
        logger.info("─" * 50)

        with conn.session() as session:
            health = validate_graph(session)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n[OK]  Neo4j graph construction complete!")
    print(f"   Entity nodes      : {stats['entity_nodes']}")
    print(f"   Chunk nodes       : {stats['chunk_nodes']}")
    print(f"   APPEARS_IN edges  : {stats['appears_in_edges']}")
    print(f"   Relationship edges: {stats['total_relationships']}")

    if health.get("orphan_count", 0) > 0:
        print(f"   ⚠ Orphaned entities: {health['orphan_count']}")


if __name__ == "__main__":
    main()
