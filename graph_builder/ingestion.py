"""
Core graph ingestion service.

Reads entities and relationships and MERGEs them into Neo4j.
All operations are idempotent — safe to re-run without creating
duplicate nodes or edges.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from graph_builder.connection import Neo4jConnection
from graph_builder.constraints import ensure_schema
from graph_builder.schema import (
    CYPHER_MERGE_APPEARS_IN,
    CYPHER_MERGE_CHUNK,
    CYPHER_MERGE_ENTITY,
    cypher_merge_relationship,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default batch size for MERGE operations
DEFAULT_BATCH_SIZE = 100


class GraphIngestionService:
    """MERGE-based graph ingestion service for entities and relationships.

    Usage
    -----
    >>> with Neo4jConnection() as conn:
    ...     service = GraphIngestionService(conn)
    ...     stats = service.ingest_all(entities, relationships)
    """

    def __init__(self, connection: Neo4jConnection):
        self._conn = connection

    # ── Entity ingestion ──────────────────────────────────────────────────

    def ingest_entities(
        self,
        entities: List[Dict[str, Any]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Dict[str, int]:
        """MERGE entity nodes, chunk nodes, and APPEARS_IN edges.

        Parameters
        ----------
        entities:
            List of entity dicts from ``entities.json``.
        batch_size:
            Number of entities to process per transaction.

        Returns
        -------
        dict
            Counts: ``entity_nodes``, ``chunk_nodes``, ``appears_in_edges``.
        """
        logger.info(f"Ingesting {len(entities)} entities …")

        entity_count  = 0
        chunk_ids_seen: set = set()
        appears_count = 0

        for start in range(0, len(entities), batch_size):
            batch = entities[start : start + batch_size]

            with self._conn.session() as session:
                with session.begin_transaction() as tx:
                    for ent in batch:
                        # ── MERGE Entity node ─────────────────────────
                        tx.run(
                            CYPHER_MERGE_ENTITY,
                            entity_id=ent["entity_id"],
                            entity_name=ent["entity_name"],
                            entity_type=ent["entity_type"],
                        )
                        entity_count += 1

                        # ── MERGE Chunk node ──────────────────────────
                        chunk_id = ent["chunk_id"]
                        if chunk_id not in chunk_ids_seen:
                            tx.run(
                                CYPHER_MERGE_CHUNK,
                                chunk_id=chunk_id,
                                page_number=ent["page_number"],
                            )
                            chunk_ids_seen.add(chunk_id)

                        # ── MERGE APPEARS_IN edge ─────────────────────
                        tx.run(
                            CYPHER_MERGE_APPEARS_IN,
                            entity_id=ent["entity_id"],
                            chunk_id=chunk_id,
                            confidence=ent.get("confidence", 1.0),
                        )
                        appears_count += 1

                    tx.commit()

            logger.info(
                f"  Batch {start // batch_size + 1}: "
                f"{len(batch)} entities merged"
            )

        stats = {
            "entity_nodes":     entity_count,
            "chunk_nodes":      len(chunk_ids_seen),
            "appears_in_edges": appears_count,
        }

        logger.info(
            f"  Entity ingestion done — "
            f"{stats['entity_nodes']} entities, "
            f"{stats['chunk_nodes']} chunks, "
            f"{stats['appears_in_edges']} APPEARS_IN edges"
        )
        return stats

    # ── Relationship ingestion ────────────────────────────────────────────

    def ingest_relationships(
        self,
        relationships: List[Dict[str, Any]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Dict[str, int]:
        """MERGE typed relationship edges between entity nodes.

        Parameters
        ----------
        relationships:
            List of relationship dicts from ``relationships.json``.
        batch_size:
            Number of relationships to process per transaction.

        Returns
        -------
        dict
            Counts: ``total_relationships`` and per-type breakdown.
        """
        logger.info(f"Ingesting {len(relationships)} relationships …")

        total_count = 0
        type_counts: Dict[str, int] = {}

        for start in range(0, len(relationships), batch_size):
            batch = relationships[start : start + batch_size]

            with self._conn.session() as session:
                with session.begin_transaction() as tx:
                    for rel in batch:
                        rel_type = rel["relationship_type"]

                        try:
                            cypher = cypher_merge_relationship(rel_type)
                        except ValueError as exc:
                            logger.warning(
                                f"  ⚠  Skipping relationship "
                                f"{rel.get('relationship_id', '?')}: {exc}"
                            )
                            continue

                        tx.run(
                            cypher,
                            source_entity_id=rel["source_entity_id"],
                            target_entity_id=rel["target_entity_id"],
                            relationship_id=rel["relationship_id"],
                            confidence=rel.get("confidence", 1.0),
                            chunk_id=rel.get("chunk_id", ""),
                            page_number=rel.get("page_number", 0),
                        )

                        total_count += 1
                        type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

                    tx.commit()

            logger.info(
                f"  Batch {start // batch_size + 1}: "
                f"{len(batch)} relationships merged"
            )

        stats = {
            "total_relationships": total_count,
            **{f"type_{k}": v for k, v in type_counts.items()},
        }

        logger.info(
            f"  Relationship ingestion done — "
            f"{total_count} relationships across "
            f"{len(type_counts)} types"
        )
        return stats

    # ── Full pipeline ─────────────────────────────────────────────────────

    def ingest_all(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Dict[str, Any]:
        """Run the complete ingestion pipeline.

        1. Create constraints & indexes
        2. MERGE entity and chunk nodes
        3. MERGE relationship edges

        Parameters
        ----------
        entities:
            List of entity dicts.
        relationships:
            List of relationship dicts.
        batch_size:
            Batch size for MERGE operations.

        Returns
        -------
        dict
            Combined statistics from all stages.
        """
        logger.info(f"\n{'═' * 50}")
        logger.info(f"  NEO4J GRAPH INGESTION")
        logger.info(f"{'═' * 50}")

        # Step 1: Schema
        with self._conn.session() as session:
            ensure_schema(session)

        # Step 2: Entities + Chunks + APPEARS_IN
        entity_stats = self.ingest_entities(entities, batch_size)

        # Step 3: Relationships
        rel_stats = self.ingest_relationships(relationships, batch_size)

        combined = {**entity_stats, **rel_stats}

        logger.info(f"\n{'═' * 50}")
        logger.info(f"  INGESTION COMPLETE")
        logger.info(f"{'═' * 50}")
        for key, value in combined.items():
            logger.info(f"  {key:25s} : {value}")
        logger.info(f"{'═' * 50}")

        return combined
