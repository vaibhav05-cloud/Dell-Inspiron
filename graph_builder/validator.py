"""
Pre-ingestion validation and post-ingestion health checks.

Catches data integrity issues before they reach Neo4j, and
verifies the graph state after ingestion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from graph_builder.schema import (
    CYPHER_COUNT_APPEARS,
    CYPHER_COUNT_CHUNKS,
    CYPHER_COUNT_ENTITIES,
    CYPHER_COUNT_RELATIONSHIPS_BY_TYPE,
    CYPHER_ORPHAN_ENTITIES,
    VALID_RELATIONSHIP_TYPES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  PRE-INGESTION VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_inputs(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate entities and relationships before ingestion.

    Checks
    ------
    - All entity_ids are unique
    - All relationship source/target entity_ids exist in the entity set
    - No self-referencing relationships
    - All relationship types are valid

    Parameters
    ----------
    entities:
        List of entity dicts.
    relationships:
        List of relationship dicts.

    Returns
    -------
    dict
        Validation report with ``valid`` bool and any ``errors``/``warnings``.
    """
    errors: List[str]   = []
    warnings: List[str] = []

    # ── Entity ID uniqueness ──────────────────────────────────────────────
    entity_ids: set = set()
    duplicate_ids: List[str] = []

    for ent in entities:
        eid = ent.get("entity_id", "")
        if eid in entity_ids:
            duplicate_ids.append(eid)
        entity_ids.add(eid)

    if duplicate_ids:
        errors.append(
            f"Duplicate entity_ids found: {duplicate_ids[:10]}"
            + (f" … and {len(duplicate_ids) - 10} more"
               if len(duplicate_ids) > 10 else "")
        )

    # ── Relationship validation ───────────────────────────────────────────
    missing_source: List[str] = []
    missing_target: List[str] = []
    self_refs: List[str]      = []
    invalid_types: List[str]  = []

    for rel in relationships:
        rid  = rel.get("relationship_id", "?")
        src  = rel.get("source_entity_id", "")
        tgt  = rel.get("target_entity_id", "")
        rtype = rel.get("relationship_type", "")

        if src not in entity_ids:
            missing_source.append(f"{rid} → source={src}")

        if tgt not in entity_ids:
            missing_target.append(f"{rid} → target={tgt}")

        if src == tgt and src:
            self_refs.append(f"{rid}: {src}")

        if rtype not in VALID_RELATIONSHIP_TYPES:
            invalid_types.append(f"{rid}: {rtype}")

    if missing_source:
        errors.append(
            f"Relationships with missing source entity: "
            f"{missing_source[:5]}"
        )
    if missing_target:
        errors.append(
            f"Relationships with missing target entity: "
            f"{missing_target[:5]}"
        )
    if self_refs:
        warnings.append(
            f"Self-referencing relationships: {self_refs[:5]}"
        )
    if invalid_types:
        warnings.append(
            f"Relationships with unknown type: {invalid_types[:5]}"
        )

    # ── Report ────────────────────────────────────────────────────────────
    is_valid = len(errors) == 0

    report = {
        "valid":         is_valid,
        "entity_count":  len(entities),
        "relationship_count": len(relationships),
        "unique_entity_ids":  len(entity_ids),
        "errors":        errors,
        "warnings":      warnings,
    }

    if is_valid:
        logger.info("  Pre-ingestion validation passed ✓")
    else:
        logger.error("  Pre-ingestion validation FAILED ✗")
        for err in errors:
            logger.error(f"    ERROR: {err}")

    for warn in warnings:
        logger.warning(f"    WARNING: {warn}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
#  POST-INGESTION HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

def validate_graph(session) -> Dict[str, Any]:
    """Run health checks against the Neo4j graph after ingestion.

    Checks
    ------
    - Entity node count
    - Chunk node count
    - APPEARS_IN edge count
    - Relationship edge counts by type
    - Orphaned entities (no APPEARS_IN edge)

    Parameters
    ----------
    session:
        An open ``neo4j.Session``.

    Returns
    -------
    dict
        Health-check report.
    """
    logger.info("Running post-ingestion graph health check …")

    report: Dict[str, Any] = {}

    # Node counts
    result = session.run(CYPHER_COUNT_ENTITIES).single()
    report["entity_nodes"] = result["count"] if result else 0

    result = session.run(CYPHER_COUNT_CHUNKS).single()
    report["chunk_nodes"] = result["count"] if result else 0

    # APPEARS_IN count
    result = session.run(CYPHER_COUNT_APPEARS).single()
    report["appears_in_edges"] = result["count"] if result else 0

    # Relationship counts by type
    rel_counts = {}
    results = session.run(CYPHER_COUNT_RELATIONSHIPS_BY_TYPE)
    for record in results:
        rel_counts[record["rel_type"]] = record["count"]
    report["relationships_by_type"] = rel_counts
    report["total_relationship_edges"] = sum(rel_counts.values())

    # Orphaned entities
    orphans = []
    results = session.run(CYPHER_ORPHAN_ENTITIES)
    for record in results:
        orphans.append({
            "entity_id":   record["entity_id"],
            "entity_name": record["entity_name"],
        })
    report["orphaned_entities"] = orphans
    report["orphan_count"] = len(orphans)

    # Summary
    logger.info(f"  Entity nodes      : {report['entity_nodes']}")
    logger.info(f"  Chunk nodes       : {report['chunk_nodes']}")
    logger.info(f"  APPEARS_IN edges  : {report['appears_in_edges']}")
    logger.info(f"  Relationship edges: {report['total_relationship_edges']}")

    if rel_counts:
        for rtype, count in sorted(rel_counts.items()):
            logger.info(f"    {rtype:20s} : {count}")

    if orphans:
        logger.warning(
            f"  ⚠ {len(orphans)} orphaned entities "
            f"(no APPEARS_IN edge)"
        )
    else:
        logger.info("  No orphaned entities ✓")

    logger.info("Health check complete ✓")

    return report
