"""
graph/output_adapter.py  —  Schema Bridge

Converts Vaibhav's pipeline output (entities.json + relationships.json)
into the format expected by the teammate's graph_builder ingestion service.

WHY THIS IS NEEDED
──────────────────
Vaibhav's extractor outputs:
    entities:       {"name", "type", "chunk_id", "page_number", ...}
    relationships:  {"source", "relation", "target", "chunk_id", ...}

Teammate's graph_builder expects:
    entities:       {"entity_id", "entity_name", "entity_type", "chunk_id",
                     "page_number", "confidence"}
    relationships:  {"relationship_id", "source_entity_id", "target_entity_id",
                     "relationship_type", "chunk_id", "page_number", "confidence"}

This adapter bridges the gap without touching either codebase.

OUTPUT FILES
────────────
    output/entities_neo4j.json       ← feed to graph_builder.runner
    output/relationships_neo4j.json  ← feed to graph_builder.runner
    (original entities.json and relationships.json are kept intact)

RUN GRAPH_BUILDER AFTER THIS WITH:
    python -m graph_builder.runner \
        --entities output/entities_neo4j.json \
        --relationships output/relationships_neo4j.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  RELATIONSHIP TYPE MAPPING
#  Maps Vaibhav's extractor types → teammate's VALID_RELATIONSHIP_TYPES
# ─────────────────────────────────────────────────────────────────────────────

REL_TYPE_MAP: dict[str, str] = {
    # Direct matches (already valid)
    "USES":           "USES",
    "PART_OF":        "PART_OF",
    "DEPENDS_ON":     "DEPENDS_ON",
    "LOCATED_IN":     "LOCATED_IN",
    "PRODUCES":       "PRODUCES",          # added to schema.py

    # Mapped to nearest valid type
    "SUPPORTS":       "CONTRIBUTES_TO",    # supports ≈ contributes to
    "MANAGES":        "RELATED_TO",
    "CONTAINS":       "PART_OF",           # contains ≈ part_of (reversed sense)
    "CONNECTED_TO":   "RELATED_TO",
    "HOSTS":          "RELATED_TO",
    "CO_OCCURS_WITH": "RELATED_TO",        # context co-occurrence → generic

    # Fallback
    "RELATED_TO":     "RELATED_TO",
    "CUSTOM":         "CUSTOM",
}

# Confidence scores by relationship source
# Graph-traversal in retriever filters by min_confidence,
# so Mistral relationships are prioritised over co-occurrence.
CONFIDENCE_BY_TYPE: dict[str, float] = {
    "CO_OCCURS_WITH": 0.5,   # lower — just context proximity
}
DEFAULT_CONFIDENCE = 1.0     # Mistral semantic relationships


# ─────────────────────────────────────────────────────────────────────────────
#  ID GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def make_entity_id(name: str) -> str:
    """
    Generate a stable, unique, URL-safe entity_id from an entity name.
    Same name always produces the same ID (deterministic).

    Example: "Dell Computer Corporation" → "ent_dell_computer_corp_a1b2c3d4"
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:24]
    h    = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"ent_{slug}_{h}"


def make_relationship_id(
    source_id: str,
    rel_type:  str,
    target_id: str,
    chunk_id:  str,
    index:     int,
) -> str:
    """
    Generate a stable unique relationship_id.
    Includes index to handle multiple edges of same type between same nodes.
    """
    key = f"{source_id}|{rel_type}|{target_id}|{chunk_id}|{index}"
    h   = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return f"rel_{h}"


# ─────────────────────────────────────────────────────────────────────────────
#  ENTITY ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

def adapt_entities(entities: list) -> tuple[list, dict[str, str]]:
    """
    Transform entities from Vaibhav's format → teammate's format.

    Returns
    -------
    adapted       : list of entities in Neo4j ingestion format
    entity_id_map : dict[name_lower → entity_id]  (for relationship adapter)
    """
    adapted:       list          = []
    entity_id_map: dict[str, str] = {}

    for ent in entities:
        name = ent.get("name", "").strip()
        if not name:
            continue

        eid = make_entity_id(name)
        entity_id_map[name.lower()] = eid

        adapted.append({
            "entity_id":   eid,
            "entity_name": name,
            "entity_type": ent.get("type", "UNKNOWN"),
            "chunk_id":    ent.get("chunk_id",    ""),
            "page_number": ent.get("page_number", 0) or 0,
            "confidence":  DEFAULT_CONFIDENCE,
        })

    logger.info(f"Adapted {len(adapted)} entities")
    return adapted, entity_id_map


# ─────────────────────────────────────────────────────────────────────────────
#  RELATIONSHIP ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

def adapt_relationships(
    relationships:  list,
    entity_id_map:  dict[str, str],
    chunks:         list,
) -> list:
    """
    Transform relationships from Vaibhav's format → teammate's format.

    Parameters
    ----------
    relationships  : list from relationships.json
    entity_id_map  : name_lower → entity_id  (from adapt_entities)
    chunks         : list from chunks.json  (for page_number lookup)
    """
    # Build chunk_id → page_number map for relationships that lack page_number
    chunk_page: dict[str, int] = {
        c["chunk_id"]: c.get("page_number", 0) or 0
        for c in chunks
    }

    adapted:  list = []
    skipped = 0
    index   = 0

    for rel in relationships:
        source_name = rel.get("source", "").strip()
        target_name = rel.get("target", "").strip()
        our_type    = rel.get("relation", "RELATED_TO").upper().strip()
        chunk_id    = rel.get("chunk_id",    "")

        # ── Map relationship type ────────────────────────────────────────────
        mapped_type = REL_TYPE_MAP.get(our_type, "RELATED_TO")

        # ── Resolve entity IDs ───────────────────────────────────────────────
        source_id = entity_id_map.get(source_name.lower())
        target_id = entity_id_map.get(target_name.lower())

        if not source_id or not target_id:
            skipped += 1
            continue  # entity was pruned or not extracted

        # Skip self-loops
        if source_id == target_id:
            skipped += 1
            continue

        page_number = (
            rel.get("page_number")
            or chunk_page.get(chunk_id, 0)
            or 0
        )

        # CO_OCCURS_WITH gets lower confidence so retriever prefers Mistral rels
        confidence = CONFIDENCE_BY_TYPE.get(our_type, DEFAULT_CONFIDENCE)

        adapted.append({
            "relationship_id":   make_relationship_id(
                                     source_id, mapped_type, target_id,
                                     chunk_id, index
                                 ),
            "source_entity_id":  source_id,
            "target_entity_id":  target_id,
            "relationship_type": mapped_type,
            "chunk_id":          chunk_id,
            "page_number":       page_number,
            "confidence":        confidence,
        })
        index += 1

    logger.info(
        f"Adapted {len(adapted)} relationships "
        f"({skipped} skipped — entity not found or self-loop)"
    )
    return adapted


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ADAPTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def adapt_for_neo4j(
    entities:      list,
    relationships: list,
    chunks:        list,
    output_dir:    str = "output",
) -> dict[str, str]:
    """
    Run the full adaptation and save Neo4j-ready files.

    Returns paths to the two output files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step A: Adapt entities ────────────────────────────────────────────────
    adapted_entities, entity_id_map = adapt_entities(entities)

    entities_path = out / "entities_neo4j.json"
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(adapted_entities, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved → {entities_path}")

    # ── Step B: Adapt relationships ───────────────────────────────────────────
    adapted_relationships = adapt_relationships(relationships, entity_id_map, chunks)

    rels_path = out / "relationships_neo4j.json"
    with open(rels_path, "w", encoding="utf-8") as f:
        json.dump(adapted_relationships, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved → {rels_path}")

    return {
        "entities_neo4j":      str(entities_path),
        "relationships_neo4j": str(rels_path),
    }