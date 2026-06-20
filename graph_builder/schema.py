"""
Neo4j graph schema constants and Cypher query templates.

Centralises all node labels, property keys, and Cypher statements
so they can be reviewed and modified in one place.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
#  NODE LABELS
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_LABEL = "Entity"
CHUNK_LABEL  = "Chunk"


# ─────────────────────────────────────────────────────────────────────────────
#  RELATIONSHIP LABELS
# ─────────────────────────────────────────────────────────────────────────────

APPEARS_IN_REL = "APPEARS_IN"

# All valid relationship types from the extraction layer
VALID_RELATIONSHIP_TYPES = frozenset({
    "RELATED_TO",
    "USES",
    "DEPENDS_ON",
    "CONTRIBUTES_TO",
    "IMPROVES",
    "REDUCES",
    "INCREASES",
    "PART_OF",
    "BELONGS_TO",
    "LOCATED_IN",
    "CUSTOM",
})


# ─────────────────────────────────────────────────────────────────────────────
#  PROPERTY KEYS
# ─────────────────────────────────────────────────────────────────────────────

# Entity node properties
PROP_ENTITY_ID   = "entity_id"
PROP_ENTITY_NAME = "entity_name"
PROP_ENTITY_TYPE = "entity_type"

# Chunk node properties
PROP_CHUNK_ID    = "chunk_id"
PROP_PAGE_NUMBER = "page_number"

# Relationship properties
PROP_RELATIONSHIP_ID = "relationship_id"
PROP_CONFIDENCE      = "confidence"


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTRAINT CYPHER
# ─────────────────────────────────────────────────────────────────────────────

CYPHER_CONSTRAINT_ENTITY_ID = (
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
)

CYPHER_CONSTRAINT_CHUNK_ID = (
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
)


# ─────────────────────────────────────────────────────────────────────────────
#  INDEX CYPHER
# ─────────────────────────────────────────────────────────────────────────────

CYPHER_INDEX_ENTITY_NAME = (
    "CREATE INDEX entity_name_idx IF NOT EXISTS "
    "FOR (e:Entity) ON (e.entity_name)"
)

CYPHER_INDEX_ENTITY_TYPE = (
    "CREATE INDEX entity_type_idx IF NOT EXISTS "
    "FOR (e:Entity) ON (e.entity_type)"
)

CYPHER_INDEX_CHUNK_PAGE = (
    "CREATE INDEX chunk_page_idx IF NOT EXISTS "
    "FOR (c:Chunk) ON (c.page_number)"
)


# ─────────────────────────────────────────────────────────────────────────────
#  MERGE CYPHER TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

# MERGE an Entity node (upsert by entity_id)
CYPHER_MERGE_ENTITY = """
MERGE (e:Entity {entity_id: $entity_id})
ON CREATE SET
    e.entity_name = $entity_name,
    e.entity_type = $entity_type
ON MATCH SET
    e.entity_name = $entity_name,
    e.entity_type = $entity_type
"""

# MERGE a Chunk node (upsert by chunk_id)
CYPHER_MERGE_CHUNK = """
MERGE (c:Chunk {chunk_id: $chunk_id})
ON CREATE SET c.page_number = $page_number
ON MATCH SET  c.page_number = $page_number
"""

# MERGE an APPEARS_IN relationship (entity → chunk)
CYPHER_MERGE_APPEARS_IN = """
MATCH (e:Entity {entity_id: $entity_id})
MATCH (c:Chunk  {chunk_id:  $chunk_id})
MERGE (e)-[r:APPEARS_IN]->(c)
ON CREATE SET r.confidence = $confidence
ON MATCH SET  r.confidence = $confidence
"""

# MERGE a typed relationship between two entities.
# Because Neo4j does not support parameterised relationship types,
# we generate one template per relationship type from the enum.
def cypher_merge_relationship(rel_type: str) -> str:
    """Return a MERGE Cypher statement for a specific relationship type.

    Parameters
    ----------
    rel_type:
        A valid relationship type string (e.g. ``"USES"``).
        Must be in ``VALID_RELATIONSHIP_TYPES``.

    Returns
    -------
    str
        A parameterised Cypher MERGE statement.
    """
    if rel_type not in VALID_RELATIONSHIP_TYPES:
        raise ValueError(
            f"Unknown relationship type: {rel_type!r}. "
            f"Must be one of {sorted(VALID_RELATIONSHIP_TYPES)}"
        )

    return f"""
MATCH (src:Entity {{entity_id: $source_entity_id}})
MATCH (tgt:Entity {{entity_id: $target_entity_id}})
MERGE (src)-[r:{rel_type} {{relationship_id: $relationship_id}}]->(tgt)
ON CREATE SET
    r.confidence  = $confidence,
    r.chunk_id    = $chunk_id,
    r.page_number = $page_number
ON MATCH SET
    r.confidence  = $confidence,
    r.chunk_id    = $chunk_id,
    r.page_number = $page_number
"""


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION / HEALTH-CHECK CYPHER
# ─────────────────────────────────────────────────────────────────────────────

CYPHER_COUNT_ENTITIES = "MATCH (e:Entity) RETURN count(e) AS count"
CYPHER_COUNT_CHUNKS   = "MATCH (c:Chunk) RETURN count(c) AS count"
CYPHER_COUNT_APPEARS  = "MATCH ()-[r:APPEARS_IN]->() RETURN count(r) AS count"

CYPHER_COUNT_RELATIONSHIPS_BY_TYPE = """
MATCH ()-[r]->()
WHERE NOT type(r) = 'APPEARS_IN'
RETURN type(r) AS rel_type, count(r) AS count
ORDER BY count DESC
"""

CYPHER_ORPHAN_ENTITIES = """
MATCH (e:Entity)
WHERE NOT (e)-[:APPEARS_IN]->()
RETURN e.entity_id AS entity_id, e.entity_name AS entity_name
"""
