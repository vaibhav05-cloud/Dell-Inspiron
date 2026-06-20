"""
Idempotent Neo4j constraint and index creation.

All statements use ``IF NOT EXISTS`` so they are safe to run
repeatedly without error.
"""

from __future__ import annotations

import logging

from graph_builder.schema import (
    CYPHER_CONSTRAINT_CHUNK_ID,
    CYPHER_CONSTRAINT_ENTITY_ID,
    CYPHER_INDEX_CHUNK_PAGE,
    CYPHER_INDEX_ENTITY_NAME,
    CYPHER_INDEX_ENTITY_TYPE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ensure_constraints(session) -> None:
    """Create uniqueness constraints if they do not already exist.

    Parameters
    ----------
    session:
        An open ``neo4j.Session``.
    """
    constraints = [
        ("entity_id_unique", CYPHER_CONSTRAINT_ENTITY_ID),
        ("chunk_id_unique",  CYPHER_CONSTRAINT_CHUNK_ID),
    ]

    for name, cypher in constraints:
        logger.info(f"  Ensuring constraint: {name}")
        session.run(cypher)

    logger.info("  Constraints ready ✓")


def ensure_indexes(session) -> None:
    """Create indexes if they do not already exist.

    Parameters
    ----------
    session:
        An open ``neo4j.Session``.
    """
    indexes = [
        ("entity_name_idx", CYPHER_INDEX_ENTITY_NAME),
        ("entity_type_idx", CYPHER_INDEX_ENTITY_TYPE),
        ("chunk_page_idx",  CYPHER_INDEX_CHUNK_PAGE),
    ]

    for name, cypher in indexes:
        logger.info(f"  Ensuring index: {name}")
        session.run(cypher)

    logger.info("  Indexes ready ✓")


def ensure_schema(session) -> None:
    """Create all constraints and indexes (idempotent).

    Convenience function that calls both ``ensure_constraints``
    and ``ensure_indexes``.

    Parameters
    ----------
    session:
        An open ``neo4j.Session``.
    """
    logger.info("Setting up Neo4j schema …")
    ensure_constraints(session)
    ensure_indexes(session)
    logger.info("Schema setup complete ✓")
