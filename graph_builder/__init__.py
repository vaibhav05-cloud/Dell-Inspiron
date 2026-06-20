"""
Neo4j Graph Builder for the GraphRAG pipeline.

Public API
----------
- ``GraphIngestionService`` — main class to ingest entities + relationships
- ``Neo4jConnection``       — managed Neo4j driver connection
- ``run_graph_builder``     — convenience function for one-call usage

Usage
-----
>>> from graph_builder import Neo4jConnection, GraphIngestionService
>>> with Neo4jConnection() as conn:
...     service = GraphIngestionService(conn)
...     stats = service.ingest_all(entities, relationships)

Or run from the command line:
    python -m graph_builder.runner
"""

from graph_builder.connection import Neo4jConnection
from graph_builder.ingestion import GraphIngestionService
from graph_builder.validator import validate_graph, validate_inputs


def run_graph_builder(
    entities: list,
    relationships: list,
    batch_size: int = 100,
) -> dict:
    """Convenience wrapper: connect, validate, ingest, and health-check.

    Parameters
    ----------
    entities:
        List of entity dicts (loaded from entities.json).
    relationships:
        List of relationship dicts (loaded from relationships.json).
    batch_size:
        Batch size for MERGE operations.

    Returns
    -------
    dict
        Combined ingestion statistics.
    """
    # Pre-validation
    report = validate_inputs(entities, relationships)
    if not report["valid"]:
        raise ValueError(
            f"Pre-ingestion validation failed: {report['errors']}"
        )

    # Ingest
    with Neo4jConnection() as conn:
        service = GraphIngestionService(conn)
        stats = service.ingest_all(entities, relationships, batch_size)

        # Post-validation
        with conn.session() as session:
            validate_graph(session)

    return stats


__all__ = [
    "GraphIngestionService",
    "Neo4jConnection",
    "run_graph_builder",
    "validate_graph",
    "validate_inputs",
]
