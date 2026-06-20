"""
Relationship Extraction Layer for the GraphRAG pipeline.

Public API
----------
- ``RelationshipExtractor``      — main class to extract relationships
- ``run_relationship_extraction`` — convenience function for one-call usage

Usage
-----
>>> from relationship import RelationshipExtractor
>>> extractor = RelationshipExtractor()
>>> result = extractor.extract_all(chunks, entities)

Or run from the command line:
    python -m relationship.runner
"""

from relationship.extractor import RelationshipExtractor
from relationship.schema import (
    ExtractedRelationship,
    LLMRelationshipOutput,
    RelationshipExtractionResult,
    RelationshipType,
)


def run_relationship_extraction(
    chunks: list,
    entities: list,
) -> RelationshipExtractionResult:
    """Convenience wrapper: build an extractor and process all chunks.

    Parameters
    ----------
    chunks:
        List of chunk dicts (loaded from chunks.json).
    entities:
        List of entity dicts (loaded from entities.json).

    Returns
    -------
    RelationshipExtractionResult
        Aggregated extraction result with all relationships.
    """
    extractor = RelationshipExtractor()
    return extractor.extract_all(chunks, entities)


__all__ = [
    "RelationshipExtractor",
    "RelationshipType",
    "ExtractedRelationship",
    "LLMRelationshipOutput",
    "RelationshipExtractionResult",
    "run_relationship_extraction",
]
