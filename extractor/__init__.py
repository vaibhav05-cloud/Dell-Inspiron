"""
Entity Extraction Layer for the GraphRAG pipeline.

Public API
----------
- ``EntityExtractor``  — main class to extract entities from chunks
- ``run_extraction``   — convenience function for one-call usage

Usage
-----
>>> from extractor import EntityExtractor
>>> extractor = EntityExtractor()
>>> result = extractor.extract_all(chunks)

Or run from the command line:
    python -m extractor.runner
"""

from extractor.extractor import EntityExtractor
from extractor.schema import (
    EntityType,
    ExtractedEntity,
    ExtractionResult,
    LLMEntityOutput,
)


def run_extraction(
    chunks: list,
) -> ExtractionResult:
    """Convenience wrapper: build an extractor and process all chunks.

    Parameters
    ----------
    chunks:
        List of chunk dicts (loaded from chunks.json).

    Returns
    -------
    ExtractionResult
        Aggregated extraction result with all entities.
    """
    extractor = EntityExtractor()
    return extractor.extract_all(chunks)


__all__ = [
    "EntityExtractor",
    "EntityType",
    "ExtractedEntity",
    "ExtractionResult",
    "LLMEntityOutput",
    "run_extraction",
]
