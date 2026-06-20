"""
Intra-chunk entity deduplication.

Removes duplicate entities within the same chunk based on
(entity_name_lower, entity_type).  When duplicates are found
the entry with the highest confidence score is kept.
"""

from __future__ import annotations

from typing import List

from extractor.schema import ExtractedEntity


def deduplicate(entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
    """Remove duplicate entities within a single chunk.

    Duplicates are identified by the composite key
    ``(entity_name.lower().strip(), entity_type)``.

    When two or more entities share the same key the one with
    the highest ``confidence`` score is retained.

    Parameters
    ----------
    entities:
        List of entities extracted from **one** chunk.

    Returns
    -------
    List[ExtractedEntity]
        De-duplicated list, preserving original insertion order
        for the winning entries.
    """
    seen: dict[tuple[str, str], ExtractedEntity] = {}

    for entity in entities:
        key = (
            entity.entity_name.lower().strip(),
            entity.entity_type.value,
        )

        if key not in seen:
            seen[key] = entity
        else:
            # Keep the higher-confidence entry
            if entity.confidence > seen[key].confidence:
                seen[key] = entity

    return list(seen.values())
