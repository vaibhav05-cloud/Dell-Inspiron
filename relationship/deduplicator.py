"""
Intra-chunk relationship deduplication.

Removes duplicate relationships within the same chunk based on
(source_entity_id, target_entity_id, relationship_type).
When duplicates are found the entry with the highest confidence
score is kept.
"""

from __future__ import annotations

from typing import List

from relationship.schema import ExtractedRelationship


def deduplicate_relationships(
    relationships: List[ExtractedRelationship],
) -> List[ExtractedRelationship]:
    """Remove duplicate relationships within a single chunk.

    Duplicates are identified by the composite key
    ``(source_entity_id, target_entity_id, relationship_type)``.

    When two or more relationships share the same key the one with
    the highest ``confidence`` score is retained.

    Parameters
    ----------
    relationships:
        List of relationships extracted from **one** chunk.

    Returns
    -------
    List[ExtractedRelationship]
        De-duplicated list, preserving original insertion order
        for the winning entries.
    """
    seen: dict[tuple[str, str, str], ExtractedRelationship] = {}

    for rel in relationships:
        key = (
            rel.source_entity_id,
            rel.target_entity_id,
            rel.relationship_type.value,
        )

        if key not in seen:
            seen[key] = rel
        else:
            # Keep the higher-confidence entry
            if rel.confidence > seen[key].confidence:
                seen[key] = rel

    return list(seen.values())
