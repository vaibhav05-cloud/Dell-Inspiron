"""
Relationship schema definitions for the Relationship Extraction Layer.

Defines relationship types, the canonical relationship model, and
container types used throughout the extraction pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
#  RELATIONSHIP TYPES
# ─────────────────────────────────────────────────────────────────────────────

class RelationshipType(str, Enum):
    """All relationship types the extraction layer recognises."""

    RELATED_TO      = "RELATED_TO"
    USES            = "USES"
    DEPENDS_ON      = "DEPENDS_ON"
    CONTRIBUTES_TO  = "CONTRIBUTES_TO"
    IMPROVES        = "IMPROVES"
    REDUCES         = "REDUCES"
    INCREASES       = "INCREASES"
    PART_OF         = "PART_OF"
    BELONGS_TO      = "BELONGS_TO"
    LOCATED_IN      = "LOCATED_IN"
    CUSTOM          = "CUSTOM"


# ─────────────────────────────────────────────────────────────────────────────
#  LLM OUTPUT SCHEMA  (what the model returns)
# ─────────────────────────────────────────────────────────────────────────────

class LLMRelationship(BaseModel):
    """Schema bound to the LLM's structured-output parser.

    The LLM references entities by *name*.  The extractor resolves
    names to entity_ids after the call.
    """

    source_entity_name: str = Field(
        ...,
        description="Name of the source entity (must match an entity "
                    "from the provided entity list).",
    )
    target_entity_name: str = Field(
        ...,
        description="Name of the target entity (must match an entity "
                    "from the provided entity list).",
    )
    relationship_type: RelationshipType = Field(
        ...,
        description="One of: RELATED_TO, USES, DEPENDS_ON, CONTRIBUTES_TO, "
                    "IMPROVES, REDUCES, INCREASES, PART_OF, BELONGS_TO, "
                    "LOCATED_IN, CUSTOM.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )


class LLMRelationshipOutput(BaseModel):
    """Top-level wrapper returned by the relationship extraction chain."""

    relationships: List[LLMRelationship] = Field(
        default_factory=list,
        description="List of relationships extracted from the chunk.",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CANONICAL RELATIONSHIP  (stored in relationships.json)
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedRelationship(BaseModel):
    """Fully-qualified relationship ready for persistence."""

    relationship_id:   str              = Field(..., description="Unique ID, e.g. rel_chunk_18_0")
    source_entity_id:  str              = Field(..., description="entity_id of the source entity")
    target_entity_id:  str              = Field(..., description="entity_id of the target entity")
    relationship_type: RelationshipType = Field(..., description="Relationship type enum value")
    confidence:        float            = Field(..., ge=0.0, le=1.0, description="0.0–1.0 confidence")
    chunk_id:          str              = Field(..., description="Source chunk ID from chunks.json")
    page_number:       int              = Field(..., description="PDF page number")


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION RESULT  (full pipeline output)
# ─────────────────────────────────────────────────────────────────────────────

class RelationshipExtractionResult(BaseModel):
    """Container for the complete relationship extraction run."""

    total_chunks_processed: int                    = 0
    total_relationships_extracted: int             = 0
    relationships_by_type: dict[str, int]          = Field(default_factory=dict)
    relationships: List[ExtractedRelationship]     = Field(default_factory=list)
