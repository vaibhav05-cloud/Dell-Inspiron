"""
Entity schema definitions for the Entity Extraction Layer.

Defines all entity types, the canonical entity model, and
container types used throughout the extraction pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
#  ENTITY TYPES
# ─────────────────────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    """All entity types the extraction layer recognises."""

    ORGANIZATION = "ORGANIZATION"
    PERSON       = "PERSON"
    PRODUCT      = "PRODUCT"
    TECHNOLOGY   = "TECHNOLOGY"
    PROJECT      = "PROJECT"
    LOCATION     = "LOCATION"
    METRIC_KPI   = "METRIC_KPI"
    DATE         = "DATE"
    CONCEPT      = "CONCEPT"


# ─────────────────────────────────────────────────────────────────────────────
#  LLM OUTPUT SCHEMA  (what the model returns)
# ─────────────────────────────────────────────────────────────────────────────

class LLMEntity(BaseModel):
    """Schema bound to the LLM's structured-output parser.

    The LLM populates *entity_name*, *entity_type*, *source_text*,
    and *confidence*.  The remaining fields (entity_id, chunk_id,
    page_number) are attached after the call by the extractor.
    """

    entity_name: str = Field(
        ...,
        description="Canonical name of the extracted entity.",
    )
    entity_type: EntityType = Field(
        ...,
        description="One of: ORGANIZATION, PERSON, PRODUCT, TECHNOLOGY, "
                    "PROJECT, LOCATION, METRIC_KPI, DATE, CONCEPT.",
    )
    source_text: str = Field(
        ...,
        description="The verbatim snippet from the chunk that evidences "
                    "this entity (keep it short, ≤120 chars).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )


class LLMEntityOutput(BaseModel):
    """Top-level wrapper returned by the extraction chain."""

    entities: List[LLMEntity] = Field(
        default_factory=list,
        description="List of entities extracted from the chunk.",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CANONICAL ENTITY  (stored in entities.json)
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedEntity(BaseModel):
    """Fully-qualified entity ready for persistence."""

    entity_id:   str        = Field(..., description="Unique ID, e.g. ent_chunk_18_0")
    entity_name: str        = Field(..., description="Canonical entity name")
    entity_type: EntityType = Field(..., description="Entity type enum value")
    chunk_id:    str        = Field(..., description="Source chunk ID from chunks.json")
    page_number: int        = Field(..., description="PDF page number")
    source_text: str        = Field(..., description="Evidence snippet from chunk")
    confidence:  float      = Field(..., ge=0.0, le=1.0, description="0.0–1.0 confidence")


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION RESULT  (full pipeline output)
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionResult(BaseModel):
    """Container for the complete extraction run."""

    total_chunks_processed: int                = 0
    total_entities_extracted: int              = 0
    entities_by_type: dict[str, int]           = Field(default_factory=dict)
    entities: List[ExtractedEntity]            = Field(default_factory=list)
