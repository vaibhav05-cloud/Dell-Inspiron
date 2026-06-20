"""
Answer Synthesis — Pydantic Data Models.

Defines the output schema for the Answer Synthesis & Response Generation Layer,
including evidence attribution, reasoning paths, and confidence scoring.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIDENCE LEVEL
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    """Discrete confidence bucket for the synthesized answer."""

    HIGH   = "High"
    MEDIUM = "Medium"
    LOW    = "Low"


# ─────────────────────────────────────────────────────────────────────────────
#  EVIDENCE ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceAttribution(BaseModel):
    """Citation for a single evidence chunk used in the answer."""

    page_number: int = Field(..., description="Source PDF page number.")
    chunk_id: str    = Field(..., description="Unique chunk identifier.")
    source_document: str = Field(
        ..., description="Filename of the source document."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SYNTHESIS RESULT (final user-facing response)
# ─────────────────────────────────────────────────────────────────────────────

class SynthesisResult(BaseModel):
    """Final output of the Answer Synthesis Layer.

    Conforms to the required JSON response format:
    {
        "answer": "...",
        "evidence": [...],
        "reasoning_path": [...],
        "confidence": "High"
    }
    """

    answer: str = Field(
        ...,
        description="Concise, grounded answer synthesized from evidence.",
    )
    evidence: List[EvidenceAttribution] = Field(
        ...,
        description="Source attributions for every piece of evidence used.",
    )
    reasoning_path: List[str] = Field(
        ...,
        description="Human-readable reasoning trail using graph relationships.",
    )
    confidence: str = Field(
        ...,
        description="Confidence level: High, Medium, or Low.",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CONSOLIDATED EVIDENCE (internal — used between consolidation and LLM call)
# ─────────────────────────────────────────────────────────────────────────────

class ConsolidatedEvidence(BaseModel):
    """Internal representation of merged evidence from the Top 3 chunks.

    This is NOT part of the final output — it feeds the LLM prompt.
    """

    merged_context: str = Field(
        ...,
        description="Deduplicated, merged text from all top chunks.",
    )
    common_facts: List[str] = Field(
        default_factory=list,
        description="Facts appearing in 2+ chunks (high confidence).",
    )
    complementary_facts: List[str] = Field(
        default_factory=list,
        description="Facts unique to a single chunk.",
    )
    graph_relationships: List[str] = Field(
        default_factory=list,
        description="Formatted relationship strings from graph_paths.",
    )
    attributions: List[EvidenceAttribution] = Field(
        default_factory=list,
        description="Evidence attributions extracted from the top chunks.",
    )
    relevance_scores: List[float] = Field(
        default_factory=list,
        description="Rerank scores of the top chunks (for confidence calc).",
    )
    relationship_confidences: List[float] = Field(
        default_factory=list,
        description="Confidence values from graph paths (for confidence calc).",
    )
