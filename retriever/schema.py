"""
Pydantic data models for the GraphRAG retrieval pipeline.

Defines all inter-stage data structures used to pass data
between retrieval pipeline stages.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
#  QUERY INTENT
# ─────────────────────────────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    """Classifies the user's query intent."""

    FACTUAL     = "FACTUAL"       # Direct fact lookup
    RELATIONSHIP = "RELATIONSHIP"  # How entities relate
    COMPARISON  = "COMPARISON"     # Compare entities/concepts
    PROCEDURAL  = "PROCEDURAL"     # How-to / step-by-step
    EXPLORATORY = "EXPLORATORY"    # Open-ended exploration


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1: QUERY ANALYSIS & RETRIEVAL PLAN
# ─────────────────────────────────────────────────────────────────────────────

class AgenticRetrievalPlan(BaseModel):
    """Planner decision on which retrieval strategies are needed."""

    semantic_search_needed: bool = Field(
        default=True,
        description="Whether to perform semantic search.",
    )
    graph_search_needed: bool = Field(
        default=True,
        description="Whether to perform Neo4j graph traversal.",
    )
    both_needed: bool = Field(
        default=True,
        description="Whether both search types are needed.",
    )
    traversal_depth: int = Field(
        default=2,
        ge=1,
        le=2,
        description="Max graph traversal depth (1 or 2).",
    )


class QueryAnalysis(BaseModel):
    """Output of the Query Planner Agent."""

    query_entities: List[str] = Field(
        default_factory=list,
        description="Entity names extracted from the query.",
    )
    query_intent: QueryIntent = Field(
        default=QueryIntent.FACTUAL,
        description="Classified intent of the query.",
    )
    retrieval_plan: AgenticRetrievalPlan = Field(
        default_factory=AgenticRetrievalPlan,
        description="Strategy for retrieval source weighting.",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  RETRIEVAL CANDIDATE (unified chunk representation)
# ─────────────────────────────────────────────────────────────────────────────

class CandidateSource(str, Enum):
    """Where a retrieval candidate originated."""

    SEMANTIC = "SEMANTIC"
    GRAPH    = "GRAPH"
    BOTH     = "BOTH"


class RetrievalCandidate(BaseModel):
    """A single chunk candidate from any retrieval source."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk text content")
    page_number: int = Field(default=0, description="Source PDF page")
    section_name: str = Field(default="", description="Section heading")
    source_file: str = Field(default="", description="Source PDF filename")
    source: CandidateSource = Field(
        default=CandidateSource.SEMANTIC,
        description="Which retrieval source produced this candidate.",
    )
    similarity_score: float = Field(
        default=0.0,
        description="Semantic similarity score (0.0–1.0).",
    )
    graph_relevance_score: float = Field(
        default=0.0,
        description="Graph relevance score based on traversal distance and entity frequency (0.0–1.0).",
    )
    relationship_confidence: float = Field(
        default=0.0,
        description="Average confidence of relationships connecting this chunk's entities (0.0–1.0).",
    )
    hybrid_score: float = Field(
        default=0.0,
        description="Weighted hybrid score.",
    )
    rerank_score: float = Field(
        default=0.0,
        description="Cross-encoder re-ranking score.",
    )
    supporting_context: str = Field(
        default="",
        description="Additional context from expansion (neighbor chunks, graph paths).",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH CANDIDATE (entity/relationship metadata from traversal)
# ─────────────────────────────────────────────────────────────────────────────

class GraphEntity(BaseModel):
    """An entity retrieved from the Neo4j graph."""

    entity_id: str   = Field(..., description="Unique entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type label")


class GraphRelationship(BaseModel):
    """A relationship retrieved from the Neo4j graph."""

    source_name: str       = Field(..., description="Source entity name")
    target_name: str       = Field(..., description="Target entity name")
    relationship_type: str = Field(..., description="Relationship type label")
    confidence: float      = Field(default=0.0, description="Confidence score")


class GraphCandidate(BaseModel):
    """Container for graph traversal results."""

    matched_entities: List[GraphEntity] = Field(
        default_factory=list,
        description="Entities matched from the query.",
    )
    connected_entities: List[GraphEntity] = Field(
        default_factory=list,
        description="Entities found via traversal.",
    )
    relationships: List[GraphRelationship] = Field(
        default_factory=list,
        description="Relationships discovered during traversal.",
    )
    chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk IDs associated with graph-matched entities.",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE TIMINGS & PIPELINE RESULT
# ─────────────────────────────────────────────────────────────────────────────

class StageTiming(BaseModel):
    """Timing information for a single pipeline stage."""

    stage_name: str    = Field(..., description="Name of the stage")
    duration_ms: float = Field(..., description="Duration in milliseconds")


class RetrievalResult(BaseModel):
    """Final output of the retrieval pipeline, conforming to user specifications."""

    answer_context: str = Field(
        ...,
        description="Compressed context string ready for LLM consumption.",
    )
    evidence_chunks: List[dict] = Field(
        ...,
        description="Detailed list of source chunks serving as evidence.",
    )
    graph_paths: List[dict] = Field(
        ...,
        description="List of graph relationship paths traversed.",
    )
    source_entities: List[dict] = Field(
        ...,
        description="Unique entities matched or traversed in the graph.",
    )
    retrieval_metadata: dict = Field(
        ...,
        description="Performance metrics, planner decisions, and workflow execution metadata.",
    )
