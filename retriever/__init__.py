"""
GraphRAG Agentic Retrieval Pipeline.

Public API
----------
- ``RetrievalPipeline`` — end-to-end 7-agent retrieval
- ``QueryPlannerAgent`` — Stage 1
- ``SemanticRetrievalAgent`` — Stage 2
- ``GraphRetrievalAgent`` — Stage 3
- ``FusionAgent`` — Stage 4
- ``RerankingAgent`` — Stage 5
- ``EvidenceAgent`` — Stage 6
- ``ContextBuilderAgent`` — Stage 7
- ``retrieve()`` — convenience function for single-shot queries
"""

from retriever.pipeline import RetrievalPipeline
from retriever.query_understanding import QueryPlannerAgent
from retriever.semantic_retriever import SemanticRetrievalAgent
from retriever.graph_retriever import GraphRetrievalAgent
from retriever.orchestrator import FusionAgent
from retriever.reranker import RerankingAgent
from retriever.evidence_agent import EvidenceAgent
from retriever.context_builder import ContextBuilderAgent
from retriever.schema import (
    QueryAnalysis,
    QueryIntent,
    RetrievalCandidate,
    RetrievalResult,
    GraphCandidate,
    GraphEntity,
    GraphRelationship,
)


def retrieve(query: str, **kwargs) -> RetrievalResult:
    """Convenience function: run the full pipeline on a single query.

    Parameters
    ----------
    query:
        The user's natural-language question.
    **kwargs:
        Passed to ``RetrievalPipeline.__init__()``.

    Returns
    -------
    RetrievalResult
    """
    pipeline = RetrievalPipeline(**kwargs)
    try:
        return pipeline.retrieve(query)
    finally:
        pipeline.close()


__all__ = [
    "RetrievalPipeline",
    "QueryPlannerAgent",
    "SemanticRetrievalAgent",
    "GraphRetrievalAgent",
    "FusionAgent",
    "RerankingAgent",
    "EvidenceAgent",
    "ContextBuilderAgent",
    "QueryAnalysis",
    "QueryIntent",
    "RetrievalCandidate",
    "RetrievalResult",
    "GraphCandidate",
    "GraphEntity",
    "GraphRelationship",
    "retrieve",
]
