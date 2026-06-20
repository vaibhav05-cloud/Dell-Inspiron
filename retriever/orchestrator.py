"""
Stage 4 — Fusion Agent.

Merges and deduplicates candidates from Semantic Retrieval Agent and Graph Retrieval Agent,
applying a hybrid scoring formula based on semantic similarity, graph relevance, and relationship confidence.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from retriever.schema import CandidateSource, RetrievalCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  HYBRID SCORING WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

WEIGHT_SEMANTIC: float = 0.5
WEIGHT_GRAPH: float = 0.3
WEIGHT_REL_CONFIDENCE: float = 0.2
INTERSECTION_BONUS: float = 0.05


# ─────────────────────────────────────────────────────────────────────────────
#  FUSION AGENT
# ─────────────────────────────────────────────────────────────────────────────

class FusionAgent:
    """Merges semantic and graph retrieval results, deduplicating candidates."""

    def __init__(
        self,
        weight_semantic: float = WEIGHT_SEMANTIC,
        weight_graph: float = WEIGHT_GRAPH,
        weight_rel_confidence: float = WEIGHT_REL_CONFIDENCE,
    ):
        self._w_sem = weight_semantic
        self._w_graph = weight_graph
        self._w_rel = weight_rel_confidence

    @staticmethod
    def _normalize_scores(
        candidates: List[RetrievalCandidate],
        field: str,
    ) -> Dict[str, float]:
        """Min-max normalize a score field to [0, 1]."""
        values = [getattr(c, field) for c in candidates]
        if not values:
            return {}

        min_v = min(values)
        max_v = max(values)
        range_v = max_v - min_v

        normalized = {}
        for c in candidates:
            raw = getattr(c, field)
            normalized[c.chunk_id] = (
                (raw - min_v) / range_v if range_v > 0 else 0.5
            )

        return normalized

    def _compute_hybrid_score(
        self,
        semantic_score: float,
        graph_score: float,
        rel_confidence: float,
        in_both: bool,
    ) -> float:
        """Compute weighted hybrid score with optional intersection bonus."""
        score = (
            self._w_sem * semantic_score
            + self._w_graph * graph_score
            + self._w_rel * rel_confidence
        )

        if in_both:
            score += INTERSECTION_BONUS

        return round(min(score, 1.0), 4)

    def fuse(
        self,
        semantic_candidates: List[RetrievalCandidate],
        graph_candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """Merge and deduplicate candidates using hybrid scoring.

        Parameters
        ----------
        semantic_candidates:
            Candidates from FAISS similarity search.
        graph_candidates:
            Candidates from Neo4j graph traversal.

        Returns
        -------
        List[RetrievalCandidate]
            Fused, deduplicated candidates sorted by hybrid score descending.
        """
        logger.info("Stage 4: Running Fusion Agent …")
        logger.info(
            f"  Semantic Candidates: {len(semantic_candidates)}, Graph Candidates: {len(graph_candidates)}"
        )

        semantic_map: Dict[str, RetrievalCandidate] = {
            c.chunk_id: c for c in semantic_candidates
        }
        graph_map: Dict[str, RetrievalCandidate] = {
            c.chunk_id: c for c in graph_candidates
        }

        all_chunk_ids = set(semantic_map.keys()) | set(graph_map.keys())
        intersection_ids = set(semantic_map.keys()) & set(graph_map.keys())

        sem_norm = self._normalize_scores(
            semantic_candidates, "similarity_score"
        )

        merged: List[RetrievalCandidate] = []

        for cid in all_chunk_ids:
            sem_cand = semantic_map.get(cid)
            graph_cand = graph_map.get(cid)
            in_both = cid in intersection_ids

            base = sem_cand or graph_cand

            semantic_score = sem_norm.get(cid, 0.0)
            graph_score = (
                graph_cand.graph_relevance_score if graph_cand else 0.0
            )
            rel_confidence = (
                graph_cand.relationship_confidence if graph_cand else 0.0
            )

            hybrid = self._compute_hybrid_score(
                semantic_score=semantic_score,
                graph_score=graph_score,
                rel_confidence=rel_confidence,
                in_both=in_both,
            )

            if in_both:
                source = CandidateSource.BOTH
            elif sem_cand:
                source = CandidateSource.SEMANTIC
            else:
                source = CandidateSource.GRAPH

            merged_candidate = base.model_copy(
                update={
                    "source": source,
                    "similarity_score": (
                        sem_cand.similarity_score if sem_cand else 0.0
                    ),
                    "graph_relevance_score": graph_score,
                    "relationship_confidence": rel_confidence,
                    "hybrid_score": hybrid,
                }
            )
            merged.append(merged_candidate)

        merged.sort(key=lambda x: x.hybrid_score, reverse=True)

        logger.info(f"  Fusion complete. Fused into {len(merged)} candidates.")
        return merged
