"""
Stage 5 — Re-ranking Agent.

Uses a lightweight cross-encoder model to re-score all merged and fused
candidates against the original query, producing the final ranked list.
"""

from __future__ import annotations

import logging
from typing import List

from retriever.schema import RetrievalCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Lightweight cross-encoder (~80 MB).
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingAgent:
    """Re-ranks fused retrieval candidates using a cross-encoder model."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        logger.info(f"  Loading re-ranker model: {self._model_name} …")
        self._model = CrossEncoder(self._model_name)
        logger.info("  Re-ranker model loaded ✓")

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int = 15,
    ) -> List[RetrievalCandidate]:
        """Re-rank candidates using the cross-encoder.

        Parameters
        ----------
        query:
            The original user query.
        candidates:
            Merged/fused candidates from Fusion Agent.
        top_k:
            Maximum number of results to keep after re-ranking.

        Returns
        -------
        List[RetrievalCandidate]
            Top-K candidates sorted by re-ranking score descending.
        """
        if not candidates:
            return []

        self._ensure_loaded()

        logger.info(f"Stage 5: Running Re-ranking Agent (scoring {len(candidates)} candidates) …")

        # Build (query, candidate) pairs for batch scoring
        pairs = [(query, c.content) for c in candidates]

        scores = self._model.predict(pairs)

        scored = []
        for candidate, score in zip(candidates, scores):
            updated = candidate.model_copy(
                update={"rerank_score": float(score)}
            )
            scored.append(updated)

        scored.sort(key=lambda x: x.rerank_score, reverse=True)

        result = scored[:top_k]

        if result:
            logger.info(
                f"  Reranking complete. Top score: {result[0].rerank_score:.4f}, "
                f"bottom score: {result[-1].rerank_score:.4f}"
            )

        return result
