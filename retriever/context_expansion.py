"""
Stage 6 — Context Expansion Agent.

For each top-ranked chunk, retrieves neighbor chunks and
supporting graph relationships. Limits expansion aggressively
to avoid token explosion.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from retriever.schema import GraphRelationship, RetrievalCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  LIMITS
# ─────────────────────────────────────────────────────────────────────────────

MAX_NEIGHBOR_CHUNKS = 2        # neighbor chunks per candidate
MAX_GRAPH_RELS_PER_CHUNK = 5   # graph relationships per candidate


class ContextExpansionAgent:
    """Expands top-ranked chunks with neighbor chunks and graph relationships."""

    def __init__(
        self,
        chunks_lookup: Dict[str, dict],
        graph_retriever=None,
        entities_lookup: Optional[Dict[str, dict]] = None,
    ):
        """
        Parameters
        ----------
        chunks_lookup:
            Dict mapping chunk_id → chunk data (from chunks.json).
        graph_retriever:
            Optional GraphRetriever instance for relationship lookups.
        entities_lookup:
            Optional Dict mapping entity_id → entity data (from entities.json).
        """
        self._chunks_lookup = chunks_lookup
        self._graph_retriever = graph_retriever
        self._entities_lookup = entities_lookup or {}

        # Build ordered list of chunk IDs for neighbor lookup
        self._ordered_ids = sorted(
            chunks_lookup.keys(),
            key=lambda x: self._chunk_sort_key(x),
        )
        self._id_to_index = {
            cid: i for i, cid in enumerate(self._ordered_ids)
        }

    @staticmethod
    def _chunk_sort_key(chunk_id: str) -> int:
        """Extract numeric part from chunk_id for ordering."""
        try:
            return int(chunk_id.replace("chunk_", ""))
        except (ValueError, AttributeError):
            return 0

    def _get_neighbor_chunks(self, chunk_id: str) -> List[dict]:
        """Get adjacent chunks (previous and next) for a given chunk."""
        idx = self._id_to_index.get(chunk_id)
        if idx is None:
            return []

        neighbors = []
        # Previous chunk
        if idx > 0:
            prev_id = self._ordered_ids[idx - 1]
            prev_data = self._chunks_lookup.get(prev_id)
            if prev_data:
                neighbors.append(prev_data)

        # Next chunk
        if idx < len(self._ordered_ids) - 1:
            next_id = self._ordered_ids[idx + 1]
            next_data = self._chunks_lookup.get(next_id)
            if next_data:
                neighbors.append(next_data)

        return neighbors[:MAX_NEIGHBOR_CHUNKS]

    def _get_graph_context(self, chunk_id: str) -> List[GraphRelationship]:
        """Get graph relationships for entities in this chunk."""
        if not self._graph_retriever or not self._entities_lookup:
            return []

        # Find entity_ids that belong to this chunk
        entity_ids = [
            eid for eid, edata in self._entities_lookup.items()
            if edata.get("chunk_id") == chunk_id
        ]

        rels = []
        for eid in entity_ids[:3]:  # limit to 3 entities per chunk
            try:
                chunk_rels = self._graph_retriever.get_entity_relationships(
                    entity_id=eid,
                    max_rels=MAX_GRAPH_RELS_PER_CHUNK,
                )
                rels.extend(chunk_rels)
            except Exception:
                continue

            if len(rels) >= MAX_GRAPH_RELS_PER_CHUNK:
                break

        return rels[:MAX_GRAPH_RELS_PER_CHUNK]

    def _format_supporting_context(
        self,
        neighbor_chunks: List[dict],
        graph_rels: List[GraphRelationship],
    ) -> str:
        """Format neighbor chunks and graph relationships into a string."""
        parts = []

        if neighbor_chunks:
            for nc in neighbor_chunks:
                content = nc.get("content", "")
                if len(content) > 300:
                    content = content[:300] + "…"
                parts.append(f"[Neighbor chunk {nc['chunk_id']}]: {content}")

        if graph_rels:
            rel_strs = []
            for r in graph_rels:
                rel_strs.append(
                    f"{r.source_name} —[{r.relationship_type}]→ {r.target_name}"
                )
            parts.append("Graph relationships: " + "; ".join(rel_strs))

        return "\n".join(parts)

    def expand(
        self,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """Expand each candidate with neighbor chunks and graph context.

        Parameters
        ----------
        candidates:
            Top-ranked candidates from the re-ranker.

        Returns
        -------
        List[RetrievalCandidate]
            Candidates with supporting_context populated.
        """
        logger.info(f"Stage 6: Expanding {len(candidates)} candidates …")

        expanded = []
        for candidate in candidates:
            # Get neighbor chunks
            neighbors = self._get_neighbor_chunks(candidate.chunk_id)

            # Get graph relationships
            graph_rels = self._get_graph_context(candidate.chunk_id)

            # Format supporting context
            supporting = self._format_supporting_context(neighbors, graph_rels)

            updated = candidate.model_copy(
                update={"supporting_context": supporting}
            )
            expanded.append(updated)

        expanded_count = sum(
            1 for c in expanded if c.supporting_context
        )
        logger.info(
            f"  Expanded {expanded_count}/{len(candidates)} candidates ✓"
        )

        return expanded
