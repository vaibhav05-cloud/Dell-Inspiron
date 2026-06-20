"""
Stage 6 — Evidence Agent.

Builds structured evidence from the re-ranked candidates and graph metadata,
including source chunks, neighbor chunks, source entities, and graph paths.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from retriever.schema import GraphCandidate, GraphRelationship, RetrievalCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class EvidenceAgent:
    """Compiles chunks, entities, and graph paths into a structured evidence package."""

    def __init__(self, chunks_lookup: Dict[str, dict]):
        self._chunks_lookup = chunks_lookup
        # Sort key to find chunk order for neighbor lookup
        self._ordered_ids = sorted(
            chunks_lookup.keys(),
            key=lambda x: self._chunk_sort_key(x),
        )
        self._id_to_index = {
            cid: i for i, cid in enumerate(self._ordered_ids)
        }

    @staticmethod
    def _chunk_sort_key(chunk_id: str) -> int:
        """Extract numeric index from chunk_id for ordering."""
        try:
            return int(chunk_id.replace("chunk_", ""))
        except (ValueError, AttributeError):
            return 0

    def _get_neighbor_chunks(self, chunk_id: str, count: int = 1) -> List[dict]:
        """Fetch neighbor chunks (O(1) lookup)."""
        idx = self._id_to_index.get(chunk_id)
        if idx is None:
            return []

        neighbors = []
        if idx > 0:
            prev_id = self._ordered_ids[idx - 1]
            prev_data = self._chunks_lookup.get(prev_id)
            if prev_data:
                neighbors.append(prev_data)

        if idx < len(self._ordered_ids) - 1:
            next_id = self._ordered_ids[idx + 1]
            next_data = self._chunks_lookup.get(next_id)
            if next_data:
                neighbors.append(next_data)

        return neighbors[:count]

    def build_evidence(
        self,
        candidates: List[RetrievalCandidate],
        graph_metadata: Optional[GraphCandidate] = None,
    ) -> dict:
        """Compile final evidence containing chunks, entities, and graph paths.

        Parameters
        ----------
        candidates:
            Top re-ranked candidates.
        graph_metadata:
            Graph Candidate from Neo4j traversal.

        Returns
        -------
        dict
            Dictionay containing:
            - 'evidence_chunks': List of dicts representing the chunks (with neighbor context)
            - 'graph_paths': List of traversed path dicts
            - 'source_entities': List of unique entities matched/connected
        """
        logger.info("Stage 6: Running Evidence Agent …")

        evidence_chunks = []
        seen_chunks = set()

        for c in candidates:
            if c.chunk_id in seen_chunks:
                continue

            # Core chunk
            chunk_entry = {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "page_number": c.page_number,
                "section_name": c.section_name,
                "source_document": c.source_file,
                "score": round(c.rerank_score, 4),
                "source": c.source.value,
                "is_neighbor": False,
            }
            evidence_chunks.append(chunk_entry)
            seen_chunks.add(c.chunk_id)

            # Retrieve neighbors for context expansion
            neighbors = self._get_neighbor_chunks(c.chunk_id, count=1)
            for n in neighbors:
                nid = n["chunk_id"]
                if nid not in seen_chunks:
                    evidence_chunks.append({
                        "chunk_id": nid,
                        "content": n["content"],
                        "page_number": n.get("page_number", 0),
                        "section_name": n.get("section_name", ""),
                        "source_document": n.get("source_file", ""),
                        "score": round(c.rerank_score - 0.1, 4), # slightly lower score
                        "source": c.source.value,
                        "is_neighbor": True,
                    })
                    seen_chunks.add(nid)

        # Source entities from graph
        source_entities = []
        seen_entities = set()

        if graph_metadata:
            # Add matched entities
            for ent in graph_metadata.matched_entities:
                if ent.entity_id not in seen_entities:
                    source_entities.append({
                        "entity_id": ent.entity_id,
                        "entity_name": ent.entity_name,
                        "entity_type": ent.entity_type,
                        "relationship_role": "matched",
                    })
                    seen_entities.add(ent.entity_id)

            # Add connected entities
            for ent in graph_metadata.connected_entities:
                if ent.entity_id not in seen_entities:
                    source_entities.append({
                        "entity_id": ent.entity_id,
                        "entity_name": ent.entity_name,
                        "entity_type": ent.entity_type,
                        "relationship_role": "connected",
                    })
                    seen_entities.add(ent.entity_id)

        # Graph paths/relationships
        graph_paths = []
        seen_paths = set()

        if graph_metadata and graph_metadata.relationships:
            for rel in graph_metadata.relationships:
                path_key = (rel.source_name, rel.relationship_type, rel.target_name)
                if path_key not in seen_paths:
                    graph_paths.append({
                        "source": rel.source_name,
                        "relationship": rel.relationship_type,
                        "target": rel.target_name,
                        "confidence": round(rel.confidence, 4),
                    })
                    seen_paths.add(path_key)

        logger.info(
            f"  Evidence Agent: Compiled {len(evidence_chunks)} chunks, "
            f"{len(source_entities)} entities, {len(graph_paths)} graph paths."
        )

        return {
            "evidence_chunks": evidence_chunks,
            "source_entities": source_entities,
            "graph_paths": graph_paths,
        }
