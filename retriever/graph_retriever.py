"""
Stage 3 — Graph Retrieval Agent.

Uses Neo4j to retrieve relevant entities, relevant relationships, and relevant chunks
with bounded traversal, priority expansion, relationship confidence filtering, and evidence mapping.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from retriever.schema import (
    CandidateSource,
    GraphCandidate,
    GraphEntity,
    GraphRelationship,
    RetrievalCandidate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  CYPHER TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

CYPHER_ENTITY_LOOKUP = """
MATCH (e:Entity)
WHERE toLower(e.entity_name) CONTAINS toLower($name)
RETURN e.entity_id   AS entity_id,
       e.entity_name AS entity_name,
       e.entity_type AS entity_type,
       CASE WHEN toLower(e.entity_name) = toLower($name)
            THEN 1.0 ELSE 0.7 END AS match_quality
ORDER BY match_quality DESC, e.entity_name
LIMIT 10
"""

CYPHER_1HOP_FILTERED = """
MATCH (e:Entity {entity_id: $entity_id})-[r]-(neighbor:Entity)
WHERE NOT type(r) = 'APPEARS_IN'
  AND coalesce(r.confidence, 0.0) >= $min_confidence
RETURN neighbor.entity_id   AS entity_id,
       neighbor.entity_name AS entity_name,
       neighbor.entity_type AS entity_type,
       type(r)              AS rel_type,
       r.confidence         AS confidence,
       r.chunk_id           AS rel_chunk_id,
       r.page_number        AS rel_page_number,
       CASE WHEN startNode(r) = e THEN 'OUTGOING' ELSE 'INCOMING' END AS direction
ORDER BY r.confidence DESC
LIMIT $max_neighbors
"""

CYPHER_2HOP_FILTERED = """
MATCH (e:Entity {entity_id: $entity_id})-[r1]-(n1:Entity)-[r2]-(n2:Entity)
WHERE NOT type(r1) = 'APPEARS_IN'
  AND NOT type(r2) = 'APPEARS_IN'
  AND coalesce(r1.confidence, 0.0) >= $min_confidence
  AND coalesce(r2.confidence, 0.0) >= $min_confidence
  AND n2.entity_id <> e.entity_id
  AND n2.entity_id <> n1.entity_id
RETURN DISTINCT
       n2.entity_id   AS entity_id,
       n2.entity_name AS entity_name,
       n2.entity_type AS entity_type,
       type(r1)       AS rel_type_1,
       type(r2)       AS rel_type_2,
       r1.confidence  AS confidence_1,
       r2.confidence  AS confidence_2,
       n1.entity_name AS via_entity,
       n1.entity_id   AS via_entity_id
ORDER BY (coalesce(r1.confidence, 0) + coalesce(r2.confidence, 0)) / 2.0 DESC
LIMIT $max_neighbors
"""

CYPHER_ENTITY_CHUNKS_EVIDENCE = """
MATCH (e:Entity)-[a:APPEARS_IN]->(c:Chunk)
WHERE e.entity_id IN $entity_ids
RETURN DISTINCT
       c.chunk_id           AS chunk_id,
       c.page_number        AS page_number,
       a.confidence         AS appears_confidence,
       e.entity_id          AS entity_id,
       e.entity_name        AS entity_name,
       e.entity_type        AS entity_type
ORDER BY a.confidence DESC
LIMIT $max_chunks
"""

CYPHER_ENTITY_DEGREE = """
MATCH (e:Entity {entity_id: $entity_id})-[r]-()
WHERE NOT type(r) = 'APPEARS_IN'
RETURN count(r) AS degree
"""

CYPHER_ENTITY_RELATIONSHIPS = """
MATCH (e:Entity {entity_id: $entity_id})-[r]-(other:Entity)
WHERE NOT type(r) = 'APPEARS_IN'
  AND coalesce(r.confidence, 0.0) >= $min_confidence
RETURN e.entity_name     AS source_name,
       other.entity_name AS target_name,
       type(r)           AS rel_type,
       r.confidence      AS confidence,
       r.chunk_id        AS chunk_id,
       r.page_number     AS page_number
ORDER BY r.confidence DESC
LIMIT $max_rels
"""


# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH RETRIEVAL AGENT
# ─────────────────────────────────────────────────────────────────────────────

class GraphRetrievalAgent:
    """Neo4j graph-based retrieval agent with bounded traversal and scoring."""

    MAX_NEIGHBORS_1HOP: int = 20
    MAX_NEIGHBORS_2HOP: int = 10
    MAX_CHUNKS: int = 30
    MAX_RELATIONSHIPS: int = 10
    MIN_CONFIDENCE: float = 0.70

    SCORE_DIRECT_MATCH: float = 1.0
    SCORE_1HOP: float = 0.7
    SCORE_2HOP: float = 0.4

    def __init__(
        self,
        chunks_path: str = "output/chunks.json",
        min_confidence: float = 0.70,
    ):
        self._conn = None
        self._chunks_lookup: Dict[str, dict] = {}
        self._chunks_path = chunks_path
        self.MIN_CONFIDENCE = min_confidence

    def _ensure_connected(self) -> None:
        """Lazy-connect to Neo4j."""
        if self._conn is not None:
            return

        from graph_builder.connection import Neo4jConnection

        self._conn = Neo4jConnection()
        self._conn.connect()

    def _load_chunks_lookup(self) -> None:
        """Load chunks.json for content lookup."""
        if self._chunks_lookup:
            return

        p = Path(self._chunks_path)
        if not p.exists():
            return

        with open(p, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        self._chunks_lookup = {c["chunk_id"]: c for c in chunks}

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _compute_entity_priority(
        self,
        entity_id: str,
        match_quality: float,
        session,
    ) -> float:
        """Compute priority score for an entity expansion order."""
        import math

        try:
            result = session.run(
                CYPHER_ENTITY_DEGREE, entity_id=entity_id
            ).single()
            degree = result["degree"] if result else 0
        except Exception:
            degree = 0

        frequency_factor = math.log2(degree + 1) if degree > 0 else 0.1
        frequency_factor = min(frequency_factor / 3.5, 1.0)

        return match_quality * 0.6 + frequency_factor * 0.4

    def retrieve(
        self,
        query_entities: List[str],
        traversal_depth: int = 2,
    ) -> Tuple[List[RetrievalCandidate], GraphCandidate]:
        """Perform entity-first graph retrieval returning relevant chunks and metadata.

        Returns
        -------
        Tuple[List[RetrievalCandidate], GraphCandidate]
            Relevant chunk candidates, plus a GraphCandidate containing relevant entities and relationships.
        """
        self._ensure_connected()
        self._load_chunks_lookup()

        traversal_depth = min(traversal_depth, 2)

        logger.info(
            f"  Graph Retrieval Agent: running for {len(query_entities)} entities "
            f"(depth={traversal_depth}, min_conf={self.MIN_CONFIDENCE}) …"
        )

        graph_candidate = GraphCandidate()

        entity_scores: Dict[str, float] = {}
        entity_rel_confidences: Dict[str, List[float]] = defaultdict(list)
        chunk_entity_map: Dict[str, Set[str]] = defaultdict(set)
        chunk_evidence: Dict[str, List[dict]] = defaultdict(list)

        seen_entity_ids: Set[str] = set()

        with self._conn.session() as session:
            matched_ids_with_priority: List[Tuple[str, float]] = []

            for name in query_entities:
                records = session.run(
                    CYPHER_ENTITY_LOOKUP, name=name
                ).data()

                for rec in records:
                    eid = rec["entity_id"]
                    match_quality = rec.get("match_quality", 0.7)

                    entity = GraphEntity(
                        entity_id=eid,
                        entity_name=rec["entity_name"],
                        entity_type=rec["entity_type"],
                    )
                    graph_candidate.matched_entities.append(entity)
                    seen_entity_ids.add(eid)

                    entity_scores[eid] = self.SCORE_DIRECT_MATCH * match_quality

                    priority = self._compute_entity_priority(
                        eid, match_quality, session
                    )
                    matched_ids_with_priority.append((eid, priority))

            logger.info(
                f"    Matched {len(graph_candidate.matched_entities)} entities"
            )

            if not matched_ids_with_priority:
                return [], graph_candidate

            matched_ids_with_priority.sort(key=lambda x: x[1], reverse=True)

            hop1_entities: List[Tuple[str, float]] = []

            for eid, priority in matched_ids_with_priority:
                records = session.run(
                    CYPHER_1HOP_FILTERED,
                    entity_id=eid,
                    min_confidence=self.MIN_CONFIDENCE,
                    max_neighbors=self.MAX_NEIGHBORS_1HOP,
                ).data()

                for rec in records:
                    nid = rec["entity_id"]
                    conf = rec.get("confidence", 0.0) or 0.0

                    neighbor = GraphEntity(
                        entity_id=nid,
                        entity_name=rec["entity_name"],
                        entity_type=rec["entity_type"],
                    )

                    if nid not in seen_entity_ids:
                        graph_candidate.connected_entities.append(neighbor)
                        seen_entity_ids.add(nid)

                    hop1_score = self.SCORE_1HOP * conf
                    entity_scores[nid] = max(
                        entity_scores.get(nid, 0.0), hop1_score
                    )

                    entity_rel_confidences[eid].append(conf)
                    entity_rel_confidences[nid].append(conf)

                    rel = GraphRelationship(
                        source_name=rec["source_name"] if "source_name" in rec else rec["entity_name"], # Note: startNode determines direction, let's keep name lookup consistent
                        target_name=rec["target_name"] if "target_name" in rec else rec["entity_name"],
                        relationship_type=rec["rel_type"],
                        confidence=conf,
                    )
                    # Let's fix names using CYPHER_ENTITY_RELATIONSHIPS pattern if possible, or just set it:
                    # Actually, let's do a correct source/target name assignment:
                    # If direction is OUTGOING, eid is source, nid is target
                    # If direction is INCOMING, nid is source, eid is target
                    # Let's query entity names:
                    # We can fetch neighbor name and self name.
                    # e's name is from eid name lookup. Let's trace back from matched_entities:
                    e_name = next((x.entity_name for x in graph_candidate.matched_entities if x.entity_id == eid), eid)
                    n_name = rec["entity_name"]
                    if rec.get("direction") == "OUTGOING":
                        rel.source_name = e_name
                        rel.target_name = n_name
                    else:
                        rel.source_name = n_name
                        rel.target_name = e_name

                    graph_candidate.relationships.append(rel)
                    hop1_entities.append((nid, conf))

            if traversal_depth >= 2:
                hop2_count = 0
                for eid, priority in matched_ids_with_priority:
                    records = session.run(
                        CYPHER_2HOP_FILTERED,
                        entity_id=eid,
                        min_confidence=self.MIN_CONFIDENCE,
                        max_neighbors=self.MAX_NEIGHBORS_2HOP,
                    ).data()

                    for rec in records:
                        nid = rec["entity_id"]
                        conf1 = rec.get("confidence_1", 0.0) or 0.0
                        conf2 = rec.get("confidence_2", 0.0) or 0.0
                        avg_conf = (conf1 + conf2) / 2.0

                        neighbor2 = GraphEntity(
                            entity_id=nid,
                            entity_name=rec["entity_name"],
                            entity_type=rec["entity_type"],
                        )

                        if nid not in seen_entity_ids:
                            graph_candidate.connected_entities.append(neighbor2)
                            seen_entity_ids.add(nid)
                            hop2_count += 1

                        hop2_score = self.SCORE_2HOP * avg_conf
                        entity_scores[nid] = max(
                            entity_scores.get(nid, 0.0), hop2_score
                        )

                        entity_rel_confidences[nid].append(avg_conf)

                        # Create the 2-hop relationship
                        rel = GraphRelationship(
                            source_name=rec["via_entity"],
                            target_name=rec["entity_name"],
                            relationship_type=rec["rel_type_2"],
                            confidence=conf2,
                        )
                        graph_candidate.relationships.append(rel)

            all_relevant_ids = list(seen_entity_ids)

            records = session.run(
                CYPHER_ENTITY_CHUNKS_EVIDENCE,
                entity_ids=all_relevant_ids,
                max_chunks=self.MAX_CHUNKS,
            ).data()

            for rec in records:
                cid = rec["chunk_id"]
                eid = rec["entity_id"]
                chunk_entity_map[cid].add(eid)
                chunk_evidence[cid].append({
                    "entity_id": eid,
                    "entity_name": rec["entity_name"],
                    "entity_type": rec["entity_type"],
                    "page_number": rec["page_number"],
                    "appears_confidence": rec.get("appears_confidence", 0.0) or 0.0,
                })

            graph_candidate.chunk_ids = list(chunk_entity_map.keys())

        candidates = []
        for cid, entity_ids in chunk_entity_map.items():
            chunk_data = self._chunks_lookup.get(cid)
            if not chunk_data:
                continue

            graph_relevance = max(
                (entity_scores.get(eid, 0.0) for eid in entity_ids),
                default=0.0,
            )

            all_confs = []
            for eid in entity_ids:
                all_confs.extend(entity_rel_confidences.get(eid, []))
            avg_rel_confidence = (
                sum(all_confs) / len(all_confs) if all_confs else 0.0
            )

            evidence = chunk_evidence.get(cid, [])
            page_number = (
                evidence[0]["page_number"] if evidence else
                chunk_data.get("page_number", 0)
            )

            candidates.append(
                RetrievalCandidate(
                    chunk_id=cid,
                    content=chunk_data.get("content", ""),
                    page_number=page_number,
                    section_name=chunk_data.get("section_name", ""),
                    source_file=chunk_data.get("source_file", ""),
                    source=CandidateSource.GRAPH,
                    similarity_score=0.0,
                    graph_relevance_score=round(graph_relevance, 4),
                    relationship_confidence=round(avg_rel_confidence, 4),
                )
            )

        candidates.sort(
            key=lambda c: c.graph_relevance_score, reverse=True
        )

        return candidates, graph_candidate

    def get_entity_relationships(
        self,
        entity_id: str,
        max_rels: int = 10,
    ) -> List[GraphRelationship]:
        """Fetch relationships for context expansion agent."""
        self._ensure_connected()

        with self._conn.session() as session:
            records = session.run(
                CYPHER_ENTITY_RELATIONSHIPS,
                entity_id=entity_id,
                min_confidence=self.MIN_CONFIDENCE,
                max_rels=max_rels,
            ).data()

        rels = []
        for rec in records:
            rels.append(
                GraphRelationship(
                    source_name=rec["source_name"],
                    target_name=rec["target_name"],
                    relationship_type=rec["rel_type"],
                    confidence=rec.get("confidence", 0.0) or 0.0,
                )
            )

        return rels
