"""
Core Relationship Extractor.

Processes chunks one-by-one through the LangChain relationship extraction
chain, resolves entity names to entity_ids, deduplicates, and aggregates
into a RelationshipExtractionResult.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from relationship.chains import build_relationship_chain
from relationship.deduplicator import deduplicate_relationships
from relationship.schema import (
    ExtractedRelationship,
    LLMRelationshipOutput,
    RelationshipExtractionResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """Extracts relationships between co-occurring entities in each chunk.

    Usage
    -----
    >>> extractor = RelationshipExtractor()
    >>> result = extractor.extract_all(chunks, entities)
    >>> result.relationships   # list[ExtractedRelationship]
    """

    def __init__(self):
        logger.info("Initialising relationship-extraction chain …")
        self._chain = build_relationship_chain()
        logger.info("Relationship chain ready ✓")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _group_entities_by_chunk(
        entities: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group entity dicts by their chunk_id."""
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ent in entities:
            grouped[ent["chunk_id"]].append(ent)
        return dict(grouped)

    @staticmethod
    def _build_entity_lookup(
        chunk_entities: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Build a {entity_name_lower → entity_id} lookup for one chunk.

        If two entities share the same lowered name, the first one wins.
        """
        lookup: Dict[str, str] = {}
        for ent in chunk_entities:
            key = ent["entity_name"].lower().strip()
            if key not in lookup:
                lookup[key] = ent["entity_id"]
        return lookup

    @staticmethod
    def _format_entity_list(
        chunk_entities: List[Dict[str, Any]],
    ) -> str:
        """Format entities into a numbered text list for the LLM prompt."""
        if not chunk_entities:
            return "(no entities)"
        lines = []
        for i, ent in enumerate(chunk_entities, 1):
            lines.append(
                f"{i}. {ent['entity_name']} [{ent['entity_type']}]"
            )
        return "\n".join(lines)

    # ── Single chunk ──────────────────────────────────────────────────────

    def extract_chunk(
        self,
        chunk: Dict[str, Any],
        chunk_entities: List[Dict[str, Any]],
    ) -> List[ExtractedRelationship]:
        """Run relationship extraction on a single chunk.

        Parameters
        ----------
        chunk:
            A dict with keys: chunk_id, chunk_type, source_file,
            page_number, section_name, content.
        chunk_entities:
            List of entity dicts already extracted for this chunk.

        Returns
        -------
        list[ExtractedRelationship]
            De-duplicated relationships for this chunk.
        """
        chunk_id    = chunk["chunk_id"]
        page_number = chunk["page_number"]
        content     = chunk.get("content", "").strip()

        if not content:
            logger.warning(f"  ⚠  {chunk_id}: empty content — skipped")
            return []

        if len(chunk_entities) < 2:
            logger.info(
                f"  ⊘  {chunk_id}: <2 entities — no relationships possible"
            )
            return []

        # Build entity name → id lookup
        entity_lookup = self._build_entity_lookup(chunk_entities)
        entity_list_text = self._format_entity_list(chunk_entities)

        # Invoke the chain
        try:
            llm_output: LLMRelationshipOutput = self._chain.invoke({
                "chunk_id":     chunk_id,
                "chunk_type":   chunk.get("chunk_type", "text"),
                "section_name": chunk.get("section_name", "Unknown"),
                "page_number":  page_number,
                "entity_list":  entity_list_text,
                "content":      content,
            })
        except Exception as exc:
            logger.error(f"  ✗  {chunk_id}: chain error — {exc}")
            return []

        if not llm_output or not llm_output.relationships:
            return []

        # Resolve entity names → entity_ids and build canonical relationships
        relationships: List[ExtractedRelationship] = []
        rel_idx = 0

        for llm_rel in llm_output.relationships:
            source_key = llm_rel.source_entity_name.lower().strip()
            target_key = llm_rel.target_entity_name.lower().strip()

            source_id = entity_lookup.get(source_key)
            target_id = entity_lookup.get(target_key)

            # Skip if entity name can't be resolved
            if not source_id:
                logger.warning(
                    f"  ⚠  {chunk_id}: source entity "
                    f"'{llm_rel.source_entity_name}' not found — skipping"
                )
                continue
            if not target_id:
                logger.warning(
                    f"  ⚠  {chunk_id}: target entity "
                    f"'{llm_rel.target_entity_name}' not found — skipping"
                )
                continue

            # Skip self-referencing relationships
            if source_id == target_id:
                logger.warning(
                    f"  ⚠  {chunk_id}: self-reference "
                    f"'{llm_rel.source_entity_name}' → "
                    f"'{llm_rel.target_entity_name}' — skipping"
                )
                continue

            rel = ExtractedRelationship(
                relationship_id=f"rel_{chunk_id}_{rel_idx}",
                source_entity_id=source_id,
                target_entity_id=target_id,
                relationship_type=llm_rel.relationship_type,
                confidence=round(llm_rel.confidence, 3),
                chunk_id=chunk_id,
                page_number=page_number,
            )
            relationships.append(rel)
            rel_idx += 1

        # Intra-chunk deduplication
        relationships = deduplicate_relationships(relationships)

        return relationships

    # ── All chunks ────────────────────────────────────────────────────────

    def extract_all(
        self,
        chunks: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
    ) -> RelationshipExtractionResult:
        """Process every chunk and return aggregated relationship results.

        Parameters
        ----------
        chunks:
            The full list loaded from ``chunks.json``.
        entities:
            The full entity list loaded from ``entities.json``
            (the ``entities`` array within the file).

        Returns
        -------
        RelationshipExtractionResult
            Aggregated extraction output with stats.
        """
        # Group entities by chunk_id for efficient lookup
        entities_by_chunk = self._group_entities_by_chunk(entities)

        all_relationships: List[ExtractedRelationship] = []
        type_counter: Counter = Counter()

        total = len(chunks)
        logger.info(f"Starting relationship extraction for {total} chunks …")

        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get("chunk_id", f"unknown_{i}")
            chunk_entities = entities_by_chunk.get(chunk_id, [])

            logger.info(
                f"  [{i}/{total}]  Processing {chunk_id} "
                f"({len(chunk_entities)} entities) …"
            )

            chunk_rels = self.extract_chunk(chunk, chunk_entities)

            for rel in chunk_rels:
                type_counter[rel.relationship_type.value] += 1

            all_relationships.extend(chunk_rels)

            logger.info(
                f"  [{i}/{total}]  {chunk_id} → "
                f"{len(chunk_rels)} relationships"
            )

        result = RelationshipExtractionResult(
            total_chunks_processed=total,
            total_relationships_extracted=len(all_relationships),
            relationships_by_type=dict(type_counter),
            relationships=all_relationships,
        )

        logger.info(f"\n{'─' * 50}")
        logger.info(f"  RELATIONSHIP EXTRACTION COMPLETE")
        logger.info(f"{'─' * 50}")
        logger.info(
            f"  Chunks processed  : {result.total_chunks_processed}"
        )
        logger.info(
            f"  Relationships found: {result.total_relationships_extracted}"
        )
        for rtype, count in sorted(result.relationships_by_type.items()):
            logger.info(f"    {rtype:20s} : {count}")
        logger.info(f"{'─' * 50}")

        return result
