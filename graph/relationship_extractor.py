"""
graph/relationship_extractor.py

SPEED FIX: Uses classify_batch() instead of individual classify() calls.
  Before: 45 individual Mistral calls per chunk  → ~15 min total
  After:  7 batch calls per chunk (7 pairs each) → ~2-3 min total
          Same relationship quality, ~6x faster.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

from graph.relation_classifier import (
    BATCH_SIZE,
    RelationClassifier,
    generate_entity_pairs,
)

logger = logging.getLogger(__name__)

MAX_ENTITIES_PER_CHUNK = 10   # C(10,2)=45 pairs = matches MAX_PAIRS_PER_CHUNK

BAD_TERMS = {
    "NOTICE", "CAUTION", "NOTE", "TIP", "WARNING",
    "Button", "Drive", "Section", "Chapter",
    "one", "1-Year", "10,000 ft",
}


class RelationshipExtractor:

    def __init__(self, api_key: str):
        self.classifier = RelationClassifier(api_key)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _group_entities_by_chunk(self, entities: list) -> dict:
        grouped = defaultdict(list)
        for entity in entities:
            chunk_id = entity.get("chunk_id")
            name     = entity.get("name")
            if chunk_id and name:
                grouped[chunk_id].append(name)
        return grouped

    def _get_chunk_content(self, chunk_id: str, chunks: list) -> str:
        for chunk in chunks:
            if chunk.get("chunk_id") == chunk_id:
                return chunk.get("content", "")
        return ""

    def _is_valid_entity(self, entity: str) -> bool:
        entity = str(entity).strip()
        return len(entity) >= 3 and not entity.isdigit()

    # ── Per-chunk extraction ──────────────────────────────────────────────────

    def _extract_chunk_relationships(
        self,
        chunk_id:     str,
        entity_names: list,
        content:      str,
    ) -> list:
        relationships = []

        # Deduplicate + filter bad terms
        unique = list(dict.fromkeys(entity_names))
        filtered = [
            e for e in unique
            if self._is_valid_entity(e) and e not in BAD_TERMS
        ][:MAX_ENTITIES_PER_CHUNK]

        pairs = generate_entity_pairs(filtered)

        if not pairs:
            return relationships

        # ── BATCH: process BATCH_SIZE pairs per Mistral call ─────────────────
        total_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(
            f"      {len(filtered)} entities → {len(pairs)} pairs → "
            f"{total_batches} batch call(s)"
        )

        for batch_start in range(0, len(pairs), BATCH_SIZE):
            batch   = pairs[batch_start : batch_start + BATCH_SIZE]
            relations = self.classifier.classify_batch(batch, content)

            for (e1, e2), relation in zip(batch, relations):
                if relation == "NONE":
                    continue
                if e1.lower() == e2.lower():
                    continue

                relationships.append({
                    "source":   e1,
                    "relation": relation,
                    "target":   e2,
                    "chunk_id": chunk_id,
                })

        return relationships

    # ── Main entry point ──────────────────────────────────────────────────────

    def extract_all(self, entities: list, chunks: list) -> list:
        all_relationships  = []
        chunk_entity_map   = self._group_entities_by_chunk(entities)
        total              = len(chunk_entity_map)

        for idx, (chunk_id, entity_names) in enumerate(
            chunk_entity_map.items(), start=1
        ):
            logger.info(
                f"[{idx}/{total}] {chunk_id} "
                f"({len(entity_names)} entities)"
            )

            if len(entity_names) < 2:
                continue

            content = self._get_chunk_content(chunk_id, chunks)
            rels    = self._extract_chunk_relationships(
                chunk_id, entity_names, content
            )
            all_relationships.extend(rels)
            logger.info(f"    → {len(rels)} Mistral relations found")

        logger.info(
            f"Total Mistral relationships: {len(all_relationships)}"
        )
        return all_relationships