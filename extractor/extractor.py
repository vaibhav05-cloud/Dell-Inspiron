"""
Core Entity Extractor.

Processes chunks one-by-one through the LangChain extraction chain,
attaches metadata (chunk_id, page_number, entity_id), deduplicates,
and aggregates into an ExtractionResult.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

from extractor.chains import build_extraction_chain
from extractor.deduplicator import deduplicate
from extractor.schema import (
    EntityType,
    ExtractedEntity,
    ExtractionResult,
    LLMEntityOutput,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts named entities from every chunk in a chunks list.

    Usage
    -----
    >>> extractor = EntityExtractor()
    >>> result = extractor.extract_all(chunks)
    >>> result.entities          # list[ExtractedEntity]
    """

    def __init__(self):
        logger.info("Initialising entity-extraction chain …")
        self._chain = build_extraction_chain()
        logger.info("Extraction chain ready ✓")

    # ── Single chunk ──────────────────────────────────────────────────────

    def extract_chunk(
        self,
        chunk: Dict[str, Any],
    ) -> List[ExtractedEntity]:
        """Run extraction on a single chunk and return entities.

        Parameters
        ----------
        chunk:
            A dict with keys: chunk_id, chunk_type, source_file,
            page_number, section_name, content.

        Returns
        -------
        list[ExtractedEntity]
            De-duplicated entities for this chunk.
        """
        chunk_id    = chunk["chunk_id"]
        page_number = chunk["page_number"]
        content     = chunk.get("content", "").strip()

        if not content:
            logger.warning(f"  ⚠  {chunk_id}: empty content — skipped")
            return []

        # Invoke the chain
        try:
            llm_output: LLMEntityOutput = self._chain.invoke({
                "chunk_id":     chunk_id,
                "chunk_type":   chunk.get("chunk_type", "text"),
                "section_name": chunk.get("section_name", "Unknown"),
                "page_number":  page_number,
                "content":      content,
            })
        except Exception as exc:
            logger.error(f"  ✗  {chunk_id}: chain error — {exc}")
            return []

        if not llm_output or not llm_output.entities:
            return []

        # Convert LLM entities → canonical ExtractedEntity
        entities: List[ExtractedEntity] = []

        for idx, llm_ent in enumerate(llm_output.entities):
            entity = ExtractedEntity(
                entity_id=f"ent_{chunk_id}_{idx}",
                entity_name=llm_ent.entity_name.strip(),
                entity_type=llm_ent.entity_type,
                chunk_id=chunk_id,
                page_number=page_number,
                source_text=llm_ent.source_text.strip()[:120],
                confidence=round(llm_ent.confidence, 3),
            )
            entities.append(entity)

        # Intra-chunk deduplication
        entities = deduplicate(entities)

        return entities

    # ── All chunks ────────────────────────────────────────────────────────

    def extract_all(
        self,
        chunks: List[Dict[str, Any]],
    ) -> ExtractionResult:
        """Process every chunk and return aggregated results.

        Parameters
        ----------
        chunks:
            The full list loaded from ``chunks.json``.

        Returns
        -------
        ExtractionResult
            Aggregated extraction output with stats.
        """
        all_entities: List[ExtractedEntity] = []
        type_counter: Counter = Counter()

        total = len(chunks)
        logger.info(f"Starting extraction for {total} chunks …")

        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get("chunk_id", f"unknown_{i}")
            logger.info(f"  [{i}/{total}]  Processing {chunk_id} …")

            chunk_entities = self.extract_chunk(chunk)

            for ent in chunk_entities:
                type_counter[ent.entity_type.value] += 1

            all_entities.extend(chunk_entities)

            logger.info(
                f"  [{i}/{total}]  {chunk_id} → "
                f"{len(chunk_entities)} entities"
            )

        result = ExtractionResult(
            total_chunks_processed=total,
            total_entities_extracted=len(all_entities),
            entities_by_type=dict(type_counter),
            entities=all_entities,
        )

        logger.info(f"\n{'─' * 50}")
        logger.info(f"  EXTRACTION COMPLETE")
        logger.info(f"{'─' * 50}")
        logger.info(f"  Chunks processed : {result.total_chunks_processed}")
        logger.info(f"  Entities found   : {result.total_entities_extracted}")
        for etype, count in sorted(result.entities_by_type.items()):
            logger.info(f"    {etype:20s} : {count}")
        logger.info(f"{'─' * 50}")

        return result
