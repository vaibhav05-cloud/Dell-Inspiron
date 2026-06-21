"""
webapp/pipeline_service.py

Wires your existing pipeline modules into two things:

1. run_ingestion(job_id, pdf_path) -- runs the full real ingestion
   pipeline end-to-end (no stubs):
       parser.pdf_parser.PDFParser
    -> processor.multimodal_processor.MultimodalProcessor
    -> semantic_chunker.SemanticTextChunker
    -> extractor.extractor.EntityExtractor
    -> relationship.extractor.RelationshipExtractor
    -> graph_builder.connection.Neo4jConnection
       + graph_builder.ingestion.GraphIngestionService
    -> retriever.semantic_retriever.SemanticRetrievalAgent (pre-warms the
       FAISS index cache so the first chat query isn't slow)

   Each stage updates webapp.job_store so the frontend can show a live
   stepper + log.

2. get_retrieval_pipeline() / get_synthesis_chain() -- cached singletons
   (these load embedding models / cross-encoders / LLM clients, which is
   too expensive to redo per-request) used by the /api/query route, plus
   invalidate_pipeline_cache() called after a new ingestion so the next
   query picks up the fresh chunks/entities/FAISS index instead of stale
   in-memory state from the previous document.

All imports of your pipeline packages are done lazily, inside functions,
so importing this module doesn't require MISTRAL_API_KEY / Neo4j to be
configured just to start the Flask app (e.g. for /api/health checks).

NOTE on import path: in this repo `semantic_chunker.py` lives at the
PROJECT ROOT (a sibling of parser/, extractor/, etc.), not inside a
`chunker/` package -- so the import below is `from semantic_chunker
import SemanticTextChunker`, not `from chunker.semantic_chunker import
...`. If you later move semantic_chunker.py into a `chunker/` package,
update this one import accordingly.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from webapp import config, job_store

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
#  JSON SAVE HELPERS
#  (mirror extractor/runner.py::save_entities and
#   relationship/runner.py::save_relationships exactly, so the files on
#   disk stay byte-for-byte compatible with your existing CLI tools)
# ─────────────────────────────────────────────────────────────────────────

def _save_entities_json(result, path: Path) -> List[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    entities = [ent.model_dump(mode="json") for ent in result.entities]
    output = {
        "total_chunks_processed": result.total_chunks_processed,
        "total_entities_extracted": result.total_entities_extracted,
        "entities_by_type": result.entities_by_type,
        "entities": entities,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    return entities


def _save_relationships_json(result, path: Path) -> List[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    relationships = [rel.model_dump(mode="json") for rel in result.relationships]
    output = {
        "total_chunks_processed": result.total_chunks_processed,
        "total_relationships_extracted": result.total_relationships_extracted,
        "relationships_by_type": result.relationships_by_type,
        "relationships": relationships,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    return relationships


# ─────────────────────────────────────────────────────────────────────────
#  MAIN INGESTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────

def run_ingestion(job_id: str, pdf_path: Path) -> None:
    """Run the full ingestion pipeline for one PDF. Designed to be called
    inside a background thread (started from the Flask /api/upload route),
    so it's safe for this function to block on slow LLM/embedding calls.

    NOTE: PDFParser.parse(..., clean=True) wipes output/*.json and
    output/images/* before each run. This means ingesting a new PDF
    replaces the previously ingested document's chunks / entities /
    relationships / FAISS index -- this app is single-corpus-at-a-time,
    matching the existing pipeline's behavior.
    """
    try:
        # ── Stage 1: Parsing ────────────────────────────────────────────
        job_store.set_stage(job_id, "parsing", progress=5)
        from parser.pdf_parser import PDFParser

        pdf_parser = PDFParser(output_dir=str(config.OUTPUT_DIR))
        parsed = pdf_parser.parse(str(pdf_path), clean=True)
        parsed_json_path = pdf_parser.save_to_json(parsed)
        job_store.log(
            job_id,
            f"Parsed {parsed.total_pages} pages — "
            f"{len(parsed.texts)} text blocks, {len(parsed.tables)} tables, "
            f"{len(parsed.images)} images.",
        )

        # ── Stage 2: Multimodal enrichment (Mistral vision) ─────────────
        job_store.set_stage(job_id, "multimodal_enrichment", progress=20)
        from processor.multimodal_processor import MultimodalProcessor

        mm_processor = MultimodalProcessor()
        processed_json_path = mm_processor.process_file(parsed_json_path)
        job_store.log(job_id, f"Enriched images/tables -> {processed_json_path}")

        # ── Stage 3: Semantic chunking ───────────────────────────────────
        job_store.set_stage(job_id, "chunking", progress=40)
        from semantic_chunker import SemanticTextChunker

        text_chunker = SemanticTextChunker()
        data = text_chunker.load_processed_json(processed_json_path)

        text_chunks, next_id = text_chunker.build_text_chunks(data)
        table_chunks, next_id = text_chunker.build_table_chunks(data, next_id)
        image_chunks, next_id = text_chunker.build_image_chunks(data, next_id)
        all_chunks = text_chunks + table_chunks + image_chunks

        text_chunker.save_chunks(all_chunks, str(config.CHUNKS_PATH))
        job_store.log(
            job_id,
            f"Built {len(all_chunks)} chunks "
            f"({len(text_chunks)} text, {len(table_chunks)} table, "
            f"{len(image_chunks)} image).",
        )

        if not all_chunks:
            raise RuntimeError(
                "No chunks were produced from this PDF -- nothing to "
                "extract or ingest. Check that the PDF has extractable "
                "text (not a pure scanned image with no OCR layer)."
            )

        # ── Stage 4: Entity extraction ───────────────────────────────────
        job_store.set_stage(job_id, "entity_extraction", progress=55)
        from extractor.extractor import EntityExtractor

        entity_extractor = EntityExtractor()
        entity_result = entity_extractor.extract_all(all_chunks)
        entities = _save_entities_json(entity_result, config.ENTITIES_PATH)
        job_store.log(
            job_id,
            f"Extracted {entity_result.total_entities_extracted} entities "
            f"across {len(entity_result.entities_by_type)} types.",
        )

        # ── Stage 5: Relationship extraction ─────────────────────────────
        job_store.set_stage(job_id, "relationship_extraction", progress=70)
        from relationship.extractor import RelationshipExtractor

        rel_extractor = RelationshipExtractor()
        rel_result = rel_extractor.extract_all(all_chunks, entities)
        relationships = _save_relationships_json(rel_result, config.RELATIONSHIPS_PATH)
        job_store.log(
            job_id,
            f"Extracted {rel_result.total_relationships_extracted} relationships "
            f"across {len(rel_result.relationships_by_type)} types.",
        )

        # ── Stage 6: Neo4j graph ingestion ───────────────────────────────
        job_store.set_stage(job_id, "graph_ingestion", progress=85)
        from graph_builder.connection import Neo4jConnection
        from graph_builder.ingestion import GraphIngestionService

        with Neo4jConnection() as conn:
            ingestion_service = GraphIngestionService(conn)
            stats = ingestion_service.ingest_all(
                entities,
                relationships,
                batch_size=config.GRAPH_INGEST_BATCH_SIZE,
            )
        job_store.log(
            job_id,
            f"Neo4j ingestion complete -- "
            f"{stats.get('entity_nodes', 0)} entity nodes, "
            f"{stats.get('chunk_nodes', 0)} chunk nodes, "
            f"{stats.get('total_relationships', 0)} relationship edges.",
        )

        # ── Stage 7: Pre-warm FAISS index ────────────────────────────────
        job_store.set_stage(job_id, "indexing", progress=95)
        from retriever.semantic_retriever import SemanticRetrievalAgent

        semantic_agent = SemanticRetrievalAgent(
            chunks_path=str(config.CHUNKS_PATH),
            index_dir=str(config.FAISS_INDEX_DIR),
        )
        # Calling .retrieve() forces the lazy index build + on-disk cache;
        # we don't care about the (trivial) results, only the side effect.
        semantic_agent.retrieve(query="index warm-up", top_k=1)
        job_store.log(job_id, "FAISS index built and cached.")

        # Invalidate any cached RetrievalPipeline/SynthesisChain so the
        # next /api/query call picks up this document's fresh chunks/
        # entities/index instead of stale in-memory state from a previous
        # upload.
        invalidate_pipeline_cache()

        job_store.update(
            job_id,
            status="complete",
            stage="done",
            stage_label="Complete",
            progress=100,
        )
        job_store.log(job_id, "Ingestion pipeline complete.")

    except Exception as exc:  # noqa: BLE001 -- surface any stage failure to the UI
        logger.exception("Ingestion pipeline failed for job %s", job_id)
        job_store.update(job_id, status="error", error=str(exc))
        job_store.log(job_id, f"ERROR: {exc}")


# ─────────────────────────────────────────────────────────────────────────
#  CACHED SINGLETONS FOR /api/query
#  (embedding models, cross-encoder, LLM clients are expensive to load --
#   build once, reuse across requests, invalidate after re-ingestion)
# ─────────────────────────────────────────────────────────────────────────

_singleton_lock = threading.Lock()
_cached_pipeline = None  # type: Optional[Any]  -- retriever.pipeline.RetrievalPipeline
_cached_synth_chain = None  # type: Optional[Any]  -- synthesizer.synthesis_chain.AnswerSynthesisChain


def get_retrieval_pipeline():
    global _cached_pipeline
    with _singleton_lock:
        if _cached_pipeline is None:
            from retriever.pipeline import RetrievalPipeline

            _cached_pipeline = RetrievalPipeline(
                chunks_path=str(config.CHUNKS_PATH),
                entities_path=str(config.ENTITIES_PATH),
                index_dir=str(config.FAISS_INDEX_DIR),
                semantic_top_k=config.SEMANTIC_TOP_K,
                rerank_top_k=config.RERANK_TOP_K,
                token_budget=config.TOKEN_BUDGET,
            )
        return _cached_pipeline


def get_synthesis_chain():
    global _cached_synth_chain
    with _singleton_lock:
        if _cached_synth_chain is None:
            from synthesizer.synthesis_chain import AnswerSynthesisChain

            _cached_synth_chain = AnswerSynthesisChain()
        return _cached_synth_chain


def invalidate_pipeline_cache() -> None:
    """Drop the cached RetrievalPipeline (closing its Neo4j connection)
    so the next get_retrieval_pipeline() call rebuilds against the
    just-ingested document. The synthesis chain holds no per-corpus
    state, so it doesn't need resetting.
    """
    global _cached_pipeline
    with _singleton_lock:
        if _cached_pipeline is not None:
            try:
                _cached_pipeline.close()
            except Exception:  # noqa: BLE001
                logger.warning("Error closing previous RetrievalPipeline", exc_info=True)
            _cached_pipeline = None
