"""
End-to-end GraphRAG Agentic Retrieval Pipeline.

Orchestrates all 7 agents sequentially with timing instrumentation,
conforming to the lightweight agentic design and producing structured JSON context output.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

from retriever.schema import (
    RetrievalResult,
    StageTiming,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """7-agent lightweight retrieval workflow orchestrator.

    All components are lazily initialized on first use.
    """

    def __init__(
        self,
        chunks_path: str = "output/chunks.json",
        entities_path: str = "output/entities.json",
        index_dir: str = "output/faiss_index",
        semantic_top_k: int = 30,
        rerank_top_k: int = 15,
        token_budget: int = 4000,
    ):
        self._chunks_path = chunks_path
        self._entities_path = entities_path
        self._index_dir = index_dir
        self._semantic_top_k = semantic_top_k
        self._rerank_top_k = rerank_top_k
        self._token_budget = token_budget

        # Lazy-loaded agents
        self._query_planner = None
        self._semantic_retriever = None
        self._graph_retriever = None
        self._fusion_agent = None
        self._reranker = None
        self._evidence_agent = None
        self._context_builder = None

        # Data caches
        self._chunks_lookup: Optional[Dict[str, dict]] = None
        self._entities_lookup: Optional[Dict[str, dict]] = None

    # ── Lazy initializers ─────────────────────────────────────────────────

    def _get_query_planner(self):
        if self._query_planner is None:
            from retriever.query_understanding import QueryPlannerAgent
            self._query_planner = QueryPlannerAgent()
        return self._query_planner

    def _get_semantic_retriever(self):
        if self._semantic_retriever is None:
            from retriever.semantic_retriever import SemanticRetrievalAgent
            self._semantic_retriever = SemanticRetrievalAgent(
                chunks_path=self._chunks_path,
                index_dir=self._index_dir,
            )
        return self._semantic_retriever

    def _get_graph_retriever(self):
        if self._graph_retriever is None:
            from retriever.graph_retriever import GraphRetrievalAgent
            self._graph_retriever = GraphRetrievalAgent(
                chunks_path=self._chunks_path,
            )
        return self._graph_retriever

    def _get_fusion_agent(self):
        if self._fusion_agent is None:
            from retriever.orchestrator import FusionAgent
            self._fusion_agent = FusionAgent()
        return self._fusion_agent

    def _get_reranker(self):
        if self._reranker is None:
            from retriever.reranker import RerankingAgent
            self._reranker = RerankingAgent()
        return self._reranker

    def _get_evidence_agent(self):
        if self._evidence_agent is None:
            from retriever.evidence_agent import EvidenceAgent
            self._evidence_agent = EvidenceAgent(
                chunks_lookup=self._get_chunks_lookup(),
            )
        return self._evidence_agent

    def _get_context_builder(self):
        if self._context_builder is None:
            from retriever.context_builder import ContextBuilderAgent
            self._context_builder = ContextBuilderAgent(
                token_budget=self._token_budget,
            )
        return self._context_builder

    def _get_chunks_lookup(self) -> Dict[str, dict]:
        if self._chunks_lookup is None:
            p = Path(self._chunks_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                self._chunks_lookup = {c["chunk_id"]: c for c in chunks}
            else:
                self._chunks_lookup = {}
        return self._chunks_lookup

    def _get_entities_lookup(self) -> Dict[str, dict]:
        if self._entities_lookup is None:
            p = Path(self._entities_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entities = data.get("entities", data) if isinstance(data, dict) else data
                self._entities_lookup = {
                    e["entity_id"]: e for e in entities
                }
            else:
                self._entities_lookup = {}
        return self._entities_lookup

    # ── Main retrieve method ──────────────────────────────────────────────

    def retrieve(self, query: str) -> RetrievalResult:
        """Execute the full 7-agent agentic retrieval workflow sequentially.

        Parameters
        ----------
        query:
            The user's natural-language question.

        Returns
        -------
        RetrievalResult
            Exact structured output representation.
        """
        pipeline_start = time.perf_counter()
        timings = []

        logger.info("")
        logger.info("=" * 50)
        logger.info("  AGENTIC RETRIEVAL PIPELINE")
        logger.info("=" * 50)
        logger.info(f"  Query: {query}")
        logger.info("=" * 50)

        # ── Stage 1: Query Planner Agent ──────────────────────────────────
        t0 = time.perf_counter()
        query_analysis = self._get_query_planner().plan(query)
        t1 = time.perf_counter()
        timings.append(StageTiming(
            stage_name="Query Planner Agent",
            duration_ms=(t1 - t0) * 1000,
        ))

        plan = query_analysis.retrieval_plan

        # ── Stage 2: Semantic Retrieval Agent ─────────────────────────────
        t0 = time.perf_counter()
        semantic_candidates = []
        if plan.semantic_search_needed:
            semantic_candidates = self._get_semantic_retriever().retrieve(
                query=query,
                top_k=self._semantic_top_k,
            )
        t1 = time.perf_counter()
        timings.append(StageTiming(
            stage_name="Semantic Retrieval Agent",
            duration_ms=(t1 - t0) * 1000,
        ))

        # ── Stage 3: Graph Retrieval Agent ────────────────────────────────
        t0 = time.perf_counter()
        graph_candidates = []
        graph_metadata = None

        if plan.graph_search_needed and query_analysis.query_entities:
            graph_candidates, graph_metadata = self._get_graph_retriever().retrieve(
                query_entities=query_analysis.query_entities,
                traversal_depth=plan.traversal_depth,
            )
        t1 = time.perf_counter()
        timings.append(StageTiming(
            stage_name="Graph Retrieval Agent",
            duration_ms=(t1 - t0) * 1000,
        ))

        # ── Stage 4: Fusion Agent ─────────────────────────────────────────
        t0 = time.perf_counter()
        fused = self._get_fusion_agent().fuse(
            semantic_candidates, graph_candidates
        )
        t1 = time.perf_counter()
        timings.append(StageTiming(
            stage_name="Fusion Agent",
            duration_ms=(t1 - t0) * 1000,
        ))

        # ── Stage 5: Re-ranking Agent ─────────────────────────────────────
        t0 = time.perf_counter()
        reranked = self._get_reranker().rerank(
            query=query,
            candidates=fused,
            top_k=self._rerank_top_k,
        )
        t1 = time.perf_counter()
        timings.append(StageTiming(
            stage_name="Re-ranking Agent",
            duration_ms=(t1 - t0) * 1000,
        ))

        # ── Stage 6: Evidence Agent ───────────────────────────────────────
        t0 = time.perf_counter()
        evidence = self._get_evidence_agent().build_evidence(
            candidates=reranked,
            graph_metadata=graph_metadata,
        )
        t1 = time.perf_counter()
        timings.append(StageTiming(
            stage_name="Evidence Agent",
            duration_ms=(t1 - t0) * 1000,
        ))

        # ── Stage 7: Context Builder Agent ────────────────────────────────
        t0 = time.perf_counter()
        pipeline_end = time.perf_counter()
        total_ms = (pipeline_end - pipeline_start) * 1000

        # Construct serializable metadata
        retrieval_metadata = {
            "query": query,
            "intent": query_analysis.query_intent.value,
            "planner_plan": {
                "semantic_needed": plan.semantic_search_needed,
                "graph_needed": plan.graph_search_needed,
                "both_needed": plan.both_needed,
                "traversal_depth": plan.traversal_depth,
            },
            "candidates_count": len(fused),
            "stage_timings": [st.model_dump() for st in timings],
            "total_duration_ms": total_ms,
        }

        final_package = self._get_context_builder().build_context(
            evidence=evidence,
            retrieval_metadata=retrieval_metadata,
        )
        t1 = time.perf_counter()

        # Update timings list and total duration
        timings.append(StageTiming(
            stage_name="Context Builder Agent",
            duration_ms=(t1 - t0) * 1000,
        ))
        total_ms += (t1 - t0) * 1000
        final_package["retrieval_metadata"]["stage_timings"] = [st.model_dump() for st in timings]
        final_package["retrieval_metadata"]["total_duration_ms"] = total_ms

        result = RetrievalResult(**final_package)

        # -- Summary -------------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("  RETRIEVAL WORKFLOW COMPLETE")
        logger.info("=" * 50)
        for st in timings:
            logger.info(f"  {st.stage_name:<25s}: {st.duration_ms:>8.1f} ms")
        logger.info(f"  {'TOTAL':<25s}: {total_ms:>8.1f} ms")
        logger.info("=" * 50)

        return result

    def close(self) -> None:
        """Release resources (Neo4j connection, etc.)."""
        if self._graph_retriever is not None:
            self._graph_retriever.close()
