"""
Stage 8 — Answer Synthesis Chain.

Accepts the RetrievalResult from the 7-agent pipeline, selects the Top 3
re-ranked chunks, consolidates evidence, generates a grounded answer via
LLM, and produces a deterministic SynthesisResult with attribution,
reasoning path, and confidence scoring.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

from retriever.schema import RetrievalResult
from synthesizer.prompts import SYNTHESIS_PROMPT
from synthesizer.synthesis_schema import (
    ConfidenceLevel,
    ConsolidatedEvidence,
    EvidenceAttribution,
    SynthesisResult,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Number of top evidence chunks to use for synthesis
TOP_K_EVIDENCE = 3


class AnswerSynthesisChain:
    """LangChain-based answer synthesis chain.

    Single-pass, no-loop, deterministic-structure chain that:
    1. Selects Top 3 evidence chunks
    2. Consolidates evidence (dedup + merge)
    3. Generates answer via LLM
    4. Builds evidence attribution
    5. Constructs reasoning path from graph relationships
    6. Computes confidence score
    """

    def __init__(
        self,
        model_name: str = "mistral-small-latest",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "MISTRAL_API_KEY not found. Add it to your .env file."
            )

        self._llm = ChatMistralAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._chain = SYNTHESIS_PROMPT | self._llm

    # ── Step 1: Evidence Selection ────────────────────────────────────────

    @staticmethod
    def _select_top_evidence(
        retrieval_result: RetrievalResult,
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """Select the Top 3 highest-ranked evidence chunks.

        Returns
        -------
        Tuple of (top_chunks, graph_paths, source_entities)
        """
        evidence_chunks = retrieval_result.evidence_chunks or []

        # Filter out neighbor chunks — we only want primary evidence
        primary_chunks = [
            c for c in evidence_chunks if not c.get("is_neighbor", False)
        ]

        # If we have fewer than 3 primary chunks, include neighbors as fallback
        if len(primary_chunks) < TOP_K_EVIDENCE:
            primary_chunks = evidence_chunks

        top_chunks = primary_chunks[:TOP_K_EVIDENCE]

        return (
            top_chunks,
            retrieval_result.graph_paths or [],
            retrieval_result.source_entities or [],
        )

    # ── Step 2: Evidence Consolidation ────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Simple whitespace tokenization for overlap detection."""
        return set(text.lower().split())

    @classmethod
    def _compute_overlap(cls, text_a: str, text_b: str) -> float:
        """Token-set overlap ratio between two texts."""
        tokens_a = cls._tokenize(text_a)
        tokens_b = cls._tokenize(text_b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        smaller = min(len(tokens_a), len(tokens_b))
        return len(intersection) / smaller if smaller > 0 else 0.0

    @classmethod
    def _consolidate_evidence(
        cls,
        top_chunks: List[dict],
        graph_paths: List[dict],
    ) -> ConsolidatedEvidence:
        """Merge, deduplicate, and categorize facts from the Top 3 chunks.

        Identifies:
        - Common facts: sentences appearing in 2+ chunks (high confidence)
        - Complementary facts: sentences unique to 1 chunk
        - Graph relationships: formatted as "Source → Relationship → Target"
        """
        # Extract sentences from each chunk
        chunk_sentences: List[List[str]] = []
        for chunk in top_chunks:
            content = chunk.get("content", "")
            sentences = [
                s.strip() for s in content.replace("\n", ". ").split(".")
                if s.strip() and len(s.strip()) > 10
            ]
            chunk_sentences.append(sentences)

        # Identify common vs. complementary facts
        all_sentences = []
        for sentences in chunk_sentences:
            all_sentences.extend(sentences)

        common_facts = []
        complementary_facts = []
        seen = set()

        for i, sentences_i in enumerate(chunk_sentences):
            for sentence in sentences_i:
                if sentence in seen:
                    continue
                seen.add(sentence)

                # Check if this sentence has high overlap with any sentence in another chunk
                is_common = False
                for j, sentences_j in enumerate(chunk_sentences):
                    if i == j:
                        continue
                    for other in sentences_j:
                        if cls._compute_overlap(sentence, other) >= 0.70:
                            is_common = True
                            break
                    if is_common:
                        break

                if is_common:
                    common_facts.append(sentence)
                else:
                    complementary_facts.append(sentence)

        # Build merged context — ordered by chunk rank (highest relevance first)
        merged_parts = []
        for idx, chunk in enumerate(top_chunks):
            score = chunk.get("score", 0.0)
            source = chunk.get("source_document", "unknown")
            page = chunk.get("page_number", 0)
            section = chunk.get("section_name", "")

            header = (
                f"[Evidence {idx + 1} | Score: {score:.4f} | "
                f"Source: {source} | Page {page}"
            )
            if section:
                header += f" | Section: {section}"
            header += "]"

            merged_parts.append(f"{header}\n{chunk.get('content', '')}")

        merged_context = "\n\n".join(merged_parts)

        # Format graph relationships
        graph_relationship_strings = []
        relationship_confidences = []
        for path in graph_paths:
            src = path.get("source", "?")
            rel = path.get("relationship", "?")
            tgt = path.get("target", "?")
            conf = path.get("confidence", 0.0)
            graph_relationship_strings.append(f"{src} → {rel} → {tgt}")
            relationship_confidences.append(conf)

        # Build attributions
        attributions = []
        relevance_scores = []
        for chunk in top_chunks:
            attributions.append(EvidenceAttribution(
                page_number=chunk.get("page_number", 0),
                chunk_id=chunk.get("chunk_id", ""),
                source_document=chunk.get("source_document", ""),
            ))
            relevance_scores.append(chunk.get("score", 0.0))

        return ConsolidatedEvidence(
            merged_context=merged_context,
            common_facts=common_facts[:10],  # cap for prompt size
            complementary_facts=complementary_facts[:15],
            graph_relationships=graph_relationship_strings,
            attributions=attributions,
            relevance_scores=relevance_scores,
            relationship_confidences=relationship_confidences,
        )

    # ── Step 3: Answer Generation (LLM call) ─────────────────────────────

    def _generate_answer(
        self,
        query: str,
        consolidated: ConsolidatedEvidence,
    ) -> str:
        """Invoke the LLM to generate a grounded answer.

        This is the ONLY non-deterministic step in the chain.
        """
        # Format graph relationships for the prompt
        if consolidated.graph_relationships:
            graph_str = "\n".join(
                f"- {r}" for r in consolidated.graph_relationships
            )
        else:
            graph_str = "No graph relationships available."

        result = self._chain.invoke({
            "query": query,
            "consolidated_evidence": consolidated.merged_context,
            "graph_relationships": graph_str,
        })

        return result.content.strip()

    # ── Step 4: Evidence Attribution ──────────────────────────────────────

    @staticmethod
    def _build_attribution(
        consolidated: ConsolidatedEvidence,
    ) -> List[EvidenceAttribution]:
        """Return deterministic evidence attributions from the consolidated evidence."""
        return consolidated.attributions

    # ── Step 5: Reasoning Path ────────────────────────────────────────────

    @staticmethod
    def _build_reasoning_path(
        consolidated: ConsolidatedEvidence,
        top_chunks: List[dict],
    ) -> List[str]:
        """Build human-readable reasoning trail from graph relationships.

        Falls back to evidence-based paths if no graph relationships exist.
        """
        paths = []

        # Primary: graph relationship paths
        for rel_str in consolidated.graph_relationships:
            paths.append(rel_str)

        # Fallback: build entity-to-evidence paths from chunk metadata
        if not paths:
            for chunk in top_chunks:
                section = chunk.get("section_name", "")
                source = chunk.get("source_document", "")
                if section:
                    paths.append(f"Query → {section} (from {source})")
                elif source:
                    paths.append(f"Query → Evidence (from {source})")

        return paths if paths else ["Query → Direct Evidence Match"]

    # ── Step 6: Confidence Calculation ────────────────────────────────────

    @classmethod
    def _compute_confidence(
        cls,
        consolidated: ConsolidatedEvidence,
        top_chunks: List[dict],
    ) -> str:
        """Compute confidence score from relevance, relationship, and agreement signals.

        Formula:
            composite = (0.50 × avg_relevance)
                      + (0.25 × avg_relationship_confidence)
                      + (0.25 × evidence_agreement)

        Mapping:
            composite ≥ 0.70 → High
            composite ≥ 0.40 → Medium
            composite <  0.40 → Low
        """
        # Average retrieval relevance (rerank scores)
        scores = consolidated.relevance_scores
        if scores:
            # Normalize rerank scores to [0, 1] range
            # Cross-encoder scores can be negative or > 1, so we sigmoid-normalize
            import math
            normalized = [1.0 / (1.0 + math.exp(-s)) for s in scores]
            avg_relevance = sum(normalized) / len(normalized)
        else:
            avg_relevance = 0.0

        # Average relationship confidence
        rel_confs = consolidated.relationship_confidences
        if rel_confs:
            avg_relationship = sum(rel_confs) / len(rel_confs)
        else:
            avg_relationship = 0.0

        # Evidence agreement: pairwise token overlap across top chunks
        contents = [c.get("content", "") for c in top_chunks]
        overlaps = []
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                overlap = cls._compute_overlap(contents[i], contents[j])
                overlaps.append(overlap)
        evidence_agreement = (
            sum(overlaps) / len(overlaps) if overlaps else 0.0
        )

        # Weighted composite
        composite = (
            0.50 * avg_relevance
            + 0.25 * avg_relationship
            + 0.25 * evidence_agreement
        )

        logger.info(
            f"  Confidence breakdown: "
            f"relevance={avg_relevance:.3f}, "
            f"relationship={avg_relationship:.3f}, "
            f"agreement={evidence_agreement:.3f} "
            f"→ composite={composite:.3f}"
        )

        if composite >= 0.70:
            return ConfidenceLevel.HIGH.value
        elif composite >= 0.40:
            return ConfidenceLevel.MEDIUM.value
        else:
            return ConfidenceLevel.LOW.value

    # ── Main synthesize method ────────────────────────────────────────────

    def synthesize(self, query: str, retrieval_result: RetrievalResult) -> SynthesisResult:
        """Execute the full 6-step answer synthesis workflow.

        Parameters
        ----------
        query:
            The user's original natural-language question.
        retrieval_result:
            Complete output from the 7-agent retrieval pipeline.

        Returns
        -------
        SynthesisResult
            Deterministic-structure JSON containing answer, evidence,
            reasoning_path, and confidence.
        """
        synth_start = time.perf_counter()

        logger.info("")
        logger.info("=" * 50)
        logger.info("  ANSWER SYNTHESIS LAYER")
        logger.info("=" * 50)

        # Step 1: Evidence Selection
        logger.info("Step 1: Selecting Top 3 evidence chunks …")
        top_chunks, graph_paths, source_entities = self._select_top_evidence(
            retrieval_result
        )
        logger.info(f"  Selected {len(top_chunks)} primary chunks, "
                     f"{len(graph_paths)} graph paths.")

        # Step 2: Evidence Consolidation
        logger.info("Step 2: Consolidating evidence …")
        consolidated = self._consolidate_evidence(top_chunks, graph_paths)
        logger.info(
            f"  Common facts: {len(consolidated.common_facts)}, "
            f"Complementary facts: {len(consolidated.complementary_facts)}, "
            f"Graph relationships: {len(consolidated.graph_relationships)}"
        )

        # Step 3: Answer Generation (LLM)
        logger.info("Step 3: Generating answer via LLM …")
        t0 = time.perf_counter()
        answer = self._generate_answer(query, consolidated)
        t1 = time.perf_counter()
        logger.info(f"  Answer generated in {(t1 - t0) * 1000:.1f} ms")

        # Step 4: Evidence Attribution
        logger.info("Step 4: Building evidence attribution …")
        evidence = self._build_attribution(consolidated)

        # Step 5: Reasoning Path
        logger.info("Step 5: Constructing reasoning path …")
        reasoning_path = self._build_reasoning_path(consolidated, top_chunks)

        # Step 6: Confidence Calculation
        logger.info("Step 6: Computing confidence score …")
        confidence = self._compute_confidence(consolidated, top_chunks)

        synth_end = time.perf_counter()
        total_ms = (synth_end - synth_start) * 1000

        logger.info("")
        logger.info("=" * 50)
        logger.info("  SYNTHESIS COMPLETE")
        logger.info(f"  Confidence: {confidence}")
        logger.info(f"  Total synthesis time: {total_ms:.1f} ms")
        logger.info("=" * 50)

        return SynthesisResult(
            answer=answer,
            evidence=[e.model_dump() for e in evidence],
            reasoning_path=reasoning_path,
            confidence=confidence,
        )
