"""
Stage 7 — Context Builder Agent.

Deduplicates evidence chunks, compiles the final context string within a token budget,
and builds the final output package for answer generation.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_TOKEN_BUDGET = 4000
APPROX_CHARS_PER_TOKEN = 4
OVERLAP_THRESHOLD = 0.80


class ContextBuilderAgent:
    """Formats and compresses evidence chunks into the final context package."""

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self._token_budget = token_budget
        self._char_budget = token_budget * APPROX_CHARS_PER_TOKEN

    @staticmethod
    def _tokenize_simple(text: str) -> Set[str]:
        """Simple whitespace + punctuation tokenization."""
        return set(text.lower().split())

    @staticmethod
    def _text_fingerprint(text: str) -> str:
        """Create MD5 fingerprint of normalized content."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _compute_overlap(self, text_a: str, text_b: str) -> float:
        """Compute token overlap ratio between two texts."""
        tokens_a = self._tokenize_simple(text_a)
        tokens_b = self._tokenize_simple(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        smaller = min(len(tokens_a), len(tokens_b))
        return len(intersection) / smaller if smaller > 0 else 0.0

    def _deduplicate_chunks(self, chunks: List[dict]) -> List[dict]:
        """Remove highly overlapping chunks."""
        if len(chunks) <= 1:
            return chunks

        kept = []
        fingerprints = set()

        for chunk in chunks:
            content = chunk["content"]
            fp = self._text_fingerprint(content)

            if fp in fingerprints:
                continue

            is_duplicate = False
            for existing in kept:
                overlap = self._compute_overlap(content, existing["content"])
                if overlap >= OVERLAP_THRESHOLD:
                    if len(content) > len(existing["content"]):
                        kept.remove(existing)
                        kept.append(chunk)
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(chunk)
                fingerprints.add(fp)

        return kept

    def build_context(
        self,
        evidence: dict,
        retrieval_metadata: dict,
    ) -> dict:
        """Deduplicate chunks, format final context, and return the final context package.

        Parameters
        ----------
        evidence:
            Dictionary output from Stage 6 (Evidence Agent).
        retrieval_metadata:
            Execution metadata (timings, planner decisions, etc.)

        Returns
        -------
        dict
            Exact output format dictionary.
        """
        logger.info("Stage 7: Running Context Builder Agent …")

        raw_chunks = evidence.get("evidence_chunks", [])
        deduped = self._deduplicate_chunks(raw_chunks)

        logger.info(
            f"  Context Builder Agent: Deduplicated {len(raw_chunks)} → {len(deduped)} chunks."
        )

        context_parts = []
        total_chars = 0
        final_chunks = []

        # Build context string enforcing token budget
        for chunk in deduped:
            prefix = "[Neighbor Context] " if chunk.get("is_neighbor") else ""
            header = (
                f"--- Source: {chunk['source_document']} | "
                f"Page {chunk['page_number']} | "
                f"Section: {chunk['section_name']} | "
                f"Chunk: {chunk['chunk_id']} ---"
            )
            body = chunk["content"]
            entry = f"{header}\n{prefix}{body}\n"
            entry_chars = len(entry)

            if total_chars + entry_chars > self._char_budget:
                remaining = self._char_budget - total_chars
                if remaining > 200:
                    truncated = entry[:remaining] + "…\n"
                    context_parts.append(truncated)
                    final_chunks.append(chunk)
                break

            context_parts.append(entry)
            total_chars += entry_chars
            final_chunks.append(chunk)

        answer_context = "\n".join(context_parts)
        approx_tokens = len(answer_context) // APPROX_CHARS_PER_TOKEN

        # Update metadata
        retrieval_metadata.update({
            "final_chunks_count": len(final_chunks),
            "estimated_context_tokens": approx_tokens,
            "token_budget": self._token_budget,
        })

        logger.info(f"  Context Builder Agent: Final context size: ~{approx_tokens} tokens.")

        return {
            "answer_context": answer_context,
            "evidence_chunks": final_chunks,
            "graph_paths": evidence.get("graph_paths", []),
            "source_entities": evidence.get("source_entities", []),
            "retrieval_metadata": retrieval_metadata,
        }
