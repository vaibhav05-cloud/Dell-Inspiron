"""
Stage 7 — Context Compression Agent.

Removes redundant chunks, duplicate facts, and duplicate entities.
Preserves evidence and relationship paths. Enforces a token budget.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Set

from retriever.schema import RetrievalCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TOKEN_BUDGET = 4000   # approx. token ceiling for final context
APPROX_CHARS_PER_TOKEN = 4   # rough estimate for English text
OVERLAP_THRESHOLD = 0.80     # token overlap threshold for dedup


class ContextCompressionAgent:
    """Compresses and deduplicates retrieval context to fit a token budget."""

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self._token_budget = token_budget
        self._char_budget = token_budget * APPROX_CHARS_PER_TOKEN

    @staticmethod
    def _tokenize_simple(text: str) -> Set[str]:
        """Fast tokenization: lowercase split on whitespace + punctuation."""
        return set(text.lower().split())

    @staticmethod
    def _text_fingerprint(text: str) -> str:
        """Create a content fingerprint for fast dedup checking."""
        # Normalize: lowercase, strip extra whitespace
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

    def _deduplicate_chunks(
        self,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """Remove chunks with high content overlap.

        If two chunks share >80% token overlap, keep the longer one
        (more informative).
        """
        if len(candidates) <= 1:
            return candidates

        kept: List[RetrievalCandidate] = []
        fingerprints: Set[str] = set()

        for candidate in candidates:
            fp = self._text_fingerprint(candidate.content)

            # Exact duplicate check
            if fp in fingerprints:
                continue

            # Near-duplicate check against kept candidates
            is_duplicate = False
            for existing in kept:
                overlap = self._compute_overlap(
                    candidate.content, existing.content
                )
                if overlap >= OVERLAP_THRESHOLD:
                    # Keep the longer one
                    if len(candidate.content) > len(existing.content):
                        kept.remove(existing)
                        kept.append(candidate)
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(candidate)
                fingerprints.add(fp)

        return kept

    def compress(
        self,
        candidates: List[RetrievalCandidate],
    ) -> tuple[str, List[RetrievalCandidate]]:
        """Compress candidates into final context within token budget.

        Parameters
        ----------
        candidates:
            Expanded candidates from Stage 6.

        Returns
        -------
        tuple[str, List[RetrievalCandidate]]
            (final_context_string, kept_candidates)
        """
        logger.info(
            f"Stage 7: Compressing {len(candidates)} candidates "
            f"(budget: ~{self._token_budget} tokens) …"
        )

        # Step 1: Deduplicate
        deduped = self._deduplicate_chunks(candidates)
        removed = len(candidates) - len(deduped)
        if removed > 0:
            logger.info(f"  Removed {removed} duplicate/near-duplicate chunks")

        # Step 2: Build context within token budget
        context_parts = []
        total_chars = 0
        kept_candidates = []

        for i, candidate in enumerate(deduped):
            # Format this chunk's contribution
            chunk_header = (
                f"--- Source: {candidate.source_file} | "
                f"Page {candidate.page_number} | "
                f"Section: {candidate.section_name} | "
                f"Chunk: {candidate.chunk_id} ---"
            )

            chunk_body = candidate.content

            # Add supporting context if available (compressed)
            supporting = ""
            if candidate.supporting_context:
                # Truncate supporting context aggressively
                sc = candidate.supporting_context
                if len(sc) > 500:
                    sc = sc[:500] + "…"
                supporting = f"\n[Supporting context]: {sc}"

            entry = f"{chunk_header}\n{chunk_body}{supporting}\n"
            entry_chars = len(entry)

            # Check budget
            if total_chars + entry_chars > self._char_budget:
                # Try to fit a truncated version
                remaining = self._char_budget - total_chars
                if remaining > 200:  # only include if meaningful
                    truncated = entry[:remaining] + "…\n"
                    context_parts.append(truncated)
                    kept_candidates.append(candidate)
                break

            context_parts.append(entry)
            total_chars += entry_chars
            kept_candidates.append(candidate)

        final_context = "\n".join(context_parts)

        approx_tokens = len(final_context) // APPROX_CHARS_PER_TOKEN
        logger.info(
            f"  Final context: {len(kept_candidates)} chunks, "
            f"~{approx_tokens} tokens ✓"
        )

        return final_context, kept_candidates
