"""
Co-occurrence Relationship Extractor
No LLM required — fast, pure Python.

Core idea:
  If two entities appear in the same chunk of text,
  they are related in context → CO_OCCURS_WITH relationship.

Why this is needed:
  Mistral's relationship extractor only checks the TOP-10 entities
  per chunk and classifies ~45 pairs per chunk. Entities that appear
  in chunks with many other entities, or that are spread across
  multiple chunks, often get ZERO relationships from Mistral.
  This creates the "ring of isolated dots" problem in the graph.

  Co-occurrence adds baseline connectivity for ALL entity pairs
  that share any chunk, giving GraphRAG enough graph structure
  to traverse even when Mistral found no semantic relation.

FIX (v2):
  The naive version creates a CO_OCCURS_WITH edge for every pair of
  entities that share even one chunk. If a chunk mentions 20 entities,
  that's C(20,2) = 190 edges from ONE chunk — almost all noise. This
  blows up the graph into a near-complete hairball (e.g. 417 nodes /
  8179 edges ≈ 20 edges/node), which destroys the selectivity that
  makes GraphRAG traversal useful in the first place.

  Two independent caps fix this:
    1. max_entities_per_chunk — if a chunk has more entities than this,
       only the most "important" ones are kept (same idea as Mistral's
       top-10 cap). Importance = global entity frequency by default,
       so generic/common entities don't dominate every chunk.
    2. min_cooccurrence — an edge is only kept if the pair co-occurs
       in at least this many chunks (default 2). This kills one-off
       coincidental pairings and keeps edges that reflect a real,
       repeated association. Edge weight = number of shared chunks.

Usage in run_graph.py:
  cooccur_rels = extract_cooccurrence(chunks, entities, existing_pairs)
  # existing_pairs = pairs already covered by Mistral (to avoid duplicates)
"""

from __future__ import annotations
import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def _entity_in_chunk(entity_name: str, chunk_text: str) -> bool:
    """
    Whole-word match (case-insensitive).
    Using \b boundaries so "USB" doesn't match inside "submarine".
    """
    pattern = r'\b' + re.escape(entity_name) + r'\b'
    return bool(re.search(pattern, chunk_text, re.IGNORECASE))


def extract_cooccurrence(
    chunks: list,
    entities: list,
    existing_pairs: set | None = None,
    max_entities_per_chunk: int = 10,
    min_cooccurrence: int = 2,
) -> list:
    """
    For pairs of entities that repeatedly appear together in the same
    chunks, create a CO_OCCURS_WITH relationship. Designed to add
    baseline graph connectivity WITHOUT turning the graph into a
    near-complete hairball.

    Args:
        chunks                 : list of chunk dicts {"chunk_id": ..., "content": ...}
        entities                : list of entity dicts {"name": ..., "type": ...}
        existing_pairs          : set of (src_lower, tgt_lower) tuples already
                                   covered by Mistral — skipped to avoid duplicates.
                                   Build it with:
                                     {tuple(sorted([r["source"].lower(), r["target"].lower()]))
                                      for r in mistral_relationships}
        max_entities_per_chunk  : if a chunk contains more distinct entities than
                                   this, only the top-N most globally-frequent
                                   entities in that chunk are considered for
                                   pairing. Prevents combinatorial blowup on
                                   chunks that happen to mention many entities.
                                   Set to None to disable the cap.
        min_cooccurrence        : minimum number of distinct chunks a pair must
                                   co-occur in before an edge is created. Default 2
                                   filters out one-off, likely-coincidental pairs.
                                   Set to 1 to restore the old (noisy) behavior.

    Returns:
        list of relationship dicts with "relation": "CO_OCCURS_WITH" and a
        "weight" field = number of chunks the pair co-occurred in, plus
        "chunk_ids" listing where.
    """

    if existing_pairs is None:
        existing_pairs = set()

    # Unique entity names (deduplicated)
    entity_names = list({e["name"] for e in entities})

    # Pass 1: find which entities appear in which chunks, and count
    # global entity frequency (used to rank entities when a chunk is
    # over the max_entities_per_chunk cap).
    chunk_presence: list[tuple[str, list[str]]] = []  # (chunk_id, present_names)
    entity_freq: Counter = Counter()

    for chunk in chunks:
        content = chunk.get("content", "")
        chunk_id = chunk.get("chunk_id", "")

        present = [name for name in entity_names if _entity_in_chunk(name, content)]
        chunk_presence.append((chunk_id, present))
        entity_freq.update(present)

    # Pass 2: build pair -> (count, chunk_ids) using the capped entity list per chunk
    pair_counts: dict[tuple, dict] = {}

    for chunk_id, present in chunk_presence:
        if len(present) < 2:
            continue

        if max_entities_per_chunk is not None and len(present) > max_entities_per_chunk:
            # Keep only the most globally-frequent entities in this chunk.
            present = sorted(present, key=lambda n: entity_freq[n], reverse=True)
            present = present[:max_entities_per_chunk]

        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                e1, e2 = present[i], present[j]
                pair = tuple(sorted([e1.lower(), e2.lower()]))

                if pair in existing_pairs:
                    continue

                if pair not in pair_counts:
                    pair_counts[pair] = {"source": e1, "target": e2, "count": 0, "chunk_ids": []}
                pair_counts[pair]["count"] += 1
                pair_counts[pair]["chunk_ids"].append(chunk_id)

    # Pass 3: filter by min_cooccurrence and build final relationship list
    relationships: list[dict] = []
    for pair, info in pair_counts.items():
        if info["count"] < min_cooccurrence:
            continue
        relationships.append({
            "source":    info["source"],
            "relation":  "CO_OCCURS_WITH",
            "target":    info["target"],
            "weight":    info["count"],
            "chunk_ids": info["chunk_ids"],
        })

    logger.info(
        f"Co-occurrence: {len(relationships)} relationships kept "
        f"(after max_entities_per_chunk={max_entities_per_chunk}, "
        f"min_cooccurrence={min_cooccurrence} filtering)"
    )
    return relationships