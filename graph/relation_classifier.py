"""
graph/relation_classifier.py

SPEED FIX: Added classify_batch() method.
  Old approach: 1 Mistral call per entity pair
                45 pairs × N chunks = ~1350 individual API calls → 15 min

  New approach: 7 pairs per Mistral call (batched)
                ceil(45/7) = 7 batch calls × N chunks = ~210 API calls → 2-3 min
                Same quality, ~6x faster.

QUALITY FIX: MAX_PAIRS_PER_CHUNK stays at 45 (covers all C(10,2) pairs).
             Added PRODUCES relation.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_PAIRS_PER_CHUNK = 45
BATCH_SIZE          = 7   # pairs per Mistral call

ALLOWED_RELATIONS = {
    "USES", "SUPPORTS", "MANAGES", "PART_OF", "CONTAINS",
    "CONNECTED_TO", "DEPENDS_ON", "HOSTS", "LOCATED_IN",
    "PRODUCES", "NONE",
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_entity(name: str) -> bool:
    name = str(name).strip()
    if len(name) < 3:
        return False
    if name.isdigit():
        return False
    return True


def generate_entity_pairs(entity_names: list) -> list:
    unique = list(dict.fromkeys(
        e for e in entity_names if is_valid_entity(e)
    ))
    return list(combinations(unique, 2))[:MAX_PAIRS_PER_CHUNK]


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_SYSTEM = """
You are a knowledge graph relation extraction expert.
Return ONLY valid JSON. No markdown. No explanation.
Example: {"relation": "USES"}
If no relationship exists: {"relation": "NONE"}
"""

SINGLE_USER = """
Entity 1: {entity1}
Entity 2: {entity2}

Text:
{text}

Allowed Relations:
USES, SUPPORTS, MANAGES, PART_OF, CONTAINS,
CONNECTED_TO, DEPENDS_ON, HOSTS, LOCATED_IN, PRODUCES, NONE

Return JSON only.
"""

BATCH_SYSTEM = """
You are a knowledge graph relationship classifier.
Return ONLY a valid JSON array. No markdown. No explanation.

Example for 3 pairs:
[{"pair": 1, "relation": "PRODUCES"}, {"pair": 2, "relation": "NONE"}, {"pair": 3, "relation": "USES"}]

Rules:
- Every pair must have exactly one entry in output.
- Use NONE if no clear relationship is supported by the text.
- Never invent relationships.
"""

BATCH_USER = """
Classify the relationship for each entity pair using the text below.

Allowed relations:
USES, SUPPORTS, MANAGES, PART_OF, CONTAINS, CONNECTED_TO,
DEPENDS_ON, HOSTS, LOCATED_IN, PRODUCES, NONE

Text:
{text}

Pairs to classify:
{pairs}

Return a JSON array with one object per pair.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSIFIER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class RelationClassifier:

    def __init__(self, api_key: str):
        self.llm = ChatMistralAI(
            api_key     = api_key,
            model       = "mistral-small-latest",
            temperature = 0,
        )

    # ── Batch (primary — fast) ────────────────────────────────────────────────

    def classify_batch(
        self,
        pairs: list[tuple[str, str]],
        text:  str,
    ) -> list[str]:
        """
        Classify multiple entity pairs in ONE Mistral call.
        7 pairs per call = 6x fewer API calls = 6x faster runtime.

        Returns list of relation strings, same length as pairs.
        Falls back to 'NONE' for any pair that fails to parse.
        """
        if not pairs:
            return []

        pairs_text = "\n".join(
            f'{i + 1}. "{e1}" and "{e2}"'
            for i, (e1, e2) in enumerate(pairs)
        )

        messages = [
            SystemMessage(content=BATCH_SYSTEM),
            HumanMessage(content=BATCH_USER.format(
                text  = text[:1500],
                pairs = pairs_text,
            )),
        ]

        try:
            response = self.llm.invoke(messages)
            return self._parse_batch_response(response.content, len(pairs))
        except Exception as e:
            logger.warning(f"Batch call failed: {e}")
            return ["NONE"] * len(pairs)

    def _parse_batch_response(self, raw: str, expected: int) -> list[str]:
        """Parse batch JSON — handles markdown fences and partial responses."""
        raw = raw.strip()

        # Strip markdown fences if present
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    raw = part
                    break

        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                return ["NONE"] * expected

            rel_map: dict[int, str] = {}
            for item in data:
                if isinstance(item, dict):
                    pair_num = item.get("pair")
                    relation = str(item.get("relation", "NONE")).upper().strip()
                    if isinstance(pair_num, int) and relation in ALLOWED_RELATIONS:
                        rel_map[pair_num] = relation

            return [rel_map.get(i + 1, "NONE") for i in range(expected)]

        except Exception as e:
            logger.warning(f"Batch parse failed: {e}")
            return ["NONE"] * expected

    # ── Single (fallback only) ────────────────────────────────────────────────

    def classify(self, entity1: str, entity2: str, text: str) -> str:
        """Single-pair classification — kept as fallback."""
        messages = [
            SystemMessage(content=SINGLE_SYSTEM),
            HumanMessage(content=SINGLE_USER.format(
                entity1 = entity1,
                entity2 = entity2,
                text    = text[:1500],
            )),
        ]
        response = self.llm.invoke(messages)
        return response.content