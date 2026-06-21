"""
Step 5 — Entity Extraction
Reads chunks.json, sends each chunk to Mistral LLM,
and extracts named entities.
"""

from __future__ import annotations

import json
import logging

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# PROMPTS
# ------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert at extracting named entities from text.

Return ONLY valid JSON.

No markdown.
No explanation.
No code fences.
"""

USER_PROMPT = """
Extract only real-world entities.

Allowed:

- Organization
- Product
- Technology
- Company
- Device
- Software
- Protocol
- Location
- Person

DO NOT extract:

- Table of contents items
- Section titles
- Chapter titles
- Instructions
- Sentences
- Menu names
- Generic concepts
- UI labels
- Page headings

Return JSON array:

[
  {{
    "name": "entity name",
    "type": "PERSON | ORGANIZATION | PRODUCT | TECHNOLOGY | CONCEPT | METRIC | LOCATION | EVENT"
  }}
]

Rules:
- Extract only entities clearly present in text
- If none exist return []
- Return JSON only

Text:
{content}
"""


# ------------------------------------------------------------------
# ENTITY EXTRACTOR
# ------------------------------------------------------------------

class EntityExtractor:

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest"
    ):
        self.llm = ChatMistralAI(
            api_key=api_key,
            model=model,
            temperature=0
        )

    # --------------------------------------------------------------

    def _parse_llm_response(self, raw: str) -> list:

        raw = raw.strip()

        try:
            return json.loads(raw)
        except Exception:
            pass

        if "```" in raw:

            parts = raw.split("```")

            for part in parts:

                part = part.strip()

                if part.startswith("json"):
                    part = part[4:].strip()

                try:
                    return json.loads(part)
                except Exception:
                    continue

        return []

    # --------------------------------------------------------------

    def extract_from_chunk(self, chunk: dict) -> list:

        content = chunk.get("content", "").strip()

        if len(content) < 15:
            return []

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=USER_PROMPT.format(
                    content=content[:2000]
                )
            )
        ]

        try:

            response = self.llm.invoke(messages)

            print("\n" + "=" * 70)
            print("CHUNK :", chunk["chunk_id"])
            print("CONTENT:")
            print(content[:300])
            print("\nLLM RESPONSE:")
            print(response.content)
            print("=" * 70 + "\n")

            entities_raw = self._parse_llm_response(
                response.content
            )

            results = []

            for ent in entities_raw:

                if (
                    isinstance(ent, dict)
                    and "name" in ent
                    and "type" in ent
                ):

                    name = str(ent["name"]).strip()
                    entity_type = str(ent["type"]).upper().strip()

                if not name:
                    continue

                # -----------------------------
                # ALLOWED ENTITY TYPES
                # -----------------------------
                ALLOWED_TYPES = {
                    "ORGANIZATION",
                    "COMPANY",
                    "PRODUCT",
                    "TECHNOLOGY",
                    "DEVICE",
                    "SOFTWARE",
                    "PROTOCOL"
                }

                if entity_type not in ALLOWED_TYPES:
                    continue

                # -----------------------------
                # NOISE FILTERS
                # -----------------------------
                if len(name) < 3:
                    continue

                if name.isdigit():
                    continue

                if len(name.split()) > 8:
                    continue

                results.append(
                    {
                        "name": name,
                        "type": entity_type,
                        "chunk_id": chunk["chunk_id"],
                        "page_number": chunk.get(
                            "page_number"
                        ),
                        "section_name": chunk.get(
                            "section_name",
                            ""
                        ),
                        "source_file": chunk.get(
                            "source_file",
                            ""
                        )
                    }
                )

            return results

        except Exception as e:

            print(
                f"\nEXTRACTION ERROR [{chunk['chunk_id']}]"
            )
            print(str(e))

            return []

    # --------------------------------------------------------------

    def extract_all(self, chunks: list) -> list:

        all_entities = []

        total = len(chunks)

        for i, chunk in enumerate(chunks):

            logger.info(
                f"[{i+1}/{total}] Processing "
                f"{chunk['chunk_id']}"
            )

            entities = self.extract_from_chunk(
                chunk
            )

            logger.info(
                f"Found {len(entities)} entities"
            )

            all_entities.extend(entities)

        logger.info(
            f"Total entities extracted: "
            f"{len(all_entities)}"
        )

        return all_entities