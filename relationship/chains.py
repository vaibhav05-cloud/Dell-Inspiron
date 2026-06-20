"""
LangChain extraction chains for relationship extraction.

Builds a reusable chain:  prompt → ChatMistralAI → structured LLMRelationshipOutput.
The chain receives chunk content PLUS the list of already-extracted entities,
so the LLM can only reference known entities.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from relationship.schema import LLMRelationshipOutput

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT
# ─────────────────────────────────────────────────────────────────────────────

RELATIONSHIP_SYSTEM_PROMPT = """\
You are an expert relationship-extraction engine for a knowledge-graph pipeline.

Given a text chunk from a PDF document AND a list of entities already extracted \
from that chunk, identify **all meaningful relationships** between those entities.

### Relationship types you MUST consider
| Type | When to use |
|---|---|
| RELATED_TO | Generic association between two entities |
| USES | Entity A uses, employs, or leverages entity B |
| DEPENDS_ON | Entity A requires or depends on entity B |
| CONTRIBUTES_TO | Entity A feeds into, supports, or contributes to B |
| IMPROVES | Entity A enhances, optimises, or improves B |
| REDUCES | Entity A decreases, lowers, or reduces B |
| INCREASES | Entity A grows, raises, or increases B |
| PART_OF | Entity A is a component, subset, or member of B |
| BELONGS_TO | Entity A is owned by or affiliated with B |
| LOCATED_IN | Entity A is geographically located within B |
| CUSTOM | Any other domain-specific relation not covered above |

### Rules
1. ONLY use entity names from the provided entity list — do NOT invent new entities.
2. Source and target entity names MUST exactly match names in the entity list.
3. Each relationship MUST include:
   - `source_entity_name`: exact name from the entity list
   - `target_entity_name`: exact name from the entity list
   - `relationship_type`: one of the types above (exact string)
   - `confidence`: float 0.0–1.0 reflecting how certain the relationship is
4. A single entity pair can have multiple relationship types if justified by context.
5. If no meaningful relationships exist, return an empty list.
6. Do NOT create self-referencing relationships (source == target).
7. Prefer specific relationship types over RELATED_TO when possible.
8. For metrics/KPIs, relate them to the concepts or products they measure.\
"""

RELATIONSHIP_HUMAN_PROMPT = """\
### Chunk metadata
- chunk_id: {chunk_id}
- chunk_type: {chunk_type}
- section: {section_name}
- page: {page_number}

### Entities already extracted from this chunk
{entity_list}

### Chunk content
{content}

Extract all relationships between the entities listed above.\
"""


# ─────────────────────────────────────────────────────────────────────────────
#  CHAIN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_relationship_chain():
    """Build and return a reusable LangChain relationship extraction chain.

    Returns
    -------
    chain
        A runnable that accepts a dict with keys matching the prompt
        variables and returns an ``LLMRelationshipOutput`` instance.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY not found in environment. "
            "Add it to your .env file."
        )

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0.0,
        max_tokens=4096,
    )

    # Bind structured output so the LLM returns valid LLMRelationshipOutput
    structured_llm = llm.with_structured_output(LLMRelationshipOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RELATIONSHIP_SYSTEM_PROMPT),
        ("human",  RELATIONSHIP_HUMAN_PROMPT),
    ])

    chain = prompt | structured_llm

    return chain
