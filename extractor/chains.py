"""
LangChain extraction chains for entity extraction.

Builds a reusable chain:  prompt → ChatMistralAI → structured LLMEntityOutput.
The chain is instantiated once and invoked per-chunk.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from extractor.schema import LLMEntityOutput

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert entity-extraction engine for a knowledge-graph pipeline.

Given a text chunk from a PDF document, extract **every** meaningful entity.

### Entity types you MUST look for
| Type | What to extract |
|---|---|
| ORGANIZATION | Company names, agencies, teams, departments |
| PERSON | Named individuals (first + last name preferred) |
| PRODUCT | Product names, model numbers, SKUs, branded offerings |
| TECHNOLOGY | Frameworks, programming languages, tools, platforms, protocols |
| PROJECT | Named initiatives, programmes, campaigns |
| LOCATION | Cities, countries, regions, addresses |
| METRIC_KPI | Quantitative metrics: revenue, growth rates, percentages, scores |
| DATE | Specific dates, quarters, years, time ranges |
| CONCEPT | Domain-specific terms, themes, business concepts, strategies |

### Rules
1. Return **every** entity you find — do not skip any.
2. Each entity MUST include:
   - `entity_name`: canonical, clean name (title-case for proper nouns)
   - `entity_type`: one of the types above (exact string)
   - `source_text`: the verbatim snippet (≤120 chars) evidencing this entity
   - `confidence`: float 0.0–1.0 reflecting extraction certainty
3. If the chunk contains **no** extractable entities, return an empty list.
4. Do NOT invent entities that are not in the text.
5. Prefer specificity: "Market Size $50 Billion" is better than just "Market Size".
6. For tables, treat each data row as a potential source of METRIC_KPI entities.
7. For image descriptions, extract any products, technologies, or concepts visible.\
"""

EXTRACTION_HUMAN_PROMPT = """\
### Chunk metadata
- chunk_id: {chunk_id}
- chunk_type: {chunk_type}
- section: {section_name}
- page: {page_number}

### Chunk content
{content}

Extract all entities from this chunk.\
"""


# ─────────────────────────────────────────────────────────────────────────────
#  CHAIN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_extraction_chain():
    """Build and return a reusable LangChain extraction chain.

    Returns
    -------
    chain
        A runnable that accepts a dict with keys matching the prompt
        variables and returns an ``LLMEntityOutput`` instance.
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

    # Bind structured output so the LLM returns valid LLMEntityOutput
    structured_llm = llm.with_structured_output(LLMEntityOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human",  EXTRACTION_HUMAN_PROMPT),
    ])

    chain = prompt | structured_llm

    return chain
