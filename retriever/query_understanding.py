"""
Stage 1 — Query Planner Agent.

Uses ChatMistralAI with structured output to analyze the query, extract entities,
classify intent, and plan the retrieval strategy (semantic, graph, or both).
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from retriever.schema import QueryAnalysis, QueryIntent, AgenticRetrievalPlan

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a query planning engine for a GraphRAG retrieval system.
Your job is to analyze a user's question and produce a structured retrieval plan.

### Instructions

1. **Extract entities**: Identify every named entity, product, technology,
   concept, or topic mentioned in the query. Use canonical forms.

2. **Classify intent** — pick exactly ONE:
   | Intent | When to use |
   |---|---|
   | FACTUAL | User wants a specific fact or definition |
   | RELATIONSHIP | User asks how things connect or relate |
   | COMPARISON | User wants to compare two or more things |
   | PROCEDURAL | User wants step-by-step instructions |
   | EXPLORATORY | Open-ended question, wants broad information |

3. **Plan Retrieval Strategy**:
   - `semantic_search_needed`: True if matching natural language terms, concepts, or general text content is helpful.
   - `graph_search_needed`: True if the query targets relationships, connectivity, or specific entities that exist in the knowledge graph.
   - `both_needed`: True if BOTH semantic match and relationship traversal are required to answer the query fully.
   - `traversal_depth`: 1 for simple entity lookups, 2 for relationship connectivity or multi-hop questions.

### Rules
- If the query is complex or mentions multiple entities, set both_needed = True.
- If unsure, default to both semantic and graph being needed (both_needed = True, semantic_search_needed = True, graph_search_needed = True).
- For relationship queries, always set graph_search_needed = True.
"""

HUMAN_PROMPT = """\
User query: {query}

Analyze this query and produce the structured retrieval plan.\
"""


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT
# ─────────────────────────────────────────────────────────────────────────────

class QueryPlannerAgent:
    """Analyzes a user query and plans the retrieval strategy (semantic, graph, or both)."""

    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "MISTRAL_API_KEY not found. Add it to your .env file."
            )

        llm = ChatMistralAI(
            model="mistral-small-latest",
            api_key=api_key,
            temperature=0.0,
            max_tokens=1024,
        )

        structured_llm = llm.with_structured_output(QueryAnalysis)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  HUMAN_PROMPT),
        ])

        self._chain = prompt | structured_llm

    def plan(self, query: str) -> QueryAnalysis:
        """Analyze a user query and determine the execution plan.

        Parameters
        ----------
        query:
            The user's natural-language question.

        Returns
        -------
        QueryAnalysis
            Contains query_entities, query_intent, and retrieval_plan.
        """
        logger.info("Stage 1: Running Query Planner Agent …")

        try:
            result = self._chain.invoke({"query": query})
            # Ensure consistency: if both_needed is True, then both search types must be True.
            if result.retrieval_plan.both_needed:
                result.retrieval_plan.semantic_search_needed = True
                result.retrieval_plan.graph_search_needed = True
            elif result.retrieval_plan.semantic_search_needed and result.retrieval_plan.graph_search_needed:
                result.retrieval_plan.both_needed = True

            logger.info(
                f"  Plan determined: semantic={result.retrieval_plan.semantic_search_needed}, "
                f"graph={result.retrieval_plan.graph_search_needed}, "
                f"both={result.retrieval_plan.both_needed}"
            )
            return result
        except Exception as e:
            logger.warning(f"  Query planning failed: {e}. Using fallback defaults.")
            return QueryAnalysis(
                query_entities=[query],
                query_intent=QueryIntent.FACTUAL,
                retrieval_plan=AgenticRetrievalPlan(
                    semantic_search_needed=True,
                    graph_search_needed=True,
                    both_needed=True,
                    traversal_depth=2,
                )
            )
