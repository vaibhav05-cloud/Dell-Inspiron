import re

TECH_TERMS = [
    "Neo4j",
    "GraphRAG",
    "LangChain",
    "Mistral AI",
    "OpenAI",
    "FAISS",
    "ChromaDB",
    "LlamaIndex",
    "Milvus",
    "Weaviate"
]


def extract_technical_entities(text):

    entities = []

    for term in TECH_TERMS:

        pattern = re.escape(term)

        if re.search(pattern, text, re.IGNORECASE):

            entities.append({
                "name": term,
                "type": "TECHNOLOGY"
            })

    return entities