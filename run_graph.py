"""
run_graph.py  —  Steps 5–8  (complete pipeline to Neo4j-ready output)

Usage:
    uv run python run_graph.py

Steps:
  5   — Entity Extraction        (spaCy + tech patterns)
  5b  — Entity Resolution        (merge Dell/DELL/Dell Inc → Dell)
  6   — Relationship Extraction  (Mistral, batched 7 pairs/call)
  6b  — Co-occurrence Extraction (fast, no LLM)
  6c  — Prune Isolated Entities
  7   — Knowledge Graph          (NetworkX + pyvis HTML)
  8   — Neo4j Adapter            (convert to teammate's schema)
         → output/entities_neo4j.json
         → output/relationships_neo4j.json

After this completes, run teammate's Neo4j ingestion:
    python -m graph_builder.runner \
        --entities output/entities_neo4j.json \
        --relationships output/relationships_neo4j.json
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from graph.hybrid_entity_extractor import HybridEntityExtractor
from graph.relationship_extractor  import RelationshipExtractor
from graph.cooccurrence_extractor  import extract_cooccurrence
from graph.entity_resolver         import EntityResolver
from output_adapter          import adapt_for_neo4j
from graph.knowledge_graph         import KnowledgeGraph


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
CHUNKS_PATH = "output/chunks.json"
OUTPUT_DIR  = Path("output")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")


def load_chunks(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("chunks", [])


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved → {path}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    if not MISTRAL_KEY:
        raise ValueError("MISTRAL_API_KEY not found in .env")

    # ── Load chunks ───────────────────────────────────────────────────────────
    logger.info(f"Loading chunks from {CHUNKS_PATH}")
    chunks = load_chunks(CHUNKS_PATH)
    logger.info(f"Loaded {len(chunks)} chunks")

    # ── Step 5: Entity Extraction ─────────────────────────────────────────────
    logger.info("\n── STEP 5: Entity Extraction ──")
    entities           = HybridEntityExtractor().extract_all(chunks)
    entities_raw_count = len(entities)
    logger.info(f"Extracted {entities_raw_count} raw entities")

    # ── Step 5b: Entity Resolution ────────────────────────────────────────────
    logger.info("\n── STEP 5b: Entity Resolution ──")
    resolver              = EntityResolver()
    entities, name_map    = resolver.resolve(entities)
    entities_resolved_count = len(entities)

    # ── Step 6: Relationship Extraction (Mistral — batched) ───────────────────
    logger.info("\n── STEP 6: Relationship Extraction (Mistral — batched) ──")
    rel_extractor         = RelationshipExtractor(api_key=MISTRAL_KEY)
    mistral_relationships = rel_extractor.extract_all(entities, chunks)
    mistral_relationships = resolver.apply_to_relationships(
        mistral_relationships, name_map
    )

    # ── Step 6b: Co-occurrence Extraction ────────────────────────────────────
    logger.info("\n── STEP 6b: Co-occurrence Extraction ──")
    existing_pairs = {
        tuple(sorted([r["source"].lower(), r["target"].lower()]))
        for r in mistral_relationships
    }
    cooccur_relationships = extract_cooccurrence(chunks, entities, existing_pairs)
    cooccur_relationships = resolver.apply_to_relationships(
        cooccur_relationships, name_map
    )

    all_relationships = mistral_relationships + cooccur_relationships
    logger.info(f"Total relationships: {len(all_relationships)}")

    # Save original format (for reference / FAISS retriever)
    save_json(all_relationships, OUTPUT_DIR / "relationships.json")

    # ── Step 6c: Prune Isolated Entities ─────────────────────────────────────
    logger.info("\n── STEP 6c: Pruning Isolated Entities ──")
    connected_names = set()
    for rel in all_relationships:
        connected_names.add(rel["source"].lower())
        connected_names.add(rel["target"].lower())

    entities_before = len(entities)
    entities        = [e for e in entities if e["name"].lower() in connected_names]
    entities_pruned = entities_before - len(entities)
    logger.info(f"Pruned {entities_pruned} isolated entities")

    # Save original format (for reference / FAISS retriever)
    save_json(entities, OUTPUT_DIR / "entities.json")

    # ── Step 7: Knowledge Graph ───────────────────────────────────────────────
    logger.info("\n── STEP 7: Knowledge Graph (NetworkX + HTML) ──")
    kg = KnowledgeGraph()
    kg.build(entities, all_relationships)
    kg.save_to_json(str(OUTPUT_DIR / "graph_data.json"))
    kg.save_visualization(str(OUTPUT_DIR / "knowledge_graph.html"))
    stats = kg.get_stats()

    # ── Step 8: Neo4j Format Adapter ─────────────────────────────────────────
    logger.info("\n── STEP 8: Adapting output for Neo4j ingestion ──")
    neo4j_paths = adapt_for_neo4j(entities, all_relationships, chunks)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  STEPS 5 → 8 COMPLETE")
    print("─" * 65)
    print(f"  Entities extracted       : {entities_raw_count}")
    print(f"  Entities after resolution: {entities_resolved_count}  (merged duplicates)")
    print(f"  Entities after pruning   : {len(entities)}")
    print("─" * 65)
    print(f"  Mistral relationships    : {len(mistral_relationships)}")
    print(f"  Co-occurrence relations  : {len(cooccur_relationships)}")
    print(f"  Total relationships      : {len(all_relationships)}")
    print("─" * 65)
    print(f"  Graph nodes              : {stats['nodes']}")
    print(f"  Graph edges              : {stats['edges']}")
    print(f"  Edges per node (avg)     : {stats['edges'] / max(stats['nodes'], 1):.2f}")
    print("─" * 65)
    print(f"  entities.json            → output/entities.json")
    print(f"  relationships.json       → output/relationships.json")
    print(f"  knowledge_graph.html     → output/knowledge_graph.html")
    print("─" * 65)
    print(f"  [NEO4J] entities_neo4j.json      → {neo4j_paths['entities_neo4j']}")
    print(f"  [NEO4J] relationships_neo4j.json → {neo4j_paths['relationships_neo4j']}")
    print("─" * 65)
    print("\n  NEXT STEP — Run Neo4j ingestion:")
    print("  python -m graph_builder.runner \\")
    print("      --entities output/entities_neo4j.json \\")
    print("      --relationships output/relationships_neo4j.json")
    print()
    print("  THEN — Query GraphRAG:")
    print("  python -m retriever.runner --interactive")
    print("─" * 65)


if __name__ == "__main__":
    main()