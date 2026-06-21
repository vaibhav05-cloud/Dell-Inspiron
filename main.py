"""
Entry point for the Dell FutureMinds project.
Runs end-to-end GraphRAG pipeline: Ingestion, Knowledge Construction, and Query-Time Retrieval/Synthesis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
import subprocess

# Reconfigure standard output streams to use UTF-8 encoding on Windows to prevent UnicodeEncodeErrors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# Load environmental variables
load_dotenv()

# Setup logging directory
os.makedirs("output", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/pipeline.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Imports from existing modules
from parser.pdf_parser import PDFParser
from processor.multimodal_processor import MultimodalProcessor
from semantic_chunker import SemanticTextChunker
from graph_builder.connection import Neo4jConnection
from graph_builder.ingestion import GraphIngestionService
from graph_builder.validator import validate_inputs, validate_graph
from retriever.semantic_retriever import SemanticRetrievalAgent
from retriever.pipeline import RetrievalPipeline
from synthesizer.synthesis_chain import AnswerSynthesisChain


def clean_previous_run():
    """Completely remove old outputs so every PDF starts from scratch."""
    output_dir = Path("output")
    if not output_dir.exists():
        return
        
    files_to_delete = [
        "chunks.json", 
        "entities.json", 
        "relationships.json", 
        "entities_neo4j.json", 
        "relationships_neo4j.json", 
        ".graph_ingested", 
        ".pinecone_ingested"
    ]
    for filename in files_to_delete:
        file_path = output_dir / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted: {file_path}")
            
    images_dir = output_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
        logger.info("Deleted images folder")
        
    faiss_dir = output_dir / "faiss_index"
    if faiss_dir.exists():
        shutil.rmtree(faiss_dir)
        logger.info("Deleted FAISS index")
        
    for json_file in output_dir.glob("*_parsed.json"):
        json_file.unlink()
    for json_file in output_dir.glob("*_processed.json"):
        json_file.unlink()
        
    logger.info("✓ Previous run cleaned successfully")


def run_ingestion(pdf_path: str, force: bool = False) -> dict[str, float]:
    """Execute the Document Ingestion and Knowledge Construction pipeline with checkpointing and timing."""
    clean_previous_run()
    timings = {}
    
    # Paths for artifacts and flags
    chunks_path = Path("output/chunks.json")
    entities_path = Path("output/entities_neo4j.json")
    relationships_path = Path("output/relationships_neo4j.json")
    graph_flag_path = Path("output/.graph_ingested")
    pinecone_flag_path = Path("output/.pinecone_ingested")
    
    logger.info("=" * 60)
    logger.info("  RUNNING GRAPH-RAG INGESTION PIPELINE")
    logger.info("=" * 60)
    
    # ── Step 1-3: Parsing, Multimodal Processor, and Chunking ─────────────
    # If chunks.json exists, we can skip Parsing, Multimodal Processing, and Chunking
    if chunks_path.exists() and not force:
        logger.info(f"Checkpoint hit: '{chunks_path}' already exists. Skipping Parsing, Multimodal Processing, and Chunking.")
        timings["Parsing"] = 0.0
        timings["Chunking"] = 0.0
    else:
        # Step 1 & 2: PDF Parsing & Multimodal Extraction (measured together as Parsing)
        t_parse_start = time.perf_counter()
        logger.info(f"Starting Parsing & Multimodal Processor on '{pdf_path}'...")
        try:
            parser = PDFParser(output_dir="output")
            parsed_doc = parser.parse(pdf_path)
            parsed_path = parser.save_to_json(parsed_doc)
            
            mm_processor = MultimodalProcessor()
            processed_path = mm_processor.process_file(parsed_path)
            t_parse_end = time.perf_counter()
            timings["Parsing"] = (t_parse_end - t_parse_start) * 1000
            logger.info(f"Parsing and Multimodal Processing completed in {timings['Parsing']:.1f} ms.")
        except Exception as exc:
            logger.error(f"FATAL: Parsing stage failed: {exc}")
            raise exc
            
        # Step 3: Semantic Chunking
        t_chunk_start = time.perf_counter()
        logger.info("Starting Semantic Chunking...")
        try:
            chunker = SemanticTextChunker()
            data = chunker.load_processed_json(processed_path)
            
            text_chunks, next_id = chunker.build_text_chunks(data)
            table_chunks, next_id = chunker.build_table_chunks(data, next_id)
            image_chunks, next_id = chunker.build_image_chunks(data, next_id)
            
            all_chunks = text_chunks + table_chunks + image_chunks
            if hasattr(chunker, "deduplicate_chunks"):
                all_chunks = chunker.deduplicate_chunks(all_chunks)
                
            chunker.save_chunks(all_chunks, str(chunks_path))
            
            t_chunk_end = time.perf_counter()
            timings["Chunking"] = (t_chunk_end - t_chunk_start) * 1000
            logger.info(f"Semantic Chunking completed in {timings['Chunking']:.1f} ms. Total chunks: {len(all_chunks)}.")
        except Exception as exc:
            logger.error(f"FATAL: Semantic Chunking stage failed: {exc}")
            raise exc
        
        
        logger.info("Starting Graph Pipeline (run_graph.py)...")
        subprocess.run(
            [sys.executable, "run_graph.py"],
            check=True
        )
        
        logger.info("Graph Pipeline completed.")
        

    # ── Step 4: Entity Extraction ──────────────────────────────────────────
    

    # ── Step 6: Neo4j Graph Builder ────────────────────────────────────────
    if graph_flag_path.exists() and not force:
        logger.info(f"Checkpoint hit: '{graph_flag_path}' exists. Skipping Neo4j Graph Ingestion.")
        timings["Graph Construction"] = 0.0
    else:
        t_graph_start = time.perf_counter()
        logger.info("Starting Neo4j Graph Ingestion...")
        try:
            with open(entities_path, "r", encoding="utf-8") as f:
                entities_data = json.load(f)
            with open(relationships_path, "r", encoding="utf-8") as f:
                rels_data = json.load(f)
                
            entities = entities_data["entities"] if isinstance(entities_data, dict) and "entities" in entities_data else entities_data
            relationships = rels_data["relationships"] if isinstance(rels_data, dict) and "relationships" in rels_data else rels_data

            # Pre-ingestion validation
            validation = validate_inputs(entities, relationships)
            if not validation["valid"]:
                raise ValueError(f"Pre-ingestion validation failed: {validation['errors']}")

            # Connect and Ingest
            with Neo4jConnection() as conn:
                service = GraphIngestionService(conn)
                stats = service.ingest_all(entities, relationships)
                
                # Post-ingestion verification
                with conn.session() as session:
                    validate_graph(session)
            
            # Write checkpoint flag
            with open(graph_flag_path, "w", encoding="utf-8") as f:
                f.write("ingested")
            
            t_graph_end = time.perf_counter()
            timings["Graph Construction"] = (t_graph_end - t_graph_start) * 1000
            logger.info(f"Neo4j Graph Construction completed in {timings['Graph Construction']:.1f} ms.")
        except Exception as exc:
            logger.error(f"FATAL: Graph Ingestion stage failed: {exc}")
            raise exc

    # ── Step 7: Pinecone (FAISS) Embedding Storage ──────────────────────────
    if pinecone_flag_path.exists() and not force:
        logger.info(f"Checkpoint hit: '{pinecone_flag_path}' exists. Skipping Embedding Storage.")
        timings["Embedding Storage"] = 0.0
    else:
        t_embed_start = time.perf_counter()
        logger.info("Starting FAISS Vector Indexing...")
        try:
            agent = SemanticRetrievalAgent(
                chunks_path=str(chunks_path),
                index_dir="output/faiss_index"
            )
            # Invoke inner initialization method to build and cache FAISS index
            agent._ensure_initialized()
            
            # Write checkpoint flag
            with open(pinecone_flag_path, "w", encoding="utf-8") as f:
                f.write("ingested")
                
            t_embed_end = time.perf_counter()
            timings["Embedding Storage"] = (t_embed_end - t_embed_start) * 1000
            logger.info(f"Embedding Storage completed in {timings['Embedding Storage']:.1f} ms.")
        except Exception as exc:
            logger.error(f"FATAL: Embedding Storage stage failed: {exc}")
            raise exc
            
    logger.info("=" * 60)
    logger.info("  INGESTION PIPELINE COMPLETE")
    logger.info("=" * 60)
    for stage, ms in timings.items():
        logger.info(f"  {stage:<30s}: {ms:>8.1f} ms")
    logger.info("=" * 60)
    
    return timings


def run_query(query: str, rerank_top_k: int = 15, token_budget: int = 4000) -> None:
    """Execute the query-time retrieval and answer synthesis pipeline."""
    # Ensure Ingestion was run first
    chunks_path = Path("output/chunks.json")
    entities_path = Path("output/entities.json")
    relationships_path = Path("output/relationships.json")
    graph_flag_path = Path("output/.graph_ingested")
    pinecone_flag_path = Path("output/.pinecone_ingested")
    
    missing_artifacts = []
    if not chunks_path.exists(): missing_artifacts.append(str(chunks_path))
    if not entities_path.exists(): missing_artifacts.append(str(entities_path))
    if not relationships_path.exists(): missing_artifacts.append(str(relationships_path))
    if not graph_flag_path.exists(): missing_artifacts.append(str(graph_flag_path))
    if not pinecone_flag_path.exists(): missing_artifacts.append(str(pinecone_flag_path))
    
    if missing_artifacts:
        logger.error(f"Query Pipeline cannot run. Ingestion artifacts/flags missing: {missing_artifacts}")
        logger.error("Please run the ingestion pipeline first with: python main.py --ingest")
        sys.exit(1)
        
    logger.info("Initializing Retrieval Pipeline and Answer Synthesis Chain...")
    
    try:
        pipeline = RetrievalPipeline(
            chunks_path=str(chunks_path),
            entities_path=str(entities_path),
            index_dir="output/faiss_index",
            rerank_top_k=rerank_top_k,
            token_budget=token_budget,
        )
        synthesis_chain = AnswerSynthesisChain()
    except Exception as exc:
        logger.error(f"Failed to initialize query services: {exc}")
        sys.exit(1)

    # ── Retrieval and Answer Synthesis ────────────────────────────────────
    t_query_start = time.perf_counter()
    try:
        # Step 1: Retrieval (Query Understanding, Semantic/Graph Search, Fusion, Evidence, Context)
        t_ret_start = time.perf_counter()
        retrieval_result = pipeline.retrieve(query)
        t_ret_end = time.perf_counter()
        
        # Step 2: Answer Generation (Synthesis Chain)
        t_gen_start = time.perf_counter()
        synthesis_result = synthesis_chain.synthesize(query, retrieval_result)
        t_gen_end = time.perf_counter()
        
        t_query_end = time.perf_counter()
        
        # Extract individual agent timings from RetrievalPipeline
        stage_timings = retrieval_result.retrieval_metadata.get("stage_timings", [])
        reranking_ms = 0.0
        retrieval_ms = 0.0
        for st in stage_timings:
            if st["stage_name"] == "Re-ranking Agent":
                reranking_ms = st["duration_ms"]
            else:
                retrieval_ms += st["duration_ms"]
        
        # Answer Generation duration
        answer_gen_ms = (t_gen_end - t_gen_start) * 1000
        
        # Total Query pipeline time
        total_query_ms = (t_query_end - t_query_start) * 1000
        
    except Exception as exc:
        logger.error(f"Query Execution failed: {exc}")
        pipeline.close()
        sys.exit(1)
        
    pipeline.close()

    # ── Print Final JSON Output ───────────────────────────────────────────
    # Format evidence as requested: page_number as string
    final_output = {
        "answer": synthesis_result.answer,
        "evidence": [
            {
                "chunk_id": ev.chunk_id,
                "page_number": str(ev.page_number),
                "source_document": ev.source_document
            }
            for ev in synthesis_result.evidence
        ],
        "reasoning_path": synthesis_result.reasoning_path,
        "confidence": synthesis_result.confidence
    }
    
    print("\n" + "=" * 60)
    print("  FINAL PIPELINE OUTPUT")
    print("=" * 60)
    print(json.dumps(final_output, indent=2, ensure_ascii=False))
    print("=" * 60)
    
    # ── Print Timings Summary ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  QUERY-TIME PIPELINE TIMINGS")
    logger.info("=" * 60)
    logger.info(f"  Retrieval          : {retrieval_ms:>8.1f} ms")
    logger.info(f"  Re-ranking         : {reranking_ms:>8.1f} ms")
    logger.info(f"  Answer Generation  : {answer_gen_ms:>8.1f} ms")
    logger.info(f"  Total Query Process: {total_query_ms:>8.1f} ms")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End GraphRAG Production Pipeline Orchestrator"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ingest",
        action="store_true",
        help="Run document ingestion & knowledge construction pipeline."
    )
    group.add_argument(
        "--query",
        type=str,
        help="A single query to retrieve and synthesize an answer for."
    )
    group.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive REPL mode for querying."
    )
    
    parser.add_argument(
        "--pdf",
        default=None,
        help="Path to the PDF file for ingestion (default: None)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass checkpoints and force re-running of all stages during ingestion."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of results after re-ranking (default: 15)"
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=4000,
        help="Token budget for context building (default: 4000)"
    )
    args = parser.parse_args()

    # Ensure API Key is present
    if not os.getenv("MISTRAL_API_KEY"):
        logger.error("MISTRAL_API_KEY environment variable is not set. Please check your .env file.")
        sys.exit(1)

    if args.ingest:
        if args.pdf:
            run_ingestion(args.pdf, force=args.force)
        else:
            pdf_folder = Path("data/pdfs")
            pdf_files = list(pdf_folder.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError("No PDF found inside data/pdfs")
            pdf_path = str(pdf_files[0])
            logger.info(f"Using PDF: {pdf_path}")
            run_ingestion(pdf_path, force=args.force)
        
    elif args.query:
        run_query(args.query, rerank_top_k=args.top_k, token_budget=args.token_budget)
        
    elif args.interactive:
        print("\n" + "=" * 40)
        print("  GraphRAG End-to-End Pipeline REPL")
        print("  Type 'quit' or 'exit' to stop")
        print("=" * 40 + "\n")
        
        while True:
            try:
                query = input("Query > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
                
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Exiting.")
                break
                
            run_query(query, rerank_top_k=args.top_k, token_budget=args.token_budget)
            print()


if __name__ == "__main__":
    main()