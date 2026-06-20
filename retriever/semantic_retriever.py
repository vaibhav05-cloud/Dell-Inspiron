"""
Stage 2 — Semantic Retrieval Agent.

Uses FAISS (local) vector store backend to perform similarity search against user queries
using the same embedding model used in the chunker (all-MiniLM-L6-v2). If Pinecone
credentials were provided, it could connect to Pinecone; otherwise, it utilizes FAISS.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from retriever.schema import CandidateSource, RetrievalCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNKS_PATH = "output/chunks.json"
DEFAULT_INDEX_DIR = "output/faiss_index"


# ─────────────────────────────────────────────────────────────────────────────
#  SEMANTIC RETRIEVAL AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SemanticRetrievalAgent:
    """FAISS-based semantic retrieval agent over document chunks.

    On first use, builds a FAISS index from chunks.json and caches it
    to disk. Subsequent instantiations reload from disk for speed.
    """

    def __init__(
        self,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        index_dir: str = DEFAULT_INDEX_DIR,
    ):
        self._chunks_path = chunks_path
        self._index_dir = index_dir
        self._vectorstore = None
        self._chunks_lookup: Dict[str, dict] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-initialize: load or build the FAISS index."""
        if self._initialized:
            return

        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        logger.info("Stage 2: Initializing Semantic Retrieval Agent …")

        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
        )

        index_path = Path(self._index_dir)

        # Try loading cached index
        if index_path.exists() and (index_path / "index.faiss").exists():
            logger.info(f"  Loading cached FAISS index from {index_path}")
            self._vectorstore = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            self._load_chunks_lookup()
            self._initialized = True
            logger.info("  FAISS index loaded from cache ✓")
            return

        # Build new index
        logger.info(f"  Building FAISS index from {self._chunks_path} …")
        self._load_chunks_lookup()

        documents = []
        for chunk in self._chunks_lookup.values():
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk.get("page_number", 0),
                    "section_name": chunk.get("section_name", ""),
                    "source_file": chunk.get("source_file", ""),
                    "chunk_type": chunk.get("chunk_type", "text"),
                },
            )
            documents.append(doc)

        logger.info(f"  Indexing {len(documents)} documents …")
        self._vectorstore = FAISS.from_documents(documents, embeddings)

        # Cache to disk
        index_path.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(index_path))
        logger.info(f"  FAISS index cached to {index_path} ✓")

        self._initialized = True

    def _load_chunks_lookup(self) -> None:
        """Load chunks.json into a dict keyed by chunk_id."""
        if self._chunks_lookup:
            return

        p = Path(self._chunks_path)
        if not p.exists():
            raise FileNotFoundError(f"Chunks file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        self._chunks_lookup = {c["chunk_id"]: c for c in chunks}
        logger.info(f"  Loaded {len(self._chunks_lookup)} chunks for lookup")

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
    ) -> List[RetrievalCandidate]:
        """Perform semantic search and return ranked candidates.

        Parameters
        ----------
        query:
            The user's natural-language question.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        List[RetrievalCandidate]
            Candidates sorted by similarity score descending.
        """
        self._ensure_initialized()

        logger.info(f"  Semantic Retrieval Agent: Searching (top_k={top_k}) …")

        results = self._vectorstore.similarity_search_with_score(
            query, k=top_k
        )

        candidates = []
        for doc, score in results:
            # FAISS returns L2 distance; convert to similarity (lower = better)
            similarity = 1.0 / (1.0 + score)

            meta = doc.metadata
            candidates.append(
                RetrievalCandidate(
                    chunk_id=meta.get("chunk_id", ""),
                    content=doc.page_content,
                    page_number=meta.get("page_number", 0),
                    section_name=meta.get("section_name", ""),
                    source_file=meta.get("source_file", ""),
                    source=CandidateSource.SEMANTIC,
                    similarity_score=similarity,
                )
            )

        logger.info(f"  Semantic Retrieval Agent: Retrieved {len(candidates)} candidates ✓")
        return candidates

    @property
    def chunks_lookup(self) -> Dict[str, dict]:
        """Access the chunks lookup dict (loads if needed)."""
        self._load_chunks_lookup()
        return self._chunks_lookup
