"""
webapp/config.py

Central configuration for the Flask app: paths into the existing
pipeline's `output/` and `data/pdfs/` folders, plus tunable thresholds
exposed as environment variables, and the Flask server settings.

IMPORTANT: this file assumes it lives at <project_root>/webapp/config.py,
i.e. as a sibling of your existing `parser/`, `processor/`,
`semantic_chunker.py`, `extractor/`, `relationship/`, `graph_builder/`,
`retriever/`, `synthesizer/` packages. webapp/app.py adds <project_root>
to sys.path on startup so those packages remain importable exactly as
your existing CLI runners (extractor/runner.py etc.) already do.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────

WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent

# Load the same .env your pipeline scripts already use (MISTRAL_API_KEY,
# NEO4J_* vars, etc.) regardless of the working directory Flask is
# started from.
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR = PROJECT_ROOT / "output"
UPLOADS_DIR = PROJECT_ROOT / "data" / "pdfs"

CHUNKS_PATH = OUTPUT_DIR / "chunks.json"
ENTITIES_PATH = OUTPUT_DIR / "entities.json"
RELATIONSHIPS_PATH = OUTPUT_DIR / "relationships.json"
FAISS_INDEX_DIR = OUTPUT_DIR / "faiss_index"

# ── Retrieval / synthesis tuning ────────────────────────────────────────

SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "30"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "15"))
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "4000"))
GRAPH_INGEST_BATCH_SIZE = int(os.getenv("GRAPH_INGEST_BATCH_SIZE", "100"))

# Sigmoid-normalized cross-encoder score below which a query is treated
# as OUT OF SCOPE for the ingested document, short-circuiting before the
# synthesis LLM call. Raw cross-encoder scores are unbounded logits;
# sigmoid(raw_score) maps them to (0, 1). sigmoid(0) = 0.5 is "borderline
# relevant". 0.30 is a reasonably permissive default -- lower it to allow
# more borderline answers through, raise it to be stricter about
# refusing out-of-scope questions. Tune against your own document plus a
# few known off-topic test questions.
OUT_OF_SCOPE_THRESHOLD = float(os.getenv("OUT_OF_SCOPE_THRESHOLD", "0.30"))

OUT_OF_SCOPE_MESSAGE = (
    "I couldn't find anything in the uploaded document that addresses "
    "this question. Try rephrasing, or ask something covered in the "
    "source PDF."
)

# ── Upload / Flask server settings ──────────────────────────────────────

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").strip().lower() in ("1", "true", "yes")
