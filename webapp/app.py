"""
webapp/app.py  — UPDATED

Changes vs original:
  - if __name__ == "__main__": now uses threaded=True so background
    ingestion thread + browser polling requests run concurrently.
  - Added reloader_type="stat" and extra_files when debug mode is on
    so `python -m webapp.app` also gets the safe reloader.

All other code is identical to the original.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request

from webapp import config, job_store, pipeline_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024


# ── Page routes ───────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html", active="index")


@app.get("/upload")
def upload_page():
    return render_template("upload.html", active="upload", stages=job_store.PIPELINE_STAGES)


@app.get("/chat")
def chat_page():
    return render_template("chat.html", active="chat")


@app.get("/graph")
def graph_page():
    return render_template("graph.html", active="graph")


@app.get("/dashboard")
def dashboard_page():
    return render_template("dashboard.html", active="dashboard")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> Any:
    return jsonify({"status": "ok"})


# ── Upload + ingest ───────────────────────────────────────────────────────────

@app.post("/api/upload")
def upload_pdf() -> Any:
    file = request.files.get("file")
    if file is None or not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"detail": "Only PDF files are supported."}), 400

    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = config.UPLOADS_DIR / file.filename
    file.save(dest_path)

    job_id = job_store.create_job(filename=file.filename)

    thread = threading.Thread(
        target=pipeline_service.run_ingestion,
        args=(job_id, dest_path),
        daemon=True,
    )
    thread.start()

    logger.info("Queued ingestion job %s for %s", job_id, file.filename)
    return jsonify({"job_id": job_id, "filename": file.filename, "status": "queued"})


@app.get("/api/status/<job_id>")
def get_status(job_id: str) -> Any:
    job = job_store.get(job_id)
    if job is None:
        return jsonify({"detail": "Job not found."}), 404
    return jsonify(job)


# ── Documents + graph ─────────────────────────────────────────────────────────

@app.get("/api/documents")
def list_documents() -> Any:
    latest_complete = job_store.get_latest_complete()
    if latest_complete is None or not config.CHUNKS_PATH.exists():
        return jsonify({"documents": []})

    chunks = _load_json(config.CHUNKS_PATH, default=[])
    entities_wrapper = _load_json(config.ENTITIES_PATH, default={})
    relationships_wrapper = _load_json(config.RELATIONSHIPS_PATH, default={})

    doc_info = {
        "filename": latest_complete.get("filename"),
        "ingested_at": latest_complete.get("updated_at"),
        "chunk_count": len(chunks) if isinstance(chunks, list) else 0,
        "entity_count": entities_wrapper.get("total_entities_extracted", 0),
        "relationship_count": relationships_wrapper.get("total_relationships_extracted", 0),
        "entities_by_type": entities_wrapper.get("entities_by_type", {}),
        "relationships_by_type": relationships_wrapper.get("relationships_by_type", {}),
    }
    return jsonify({"documents": [doc_info]})


@app.get("/api/graph")
def get_graph() -> Any:
    if not config.ENTITIES_PATH.exists() or not config.RELATIONSHIPS_PATH.exists():
        return jsonify({"detail": "No graph data yet -- upload and ingest a PDF first."}), 404

    entities_wrapper = _load_json(config.ENTITIES_PATH, default={})
    relationships_wrapper = _load_json(config.RELATIONSHIPS_PATH, default={})

    return jsonify(
        {
            "entities": entities_wrapper.get("entities", []),
            "relationships": relationships_wrapper.get("relationships", []),
        }
    )


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/api/query")
def query() -> Any:
    body = request.get_json(silent=True) or {}
    user_query = (body.get("query") or "").strip()
    if not user_query:
        return jsonify({"detail": "query is required."}), 400

    if not config.CHUNKS_PATH.exists():
        return jsonify({"detail": "No document ingested yet — upload a PDF first."}), 400

    pipeline = pipeline_service.get_retrieval_pipeline()

    try:
        retrieval_result = pipeline.retrieve(user_query)
    except Exception as exc:
        logger.exception("Retrieval failed")
        return jsonify({"detail": f"Retrieval failed: {exc}"}), 500

    evidence_chunks = retrieval_result.evidence_chunks or []
    primary_scores = [c["score"] for c in evidence_chunks if not c.get("is_neighbor")]

    if not primary_scores:
        return jsonify(_out_of_scope_payload(retrieval_result))

    top_raw_score = max(primary_scores)
    top_normalized = 1.0 / (1.0 + math.exp(-top_raw_score))

    if top_normalized < config.OUT_OF_SCOPE_THRESHOLD:
        return jsonify(_out_of_scope_payload(retrieval_result))

    synth_chain = pipeline_service.get_synthesis_chain()
    try:
        synth_result = synth_chain.synthesize(user_query, retrieval_result)
    except Exception as exc:
        logger.exception("Synthesis failed")
        return jsonify({"detail": f"Answer synthesis failed: {exc}"}), 500

    return jsonify(
        {
            "out_of_scope": False,
            "answer": synth_result.answer,
            "evidence": [e.model_dump(mode="json") for e in synth_result.evidence],
            "reasoning_path": synth_result.reasoning_path,
            "confidence": synth_result.confidence,
            "evidence_chunks": evidence_chunks,
            "graph_paths": retrieval_result.graph_paths,
            "source_entities": retrieval_result.source_entities,
            "retrieval_metadata": retrieval_result.retrieval_metadata,
        }
    )


def _out_of_scope_payload(retrieval_result: Any) -> Dict[str, Any]:
    return {
        "out_of_scope": True,
        "answer": config.OUT_OF_SCOPE_MESSAGE,
        "evidence": [],
        "reasoning_path": [],
        "confidence": "Low",
        "evidence_chunks": getattr(retrieval_result, "evidence_chunks", []) or [],
        "graph_paths": getattr(retrieval_result, "graph_paths", []) or [],
        "source_entities": getattr(retrieval_result, "source_entities", []) or [],
        "retrieval_metadata": getattr(retrieval_result, "retrieval_metadata", {}) or {},
    }


def _load_json(path: Path, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── SAFE RELOADER: watch only webapp/ source files ────────────────────
    # If debug mode is on, use stat reloader (never scans venv) so
    # PyTorch cache changes don't restart the server and wipe the job store.
    if config.FLASK_DEBUG:
        webapp_dir = Path(__file__).resolve().parent
        watch_files = [
            str(f) for f in webapp_dir.rglob("*")
            if f.suffix in (".py", ".html", ".css", ".js")
        ]
        from werkzeug.serving import run_simple
        run_simple(
            hostname      = config.FLASK_HOST,
            port          = config.FLASK_PORT,
            application   = app,
            use_reloader  = True,
            use_debugger  = True,
            threaded      = True,
            extra_files   = watch_files,
            reloader_type = "stat",   # never scans .venv/site-packages
        )
    else:
        # Production-safe: no reloader, threaded
        app.run(
            host     = config.FLASK_HOST,
            port     = config.FLASK_PORT,
            debug    = False,
            threaded = True,
        )