"""
webapp/job_store.py

Simple in-memory, thread-safe store for ingestion job progress. The
ingestion pipeline (webapp/pipeline_service.py::run_ingestion) runs on a
background daemon thread started from the /api/upload route, while the
browser polls /api/status/<job_id> every 2s from a *different* request
(handled by Flask's own thread/worker) -- so every read/write here goes
through a lock to avoid races between the writer thread and the poller.

NOTE on scope (see README "Known limitations"): this is an IN-MEMORY
store. Restarting the Flask process loses job history -- the actual
pipeline outputs (output/*.json, the FAISS index, Neo4j data) are
unaffected, since those are written to disk / the database by
pipeline_service.py independently of this module. If you need job
history to survive restarts, swap this module's internals for a real
store (SQLite, Redis, etc.) -- the function signatures below are the
only contract the rest of the app depends on, so the swap is contained
to this one file.

Job shape (the dict returned by get_status / stored per job_id)
-----------------------------------------------------------------
{
    "job_id": str,
    "filename": str,
    "status": "queued" | "running" | "complete" | "error",
    "stage": str,            # one of PIPELINE_STAGES keys, or "queued"/"done"
    "stage_label": str,      # human-readable label for the current stage
    "progress": int,         # 0-100
    "error": str | None,
    "created_at": float,     # unix timestamp
    "updated_at": float,     # unix timestamp
    "log": [{"timestamp": float, "message": str}, ...],
}
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────
#  PIPELINE STAGES
#  Single source of truth for stage keys/labels/progress used by both
#  the upload page's stepper (webapp/app.py passes this to the
#  upload.html template) and pipeline_service.py's job_store.set_stage()
#  calls. Order matches STAGE_ORDER in webapp/static/js/upload.js exactly
#  -- if you add/reorder a stage here, update that JS array too.
# ─────────────────────────────────────────────────────────────────────────

PIPELINE_STAGES: List[tuple] = [
    ("parsing", "Parsing PDF"),
    ("multimodal_enrichment", "Multimodal Enrichment"),
    ("chunking", "Semantic Chunking"),
    ("entity_extraction", "Entity Extraction"),
    ("relationship_extraction", "Relationship Extraction"),
    ("graph_ingestion", "Neo4j Graph Ingestion"),
    ("indexing", "Indexing (FAISS)"),
    ("done", "Complete"),
]

_STAGE_LABELS: Dict[str, str] = dict(PIPELINE_STAGES)

# ─────────────────────────────────────────────────────────────────────────
#  STORE
# ─────────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
# Insertion order of job_ids, oldest first -- used by get_latest_complete()
# to find the most recently created job without re-sorting on every call.
_job_order: List[str] = []


def create_job(filename: str) -> str:
    """Create a new job in the "queued" state and return its job_id."""
    job_id = uuid.uuid4().hex[:12]
    now = time.time()

    job = {
        "job_id": job_id,
        "filename": filename,
        "status": "queued",
        "stage": "queued",
        "stage_label": "Queued",
        "progress": 0,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "log": [],
    }

    with _lock:
        _jobs[job_id] = job
        _job_order.append(job_id)

    return job_id


def get(job_id: str) -> Optional[Dict[str, Any]]:
    """Return a snapshot (shallow copy) of the job, or None if unknown.

    Returns a copy (with a fresh copy of the log list) rather than the
    live dict so callers can't mutate stored state by accident, and so
    a poller reading this doesn't see a half-written job mid-update from
    the background thread.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        snapshot = dict(job)
        snapshot["log"] = list(job["log"])
        return snapshot


def get_latest_complete() -> Optional[Dict[str, Any]]:
    """Return the most recently created job whose status is "complete",
    or None if no job has completed yet. Used by /api/documents to
    decide whether there's a currently-ingested document to report on.
    """
    with _lock:
        for job_id in reversed(_job_order):
            job = _jobs.get(job_id)
            if job is not None and job["status"] == "complete":
                snapshot = dict(job)
                snapshot["log"] = list(job["log"])
                return snapshot
    return None


def set_stage(job_id: str, stage: str, progress: Optional[int] = None) -> None:
    """Move a job to a new pipeline stage. Sets status to "running" (a
    job is "running" the moment it has a real stage, as opposed to the
    initial "queued" state) and looks up the human-readable label from
    PIPELINE_STAGES automatically.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["stage"] = stage
        job["stage_label"] = _STAGE_LABELS.get(stage, stage)
        job["status"] = "running"
        if progress is not None:
            job["progress"] = progress
        job["updated_at"] = time.time()


def log(job_id: str, message: str) -> None:
    """Append a timestamped log line to the job's log panel."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["log"].append({"timestamp": time.time(), "message": message})
        job["updated_at"] = time.time()


def update(job_id: str, **fields: Any) -> None:
    """Set arbitrary fields on a job (e.g. status="complete", progress=100,
    or status="error", error=str(exc)). Used for the final state
    transition out of run_ingestion's try/except in pipeline_service.py.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated_at"] = time.time()
