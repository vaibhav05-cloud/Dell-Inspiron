import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form

from api.models.ingest import IngestResponse
from main import run_ingestion

router = APIRouter()

UPLOAD_DIR = Path("data/pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    force: bool = Form(False),
):
    if not os.getenv("MISTRAL_API_KEY"):
        raise HTTPException(500, "MISTRAL_API_KEY not set on server")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    save_path = UPLOAD_DIR / file.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(run_ingestion, str(save_path), force)
    return IngestResponse(status="ingestion_started", pdf_path=str(save_path))