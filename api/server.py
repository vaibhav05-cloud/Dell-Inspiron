from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes import health, query, ingest

load_dotenv()

app = FastAPI(title="Dell FutureMinds GraphRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(ingest.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}