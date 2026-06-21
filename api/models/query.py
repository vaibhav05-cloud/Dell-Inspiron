from __future__ import annotations
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int = 15
    token_budget: int = 4000


class EvidenceItem(BaseModel):
    page_number: str
    source_document: str


class QueryResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem]
    reasoning_path: list[str]
    confidence: str
    server_total_ms: float
    cached: bool