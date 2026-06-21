import os
import time

from fastapi import APIRouter, HTTPException

from api.models.query import QueryRequest, QueryResponse
from main import run_query

router = APIRouter()

_query_cache: dict[tuple[str, int, int], dict] = {}


def _cache_key(req: QueryRequest) -> tuple[str, int, int]:
    normalized_query = req.query.strip().lower()
    return (normalized_query, req.top_k, req.token_budget)


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not os.getenv("MISTRAL_API_KEY"):
        raise HTTPException(500, "MISTRAL_API_KEY not set on server")

    key = _cache_key(req)
    t0 = time.perf_counter()

    if key in _query_cache:
        result = dict(_query_cache[key])
        result["server_total_ms"] = (time.perf_counter() - t0) * 1000
        result["cached"] = True
        return result

    try:
        result = run_query(
            req.query,
            rerank_top_k=req.top_k,
            token_budget=req.token_budget,
        )
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))

    _query_cache[key] = dict(result)

    result["server_total_ms"] = (time.perf_counter() - t0) * 1000
    result["cached"] = False
    return result