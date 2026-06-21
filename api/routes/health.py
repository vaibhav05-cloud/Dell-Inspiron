from fastapi import APIRouter
from api.models.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")