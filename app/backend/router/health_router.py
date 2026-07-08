from fastapi import APIRouter

from backend.schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="deepresearch-backend")


@router.get("/api/v1/health", response_model=HealthResponse)
async def api_health() -> HealthResponse:
    return HealthResponse(status="ok", service="deepresearch-backend")
