from fastapi import APIRouter, Depends

from backend.config import AppSettings
from backend.dependencies import get_runtime_settings, require_admin_access


router = APIRouter(
    prefix="/api/v1/observability",
    tags=["observability"],
    dependencies=[Depends(require_admin_access)],
)


@router.get("/langsmith/status")
def langsmith_status(settings: AppSettings = Depends(get_runtime_settings)):
    return settings.langsmith_settings().status()
