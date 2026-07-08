import secrets

from fastapi import Depends, HTTPException, Request, status

from backend.config import AppSettings


def get_runtime_settings() -> AppSettings:
    return AppSettings()


def _bearer_token(value: str | None) -> str:
    if not value:
        return ""
    prefix = "Bearer "
    if value.startswith(prefix):
        return value[len(prefix) :].strip()
    return ""


def require_admin_access(
    request: Request,
    settings: AppSettings = Depends(get_runtime_settings),
) -> None:
    if not settings.admin_auth_required():
        return

    expected = settings.admin_api_key.strip()
    provided = (
        request.headers.get(settings.admin_api_key_header, "").strip()
        or _bearer_token(request.headers.get("Authorization"))
    )
    if expected and provided and secrets.compare_digest(provided, expected):
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin API key is required for this endpoint.",
    )
