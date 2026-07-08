"""API-key authentication middleware."""

import secrets

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


_DEFAULT_EXCLUDED_PATHS = {"/health", "/api/v1/health"}


def _verify_api_key(api_key: str | None, expected: str, auth_required: bool) -> bool:
    if not expected:
        return not auth_required
    return bool(api_key) and secrets.compare_digest(api_key, expected)


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validate API credentials for protected backend routes."""

    def __init__(
        self,
        app,
        *,
        api_auth_key: str = "",
        auth_required: bool = False,
        excluded_paths: set[str] | None = None,
    ):
        super().__init__(app)
        self._api_auth_key = api_auth_key.strip()
        self._auth_required = auth_required
        self._excluded_paths = excluded_paths or _DEFAULT_EXCLUDED_PATHS

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._excluded_paths:
            return await call_next(request)

        api_key: str | None = None
        x_api_key = request.headers.get("X-API-Key")
        if x_api_key:
            api_key = x_api_key.strip()
        else:
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                api_key = auth[7:].strip()

        if not _verify_api_key(api_key, self._api_auth_key, self._auth_required):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "Missing or invalid API key. Provide X-API-Key or Authorization: Bearer header."
                },
            )

        return await call_next(request)
