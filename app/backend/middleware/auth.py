"""
API 鉴权中间件。

通过 X-API-Key header 或 Authorization Bearer token 校验请求。
校验 key 为 DASHSCOPE_API_KEY 环境变量。
"""

import os
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

_EXCLUDED_PATHS = {"/health", "/api/v1/health", "/docs", "/openapi.json", "/redoc", "/api/v1/research", "/api/v1/research/stream", "/api/v1/research/run"}


def _verify_api_key(api_key: str | None) -> bool:
    expected = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not expected:
        # 未配置 API Key 时放行（开发/本地模式）
        return True
    return bool(api_key) and api_key == expected


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """校验 X-API-Key 或 Authorization: Bearer <key>。

    排除路径: /health, /docs, /openapi.json, /redoc
    """

    async def dispatch(self, request: Request, call_next):
        if any(request.url.path.startswith(p) for p in _EXCLUDED_PATHS):
            return await call_next(request)

        api_key: str | None = None

        # 优先 X-API-Key header
        x_api_key = request.headers.get("X-API-Key")
        if x_api_key:
            api_key = x_api_key.strip()
        else:
            # 尝试 Authorization: Bearer <key>
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                api_key = auth[7:].strip()

        if not _verify_api_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key. Provide X-API-Key or Authorization: Bearer header.",
            )

        return await call_next(request)
