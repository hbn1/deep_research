"""
简易速率限制中间件（滑动窗口）。

基于内存存储，每个 IP 在一定窗口内限制请求数。
"""

import time
from collections import defaultdict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

_EXCLUDED_PATHS = {"/health", "/api/v1/health", "/docs", "/openapi.json", "/redoc"}


class SlidingWindowRateLimiter(BaseHTTPMiddleware):
    """滑动窗口速率限制。

    默认: 窗口 60 秒内最多 30 请求。
    """

    def __init__(self, app, window_seconds: int = 60, max_requests: int = 30):
        super().__init__(app)
        self._window = window_seconds
        self._max = max_requests
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def _clean(self, client: str, now: float) -> None:
        cutoff = now - self._window
        self._buckets[client] = [t for t in self._buckets[client] if t > cutoff]

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        client: str = request.client.host if request.client else "unknown"
        now = time.time()
        self._clean(client, now)

        if len(self._buckets[client]) >= self._max:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self._max} requests per {self._window}s.",
            )

        self._buckets[client].append(now)

        response = await call_next(request)
        return response
