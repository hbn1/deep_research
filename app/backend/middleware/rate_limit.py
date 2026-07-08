"""Sliding-window rate limiter for local and production deployments."""

import asyncio
import time
from collections import defaultdict, deque
from typing import Deque
from uuid import uuid4

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


_DEFAULT_EXCLUDED_PATHS = {"/health", "/api/v1/health"}


class SlidingWindowRateLimiter(BaseHTTPMiddleware):
    """Limit requests per client IP within a rolling time window."""

    def __init__(
        self,
        app,
        window_seconds: int = 60,
        max_requests: int = 30,
        *,
        enabled: bool = True,
        backend: str = "memory",
        redis_url: str = "",
        trusted_proxy_headers: bool = False,
        excluded_paths: set[str] | None = None,
        max_memory_clients: int = 10000,
    ):
        super().__init__(app)
        self._window = max(1, int(window_seconds))
        self._max = max(1, int(max_requests))
        self._enabled = enabled
        self._backend = backend
        self._trusted_proxy_headers = trusted_proxy_headers
        self._excluded_paths = excluded_paths or _DEFAULT_EXCLUDED_PATHS
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)
        self._memory_lock = asyncio.Lock()
        self._max_memory_clients = max(1, int(max_memory_clients))
        self._last_bucket_prune = 0.0
        self._redis = None
        if backend == "redis":
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(redis_url, decode_responses=True)

    def _clean_bucket(self, bucket: Deque[float], now: float) -> None:
        cutoff = now - self._window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def _prune_idle_buckets(self, now: float) -> None:
        if now - self._last_bucket_prune < self._window:
            return
        self._last_bucket_prune = now
        for client in list(self._buckets):
            bucket = self._buckets[client]
            self._clean_bucket(bucket, now)
            if not bucket:
                self._buckets.pop(client, None)
        while len(self._buckets) > self._max_memory_clients:
            self._buckets.pop(next(iter(self._buckets)), None)

    def _client_key(self, request: Request) -> str:
        if self._trusted_proxy_headers:
            forwarded_for = request.headers.get("X-Forwarded-For", "")
            if forwarded_for:
                return forwarded_for.split(",", 1)[0].strip() or "unknown"
        return request.client.host if request.client else "unknown"

    async def _check_redis_limit(self, client: str, now: float) -> bool:
        if self._redis is None:
            return True
        now_ms = int(now * 1000)
        cutoff_ms = now_ms - (self._window * 1000)
        key = f"rate_limit:{client}"
        await self._redis.zremrangebyscore(key, 0, cutoff_ms)
        current = await self._redis.zcard(key)
        if current >= self._max:
            return False
        await self._redis.zadd(key, {f"{now_ms}:{uuid4().hex}": now_ms})
        await self._redis.expire(key, self._window + 1)
        return True

    async def _check_memory_limit(self, client: str, now: float) -> bool:
        async with self._memory_lock:
            self._prune_idle_buckets(now)
            bucket = self._buckets[client]
            self._clean_bucket(bucket, now)
            if len(bucket) >= self._max:
                return False
            bucket.append(now)
            return True

    async def dispatch(self, request: Request, call_next):
        if not self._enabled or request.url.path in self._excluded_paths:
            return await call_next(request)

        client = self._client_key(request)
        now = time.time()
        allowed = (
            await self._check_redis_limit(client, now)
            if self._backend == "redis"
            else await self._check_memory_limit(client, now)
        )
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": f"Rate limit exceeded: {self._max} requests per {self._window}s."},
            )

        return await call_next(request)
