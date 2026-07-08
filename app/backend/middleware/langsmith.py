from __future__ import annotations

from collections.abc import Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from observability.langsmith import trace_run


class LangSmithTraceMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, excluded_paths: Iterable[str] | None = None):
        super().__init__(app)
        self._excluded_paths = set(excluded_paths or ())

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in self._excluded_paths:
            return await call_next(request)

        request_id = getattr(request.state, "request_id", "") or request.headers.get("X-Request-ID", "")
        with trace_run(
            "http.request",
            run_type="chain",
            inputs={"method": request.method, "path": path},
            metadata={
                "request_id": request_id,
                "method": request.method,
                "path": path,
                "client": request.client.host if request.client else "",
            },
            tags=("http",),
        ) as span:
            try:
                response = await call_next(request)
            except Exception as exc:
                span.end(error=str(exc))
                raise
            span.end(outputs={"status_code": response.status_code})
            return response
