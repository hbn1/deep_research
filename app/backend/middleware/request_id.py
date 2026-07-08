"""Request correlation middleware."""

from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a stable request id to each response."""

    def __init__(self, app, *, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self._header_name = header_name

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(self._header_name, "").strip() or uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[self._header_name] = request_id
        return response
