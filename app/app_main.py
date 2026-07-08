import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import AppSettings
from backend.middleware.auth import ApiKeyAuthMiddleware
from backend.middleware.langsmith import LangSmithTraceMiddleware
from backend.middleware.rate_limit import SlidingWindowRateLimiter
from backend.middleware.request_id import RequestIdMiddleware
from backend.router import eval_router, health_router, observability_router, rag_router, research_router
from backend.service import get_workflow_service
from observability.langsmith import configure_langsmith


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.getLogger("mult_agents").setLevel(logging.INFO)
logging.getLogger("backend").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        await get_workflow_service().close()


def create_app() -> FastAPI:
    settings = AppSettings()
    settings.validate_for_runtime()
    configure_langsmith(settings.langsmith_settings())
    docs_enabled = settings.docs_enabled()
    excluded_paths = {"/health", "/api/v1/health"}
    if docs_enabled:
        excluded_paths.update({"/docs", "/openapi.json", "/redoc"})

    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    app.add_middleware(RequestIdMiddleware, header_name=settings.request_id_header)
    app.add_middleware(LangSmithTraceMiddleware, excluded_paths=excluded_paths)
    app.add_middleware(
        SlidingWindowRateLimiter,
        window_seconds=settings.rate_limit_window_seconds,
        max_requests=settings.rate_limit_max_requests,
        enabled=settings.rate_limit_enabled,
        backend=settings.rate_limit_backend,
        redis_url=settings.rate_limit_redis_url,
        trusted_proxy_headers=settings.trusted_proxy_headers,
        excluded_paths=excluded_paths,
    )
    app.add_middleware(
        ApiKeyAuthMiddleware,
        api_auth_key=settings.api_auth_key,
        auth_required=settings.auth_required(),
        excluded_paths=excluded_paths,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(research_router)
    app.include_router(rag_router)
    app.include_router(eval_router)
    app.include_router(observability_router)
    return app


app = create_app()


if __name__ == "__main__":
    runtime_settings = AppSettings()
    uvicorn.run(
        "app_main:app",
        host=runtime_settings.host,
        port=runtime_settings.port,
        reload=runtime_settings.app_env == "development",
    )
