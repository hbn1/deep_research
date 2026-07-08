#!/usr/bin/env python3
"""Focused backend API and middleware checks."""

import json
import sys
import unittest
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))


class BackendApiTests(unittest.TestCase):
    def _protected_client(self, api_auth_key: str = "", auth_required: bool = False) -> TestClient:
        from backend.middleware.auth import ApiKeyAuthMiddleware

        app = FastAPI()
        app.add_middleware(
            ApiKeyAuthMiddleware,
            api_auth_key=api_auth_key,
            auth_required=auth_required,
        )

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.get("/protected")
        def protected():
            return {"ok": True}

        return TestClient(app)

    def test_auth_middleware_accepts_expected_headers(self):
        client = self._protected_client(api_auth_key="local-secret", auth_required=True)

        self.assertEqual(client.get("/protected").status_code, 401)
        self.assertEqual(client.get("/protected", headers={"X-API-Key": "wrong"}).status_code, 401)
        self.assertEqual(client.get("/protected", headers={"X-API-Key": "local-secret"}).status_code, 200)
        self.assertEqual(
            client.get("/protected", headers={"Authorization": "Bearer local-secret"}).status_code,
            200,
        )
        self.assertEqual(client.get("/health").status_code, 200)

    def test_auth_middleware_allows_local_mode_without_key(self):
        client = self._protected_client()
        self.assertEqual(client.get("/protected").status_code, 200)

    def test_auth_middleware_fails_closed_when_required_without_key(self):
        client = self._protected_client(auth_required=True)
        self.assertEqual(client.get("/protected").status_code, 401)

    def test_production_settings_fail_without_api_auth_key(self):
        from backend.config import AppSettings

        settings = AppSettings(app_env="production", api_auth_required=True, api_auth_key="")
        with self.assertRaises(RuntimeError) as error:
            settings.validate_for_runtime()
        self.assertIn("API_AUTH_KEY", str(error.exception))

    def test_production_settings_fail_without_admin_api_key(self):
        from backend.config import AppSettings

        settings = AppSettings(app_env="production", api_auth_key="api-secret", admin_api_key="")
        with self.assertRaises(RuntimeError) as error:
            settings.validate_for_runtime()
        self.assertIn("ADMIN_API_KEY", str(error.exception))

    def test_admin_dependency_requires_separate_admin_key(self):
        from backend.config import AppSettings
        from backend.dependencies.admin import get_runtime_settings, require_admin_access

        app = FastAPI()

        @app.get("/admin", dependencies=[Depends(require_admin_access)])
        def admin():
            return {"ok": True}

        app.dependency_overrides[get_runtime_settings] = lambda: AppSettings(
            api_auth_key="api-secret",
            admin_api_key="admin-secret",
            admin_api_required=True,
        )
        client = TestClient(app)

        self.assertEqual(client.get("/admin").status_code, 403)
        self.assertEqual(client.get("/admin", headers={"X-API-Key": "api-secret"}).status_code, 403)
        self.assertEqual(client.get("/admin", headers={"X-Admin-Key": "wrong"}).status_code, 403)
        self.assertEqual(client.get("/admin", headers={"X-Admin-Key": "admin-secret"}).status_code, 200)
        self.assertEqual(client.get("/admin", headers={"Authorization": "Bearer admin-secret"}).status_code, 200)

    def test_docs_default_to_disabled_in_production(self):
        from backend.config import AppSettings

        self.assertFalse(AppSettings(app_env="production", api_auth_key="secret").docs_enabled())
        self.assertTrue(AppSettings(app_env="development").docs_enabled())

    def test_request_id_middleware_sets_response_header(self):
        from backend.middleware.request_id import RequestIdMiddleware

        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)

        @app.get("/ping")
        def ping():
            return {"ok": True}

        client = TestClient(app)
        generated = client.get("/ping")
        self.assertEqual(generated.status_code, 200)
        self.assertTrue(generated.headers.get("X-Request-ID"))
        echoed = client.get("/ping", headers={"X-Request-ID": "req-123"})
        self.assertEqual(echoed.headers["X-Request-ID"], "req-123")

    def test_research_stream_starts_with_readable_status_event(self):
        from backend.router.research_router import router
        from backend.service import get_workflow_service

        class FakeWorkflowService:
            async def stream_events(self, **kwargs):
                yield {"type": "final", "final": "done"}

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_workflow_service] = lambda: FakeWorkflowService()
        client = TestClient(app)

        response = client.post("/api/v1/research/stream", json={"query": "hello"})
        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertIn("Research task received; initializing workflow.", text)
        first_event = text.split("\n\n", 1)[0].removeprefix("data: ")
        self.assertEqual(json.loads(first_event)["type"], "status")

    def test_rate_limiter_returns_429_and_excludes_health(self):
        from backend.middleware.rate_limit import SlidingWindowRateLimiter

        app = FastAPI()
        app.add_middleware(SlidingWindowRateLimiter, window_seconds=60, max_requests=1)

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.get("/limited")
        def limited():
            return {"ok": True}

        client = TestClient(app)
        self.assertEqual(client.get("/limited").status_code, 200)
        limited_response = client.get("/limited")
        self.assertEqual(limited_response.status_code, 429)
        self.assertIn("Rate limit exceeded", limited_response.json()["detail"])
        self.assertEqual(client.get("/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
