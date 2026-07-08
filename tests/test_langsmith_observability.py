#!/usr/bin/env python3
"""Checks for optional LangSmith observability integration."""

from __future__ import annotations

import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))


class LangSmithObservabilityTests(unittest.TestCase):
    def test_settings_parse_current_and_legacy_environment_names(self):
        from observability.langsmith import LangSmithSettings

        settings = LangSmithSettings.from_mapping(
            {
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_API_KEY": "ls-key",
                "LANGSMITH_PROJECT": "deepresearch-test",
                "LANGSMITH_TAGS": "local,ci",
                "LANGSMITH_ENVIRONMENT": "test",
                "LANGSMITH_SAMPLE_RATE": "0.5",
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.api_key, "ls-key")
        self.assertEqual(settings.project, "deepresearch-test")
        self.assertEqual(settings.tags, ("local", "ci", "test"))
        self.assertEqual(settings.sample_rate, 0.5)

        deduped = LangSmithSettings.from_mapping(
            {
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_API_KEY": "ls-key",
                "LANGSMITH_TAGS": "local,ci,test,test",
                "LANGSMITH_ENVIRONMENT": "test",
            }
        )
        self.assertEqual(deduped.tags, ("local", "ci", "test"))

        legacy = LangSmithSettings.from_mapping(
            {
                "LANGCHAIN_TRACING_V2": "true",
                "LANGCHAIN_API_KEY": "legacy-key",
                "LANGCHAIN_PROJECT": "legacy-project",
            }
        )
        self.assertTrue(legacy.enabled)
        self.assertEqual(legacy.api_key, "legacy-key")
        self.assertEqual(legacy.project, "legacy-project")

    def test_configure_langsmith_sets_langchain_compatible_environment(self):
        from observability.langsmith import LangSmithSettings, configure_langsmith

        with patch.dict(os.environ, {}, clear=True):
            configure_langsmith(
                LangSmithSettings(
                    enabled=True,
                    api_key="ls-key",
                    project="deepresearch-test",
                    endpoint="https://example.langsmith",
                    tags=("unit", "unit"),
                )
            )

            self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")
            self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "true")
            self.assertEqual(os.environ["LANGSMITH_API_KEY"], "ls-key")
            self.assertEqual(os.environ["LANGCHAIN_API_KEY"], "ls-key")
            self.assertEqual(os.environ["LANGSMITH_PROJECT"], "deepresearch-test")
            self.assertEqual(os.environ["LANGCHAIN_PROJECT"], "deepresearch-test")
            self.assertEqual(os.environ["LANGSMITH_ENDPOINT"], "https://example.langsmith")
            self.assertEqual(os.environ["LANGSMITH_TAGS"], "unit")

    def test_trace_run_is_noop_when_disabled_or_not_sampled(self):
        from observability.langsmith import LangSmithSettings, trace_run

        with patch("observability.langsmith._ls_trace") as trace:
            with trace_run("unit", settings=LangSmithSettings(enabled=False)) as span:
                span.end(outputs={"ok": True})
            trace.assert_not_called()

        with patch("observability.langsmith._ls_trace") as trace:
            with trace_run(
                "unit",
                settings=LangSmithSettings(enabled=True, api_key="ls-key", sample_rate=0),
            ) as span:
                span.end(outputs={"ok": True})
            trace.assert_not_called()

    def test_trace_run_creates_langsmith_span_when_enabled(self):
        from observability.langsmith import LangSmithSettings, trace_run

        events: list[tuple[str, object]] = []

        class FakeRun:
            def end(self, outputs=None, error=None):
                events.append(("end", outputs or error))

        class FakeTrace:
            def __init__(self, *args, **kwargs):
                events.append(("trace", kwargs))

            def __enter__(self):
                return FakeRun()

            def __exit__(self, exc_type, exc, tb):
                events.append(("exit", exc_type))

        with (
            patch("observability.langsmith.tracing_context", return_value=nullcontext()),
            patch("observability.langsmith._ls_trace", side_effect=lambda *args, **kwargs: FakeTrace(*args, **kwargs)),
        ):
            with trace_run(
                "unit",
                inputs={"query": "hello"},
                metadata={"tenant_id": "tenant"},
                tags=("workflow",),
                settings=LangSmithSettings(
                    enabled=True,
                    api_key="ls-key",
                    project="deepresearch-test",
                    tags=("ci", "workflow"),
                ),
            ) as span:
                span.end(outputs={"answer": "world"})

        self.assertEqual(events[0][0], "trace")
        trace_kwargs = events[0][1]
        self.assertEqual(trace_kwargs["project_name"], "deepresearch-test")
        self.assertEqual(trace_kwargs["inputs"], {"query": "hello"})
        self.assertEqual(trace_kwargs["tags"], ["ci", "workflow"])
        self.assertIn(("end", {"answer": "world"}), events)

    def test_app_settings_validate_langsmith_key_when_enabled(self):
        from backend.config import AppSettings

        settings = AppSettings(
            langsmith_enabled=True,
            langsmith_api_key="",
            langchain_api_key="",
        )
        with self.assertRaises(RuntimeError) as error:
            settings.validate_for_runtime()
        self.assertIn("LANGSMITH_API_KEY", str(error.exception))

    def test_observability_status_endpoint_is_admin_only(self):
        from backend.config import AppSettings
        from backend.dependencies.admin import get_runtime_settings
        from backend.router.observability_router import router

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_runtime_settings] = lambda: AppSettings(
            admin_api_required=True,
            admin_api_key="admin-secret",
            langsmith_enabled=True,
            langsmith_api_key="ls-key",
            langsmith_project="deepresearch-test",
        )
        client = TestClient(app)

        self.assertEqual(client.get("/api/v1/observability/langsmith/status").status_code, 403)
        response = client.get(
            "/api/v1/observability/langsmith/status",
            headers={"X-Admin-Key": "admin-secret"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["project"], "deepresearch-test")
        self.assertTrue(response.json()["api_key_configured"])


if __name__ == "__main__":
    unittest.main()
