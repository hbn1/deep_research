#!/usr/bin/env python3
"""Focused checks for the evaluation subsystem."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
os.environ.setdefault("DASHSCOPE_API_KEY", "")


class EvaluationTests(unittest.TestCase):
    def test_dataset_loader_smoke_cases(self):
        from evaluation.dataset_loader import EvalDatasetLoader

        loader = EvalDatasetLoader(Path(__file__).resolve().parent.parent / "eval_datasets")
        datasets = {item.id: item.case_count for item in loader.list_datasets()}
        self.assertGreaterEqual(datasets["smoke"], 1)
        cases = loader.load_cases("smoke", max_cases=1)
        self.assertEqual(cases[0].id, "smoke_direct_intro")

    def test_dataset_loader_uses_env_directory(self):
        from evaluation.dataset_loader import EvalDatasetLoader

        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "prod.jsonl"
            dataset.write_text(
                '{"id":"prod_case","query":"hello","expectations":{"must_contain":["hi"]}}\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"EVAL_DATASETS_DIR": tmp}):
                datasets = EvalDatasetLoader().list_datasets()
            self.assertEqual([item.id for item in datasets], ["prod"])

    def test_result_store_uses_env_directory(self):
        from evaluation.store import EvalResultStore

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"EVAL_RESULTS_DIR": tmp}):
                store = EvalResultStore()
            self.assertEqual(store.base_dir, Path(tmp))

    def test_evaluator_flags_missing_citation(self):
        from evaluation.evaluators import evaluate_case
        from evaluation.schemas import EvalCase

        case = EvalCase(
            id="citation_case",
            query="Compare tools with citations",
            expected_route="multiagent",
            min_citations=1,
        )
        result = evaluate_case(
            case=case,
            state={
                "intent": "multiagent",
                "final": "LangGraph is useful [WEB1_0-9].",
                "source_index": [{"source_id": "WEB1_0-1", "title": "Source"}],
                "evidence_pool": [{"source_id": "WEB1_0-1", "content": "LangGraph is useful"}],
            },
            latency_ms=10,
            error=None,
            default_threshold=70,
        )
        self.assertFalse(result.passed)
        self.assertIn("CitationValidity", result.failed_evaluators)
        self.assertIn("write", result.suspected_stages)

    def test_eval_dataset_api_lists_smoke(self):
        from fastapi.testclient import TestClient
        from app_main import create_app

        client = TestClient(create_app())
        response = client.get("/api/v1/evals/datasets")
        self.assertEqual(response.status_code, 200)
        dataset_ids = {item["id"] for item in response.json()["datasets"]}
        self.assertIn("smoke", dataset_ids)


if __name__ == "__main__":
    unittest.main()
