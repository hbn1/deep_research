#!/usr/bin/env python3
"""Runtime hardening checks for enterprise deployment behavior."""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))


class RuntimeHardeningTests(unittest.TestCase):
    def test_search_cache_is_bounded_lru_and_ttl_aware(self):
        from mult_agents.search import SearchCache

        cache = SearchCache(ttl_seconds=60, max_entries=2)
        cache.set("a", [{"value": "a"}])
        cache.set("b", [{"value": "b"}])

        self.assertEqual(cache.get("a"), [{"value": "a"}])
        cache.set("c", [{"value": "c"}])

        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), [{"value": "a"}])
        self.assertEqual(cache.get("c"), [{"value": "c"}])

        expiring_cache = SearchCache(ttl_seconds=1, max_entries=2)
        expiring_cache.set("old", [{"value": "old"}])
        key = expiring_cache._key("old")
        expiring_cache._store[key] = (time.time() - 2, [{"value": "old"}])
        self.assertIsNone(expiring_cache.get("old"))

    def test_close_search_resources_closes_shared_http_client(self):
        import mult_agents.search as search_module

        if not search_module._HAS_HTTPX:
            self.skipTest("httpx is not installed")

        client = search_module._get_httpx_client()
        self.assertFalse(client.is_closed)

        search_module.close_search_resources()

        self.assertTrue(client.is_closed)

    def test_workflow_close_drains_background_persist_tasks(self):
        from backend.service.workflow_service import WorkflowService

        async def scenario():
            service = WorkflowService(config_path="missing-config.json")
            persisted: list[str] = []

            async def fake_persist_turn(_config, _query, answer):
                await asyncio.sleep(0.01)
                persisted.append(answer)

            service._persist_turn = fake_persist_turn
            runtime_config = SimpleNamespace(enable_memory=True)
            service._schedule_persist_turn(runtime_config, "question", "answer")
            self.assertEqual(len(service._background_tasks), 1)

            await service.close()

            self.assertEqual(persisted, ["answer"])
            self.assertFalse(service._background_tasks)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
