#!/usr/bin/env python3
"""Fast-path checks for direct answers that should not call an LLM."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))


class DirectAnswerFastPathTests(unittest.TestCase):
    def test_intent_marks_simple_arithmetic_as_confident_direct(self):
        from mult_agents.nodes.intent import detect_intent, is_confident_direct_query

        query = "Answer only: 1+1"

        self.assertEqual(detect_intent(query), "direct")
        self.assertTrue(is_confident_direct_query(query))

    def test_direct_answer_solves_simple_arithmetic_locally(self):
        from mult_agents.nodes.direct_answer import _deterministic_direct_answer

        self.assertEqual(_deterministic_direct_answer("Answer only: 1+1"), "2")
        self.assertEqual(_deterministic_direct_answer("what is 6 / 3?"), "2")
        self.assertEqual(_deterministic_direct_answer("请计算 7×8 等于多少"), "56")

    def test_direct_answer_rejects_unsafe_or_unsupported_input(self):
        from mult_agents.nodes.direct_answer import _deterministic_direct_answer

        self.assertIsNone(_deterministic_direct_answer("__import__('os').system('whoami')"))
        self.assertIsNone(_deterministic_direct_answer("2 ** 100"))
        self.assertIsNone(_deterministic_direct_answer("hello + 1"))

    def test_web_search_node_timeout_uses_search_config(self):
        from mult_agents.nodes.web_search import _resolve_search_node_timeout
        from mult_agents.search import SearchConfig

        config = SearchConfig(request_timeout=5.0, fetch_timeout=2.0, fetch_enabled=True, rewrite_enabled=False)
        self.assertEqual(_resolve_search_node_timeout(config), 9.0)

        with patch.dict("os.environ", {"SEARCH_NODE_TIMEOUT": "4.5"}):
            self.assertEqual(_resolve_search_node_timeout(config), 4.5)


if __name__ == "__main__":
    unittest.main()
