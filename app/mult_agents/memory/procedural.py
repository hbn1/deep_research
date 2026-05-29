"""
Procedural memory module.

Stores agent behavioral patterns, successful strategies, and learned heuristics.
Enables the agent to improve over time by remembering what works.
"""

import json as _json
import logging as _logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .base import MemoryEntry, MemoryType
from .long_term import SQLiteLongTermMemory

_logger = _logging.getLogger("mult_agents.memory")


class ProceduralMemoryStore(SQLiteLongTermMemory):
    """
    Procedural memory: stores agent behavioral patterns.

    Each entry captures:
    - trigger: When/under what conditions to apply this pattern
    - action: What to do (strategy, tool combination, approach)
    - success_rate: Historical success rate (0-1)
    - context: Additional context for pattern application
    - confidence: How confident we are in this pattern (decays with disuse)
    """

    def __init__(self, db_path=None):
        super().__init__(MemoryType.PROCEDURAL, db_path)

    def learn_pattern(
        self,
        user_id: str,
        trigger: str,
        action: str,
        context: str = "",
        importance: float = 0.5,
        thread_id: Optional[str] = None,
    ) -> str:
        """Learn a new behavioral pattern."""
        content = {
            "trigger": trigger,
            "action": action,
            "context": context,
            "success_rate": 0.5,
            "confidence": 0.5,
            "times_applied": 0,
        }

        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            user_id=user_id,
            thread_id=thread_id,
            namespace="patterns",
            importance=importance,
            metadata={"type": "behavioral_pattern"},
        )
        return self.save(entry)

    def reinforce_pattern(self, memory_id: str, success: bool = True) -> bool:
        """Reinforce a pattern based on success/failure."""
        entry = self.get(memory_id)
        if not entry or not isinstance(entry.content, dict):
            return False

        entry.record_recall()
        ct = entry.content
        ct["times_applied"] = ct.get("times_applied", 0) + 1
        n = ct["times_applied"]
        old_rate = ct.get("success_rate", 0.5)
        ct["success_rate"] = (old_rate * (n - 1) + (1.0 if success else 0.0)) / n
        ct["confidence"] = min(ct["confidence"] * 1.05 + 0.05, 1.0) if success else max(ct["confidence"] * 0.7, 0.05)
        entry.updated_at = datetime.now()
        self.save(entry)
        return True

    def get_relevant_patterns(
        self,
        user_id: str,
        context: str,
        limit: int = 5,
        min_success_rate: float = 0.3,
    ) -> List[MemoryEntry]:
        """Get patterns relevant to the current context.

        Args:
            user_id: User identifier
            context: Current context description (task type, query topic, etc.)
            limit: Max patterns to return
            min_success_rate: Minimum success rate filter
        """
        results = self.search(query=context, user_id=user_id, limit=limit * 2)

        # Filter by success rate and sort by confidence * success_rate
        filtered = []
        for entry in results:
            if not isinstance(entry.content, dict):
                continue
            sr = entry.content.get("success_rate", 0.5)
            if sr < min_success_rate:
                continue
            # Boost recent entries
            entry.record_recall()
            filtered.append(entry)

        # Sort by composite score: confidence * success_rate * retention
        def score(e):
            ct = e.content if isinstance(e.content, dict) else {}
            conf = ct.get("confidence", 0.5)
            sr = ct.get("success_rate", 0.5)
            retention = e.compute_retention_score()
            return conf * sr * 0.7 + retention * 0.3

        filtered.sort(key=score, reverse=True)
        return filtered[:limit]

    def decay_patterns(self, user_id: str) -> int:
        """Apply confidence decay to unused patterns."""
        results = self.search(query="", user_id=user_id, limit=100)
        decayed = 0
        for entry in results:
            if not isinstance(entry.content, dict):
                continue
            score = entry.compute_retention_score()
            if score < 0.1:
                self.delete(entry.id)
                decayed += 1
            elif score < 0.3:
                ct = entry.content
                ct["confidence"] = max(ct.get("confidence", 0.5) * 0.95, 0.05)
                entry.content = ct
                entry.updated_at = datetime.now()
                self.save(entry)
                decayed += 1
        if decayed:
            _logger.info("Decayed %d procedural patterns for user %s", decayed, user_id)
        return decayed

    def to_prompt_context(self, patterns: List[MemoryEntry]) -> str:
        """Format patterns for prompt injection."""
        if not patterns:
            return ""
        lines = ["## Learned Behavioral Patterns"]
        for i, entry in enumerate(patterns, 1):
            ct = entry.content if isinstance(entry.content, dict) else {}
            trigger = ct.get("trigger", "unknown")
            action = ct.get("action", "unknown")
            sr = ct.get("success_rate", 0.5)
            conf = ct.get("confidence", 0.5)
            lines.append(f"{i}. When '{trigger}': {action} (success={sr:.0%}, conf={conf:.0%})")
        return "\n".join(lines)
