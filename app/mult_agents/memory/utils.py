"""
Memory system utility functions.

Compatibility wrappers that delegate to new extractor/injector modules.
"""

import json as _json
import logging as _logging
import re as _re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .base import MemoryEntry, MemoryType

_logger = _logging.getLogger("mult_agents.memory")


def create_memory_checkpoint(thread_id, state, checkpoint_id=None):
    """Create a memory checkpoint for state recovery."""
    checkpoint = {
        "id": checkpoint_id or str(uuid4()),
        "thread_id": thread_id,
        "state": {
            "query": state.get("query"),
            "intent": state.get("intent"),
            "plan": state.get("plan"),
            "analysis": state.get("analysis"),
            "final": state.get("final"),
        },
        "created_at": datetime.now().isoformat(),
    }
    return checkpoint


def extract_memory_from_messages(messages, extract_facts=True, extract_preferences=True):
    """Extract memory from messages (delegates to extractor module)."""
    from .extractor import extract_memory_from_messages as _extract
    return _extract(messages, extract_facts, extract_preferences)


def format_memories_for_prompt(memories, max_length=2000):
    """Format memories for prompt injection (delegates to injector module)."""
    from .injector import format_memories_for_prompt as _fmt
    return _fmt(memories, max_length)


def merge_user_profile(existing, new_data):
    """Merge user profile data intelligently."""
    if existing is None:
        return new_data.copy()
    merged = existing.copy()
    for key, value in new_data.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key].update(value)
        elif key in merged and isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = list(set(merged[key] + value))
        else:
            merged[key] = value
    merged["_last_updated"] = datetime.now().isoformat()
    return merged


def calculate_memory_relevance(query, memory, time_decay_hours=168):
    """Calculate relevance score incorporating importance and decay."""
    import math
    query_lower = query.lower()
    content_str = str(memory.content).lower()
    text_score = 0.0
    if query_lower in content_str:
        text_score = 0.5
    query_words = set(query_lower.split())
    content_words = set(content_str.split())
    if query_words:
        overlap = len(query_words & content_words) / len(query_words)
        text_score = max(text_score, overlap * 0.8)
    age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
    time_factor = math.exp(-age_hours / time_decay_hours)
    access_bonus = min(memory.access_count * 0.05, 0.2)
    # Incorporate importance from MemoryEntry if available
    importance = getattr(memory, "importance", 0.5)
    return min((text_score * 0.5 + time_factor * 0.2 + access_bonus * 0.1 + importance * 0.2), 1.0)


def compress_memories(memories, target_count=10):
    """Compress memory list, keeping most relevant entries."""
    if len(memories) <= target_count:
        return memories
    unique = []
    for mem in memories:
        dup = False
        for existing in unique:
            sim = _simple_similarity(str(mem.content), str(existing.content))
            if sim > 0.8:
                dup = True
                break
        if not dup:
            unique.append(mem)
    def score(m):
        age_days = (datetime.now() - m.created_at).days
        imp = getattr(m, "importance", 0.5)
        return m.access_count * 10 - age_days + imp * 20
    unique.sort(key=score, reverse=True)
    return unique[:target_count]


def _simple_similarity(text1, text2):
    """Simple word overlap similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)
