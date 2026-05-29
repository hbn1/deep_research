"""
Structured memory injector.

Replaces simple text concatenation with structured context injection.
"""

import json as _json
import logging as _logging
from typing import Any, Dict, List, Optional

from .base import MemoryEntry, MemoryType

_logger = _logging.getLogger("mult_agents.memory")


class MemoryInjector:
    """Builds structured memory context for injection into prompts."""

    def __init__(self, max_tokens_estimate: int = 800):
        self.max_tokens_estimate = max_tokens_estimate

    def build_context(
        self,
        user_profile=None,
        recent_messages=None,
        relevant_memories=None,
        similar_tasks=None,
        conversation_summary="",
        active_procedural=None,
    ):
        context = {
            "persona": {},
            "recent_context": [],
            "relevant_knowledge": [],
            "similar_tasks": [],
            "active_procedural": active_procedural or [],
            "summary": conversation_summary,
        }

        if user_profile:
            profile = user_profile.copy()
            profile.pop("_last_updated", None)
            context["persona"] = profile

        if recent_messages:
            for msg in recent_messages[-6:]:
                msg_type = getattr(msg, "type", "human")
                text = str(getattr(msg, "content", "")).strip()
                if text:
                    if len(text) > 200:
                        text = text[:200] + "..."
                    context["recent_context"].append({
                        "role": "user" if msg_type == "human" else "assistant",
                        "content": text,
                    })

        if relevant_memories:
            for mem in relevant_memories[:8]:
                snippet = str(mem.content) if isinstance(mem.content, str) else _json.dumps(mem.content, ensure_ascii=False)
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                context["relevant_knowledge"].append({
                    "type": mem.memory_type.value,
                    "namespace": mem.namespace,
                    "content": snippet,
                })

        if similar_tasks:
            for task in similar_tasks[:3]:
                if isinstance(task.content, dict):
                    context["similar_tasks"].append({
                        "task_type": task.content.get("task_type", "unknown"),
                        "outcome": str(task.content.get("outcome", ""))[:200],
                    })

        return context

    def format_for_prompt(self, context):
        sections = []

        persona = context.get("persona", {})
        if persona:
            parts = []
            for key, value in persona.items():
                if isinstance(value, list):
                    parts.append("- {}: {}".format(key, ", ".join(str(v) for v in value[:5])))
                elif isinstance(value, dict):
                    parts.append("- {}: {}".format(key, _json.dumps(value, ensure_ascii=False)))
                else:
                    parts.append("- {}: {}".format(key, value))
            if parts:
                sections.append("## User Persona\n" + "\n".join(parts))

        recent = context.get("recent_context", [])
        if recent:
            lines = ["## Recent Conversation"]
            for item in recent:
                role_label = "User" if item["role"] == "user" else "Assistant"
                lines.append("- {}: {}".format(role_label, item["content"]))
            sections.append("\n".join(lines))

        knowledge = context.get("relevant_knowledge", [])
        if knowledge:
            lines = ["## Relevant Knowledge"]
            for i, item in enumerate(knowledge, 1):
                lines.append("{}. [{}] {}".format(i, item.get("type", ""), item.get("content", "")))
            sections.append("\n".join(lines))

        tasks = context.get("similar_tasks", [])
        if tasks:
            lines = ["## Similar Past Tasks"]
            for i, task in enumerate(tasks, 1):
                lines.append("{}. [{}] {}".format(i, task.get("task_type", ""), task.get("outcome", "")))
            sections.append("\n".join(lines))

        procedural = context.get("active_procedural", [])
        if procedural:
            lines = ["## Active Behavioral Patterns"]
            for i, proc in enumerate(procedural[:3], 1):
                lines.append("{}. When '{}': {}".format(i, proc.get("trigger", ""), proc.get("action", "")))
            sections.append("\n".join(lines))

        summary = context.get("summary", "")
        if summary:
            sections.append("## Conversation Summary\n{}".format(summary))

        result = "\n\n".join(sections).strip()
        if len(result) > 3000:
            result = result[:3000] + "\n...(truncated)"
        return result

    def build_personalized_prompt(self, user_prompt, context):
        memory_text = self.format_for_prompt(context)
        if not memory_text:
            return user_prompt
        return "{}\n\n[Memory Context]\n{}".format(user_prompt, memory_text)


def format_memories_for_prompt(memories, max_length=2000):
    """Legacy compatibility wrapper."""
    injector = MemoryInjector()
    context = injector.build_context(relevant_memories=memories)
    return injector.format_for_prompt(context)
