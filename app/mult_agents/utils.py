"""Shared utilities: logging, message helpers, and agent invocation patterns."""

import json
import logging
import os
import re
from functools import partial

from langchain_core.messages import HumanMessage

from .state import ResearchState


logger = logging.getLogger("mult_agents")

# ?? Terminal color support ??

_ANSI = {
    "reset": "\033[0m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "red": "\033[31m",
}


def colorize(text: str, color: str) -> str:
    if os.getenv("NO_COLOR"):
        return text
    code = _ANSI.get(color, "")
    if not code:
        return text
    return f"{code}{text}{_ANSI['reset']}"


def emit(node: str, content: str):
    preview = content.replace("\n", " ")
    if len(preview) > 400:
        preview = preview[:400] + "..."
    logger.info("%s ??: %s", colorize(f"[{node}]", "yellow"), preview)


# ?? Message helpers ??

def collect_tool_calls(messages) -> tuple[list, list]:
    tools = []
    tool_outputs = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else None
                if name:
                    tools.append(name)
        name = getattr(msg, "name", None)
        msg_type = getattr(msg, "type", None)
        if msg_type == "tool" and name:
            tools.append(name)
            output = getattr(msg, "content", "")
            if output:
                tool_outputs.append(f"{name}: {output}")
    return tools, tool_outputs


def with_memory_context(state: ResearchState, user_prompt: str) -> str:
    memory_context = state.get("memory_context", "").strip()
    if not memory_context:
        return user_prompt
    return f"{user_prompt}\n\n[?????]\n{memory_context}"


def log_inputs(node: str, agent_name: str, payload: dict):
    preview = {
        key: (value[:200] + "..." if isinstance(value, str) and len(value) > 200 else value)
        for key, value in payload.items()
    }
    logger.info(
        "%s ?? | agent=%s | data=%s",
        colorize(f"[{node}]", "cyan"),
        colorize(agent_name, "magenta"),
        preview,
    )


# ?? JSON agent helpers ??

def _last_content(result) -> str:
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content:
            return str(content).strip()
    return ""


def _extract_json_block(text: str) -> str:
    """Extract the first JSON object from text that may contain markdown fences."""
    text = str(text)
    # Try ```json ... ``` first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try raw { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _load_json(text: str, fallback: dict) -> dict:
    try:
        value = json.loads(_extract_json_block(text))
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return fallback


def invoke_json_agent(
    state: ResearchState,
    prompt: str,
    agent,
    agent_name: str,
    node: str,
    fallback: dict,
    max_retries: int = 1,
) -> tuple[dict, str, list]:
    """Invoke an agent expecting JSON output, with retry on parse failure.

    Args:
        max_retries: How many additional attempts after the first failure
                     (default 1 ? 2 total calls max).
    """
    human = HumanMessage(content=with_memory_context(state, prompt))
    # Optimization: Do NOT pass state["messages"] to avoid token accumulation
    result = agent.invoke({"messages": [human]})
    tools, tool_outputs = collect_tool_calls(result["messages"])
    logger.info("%s ??: %s", colorize(f"[{node}]", "green"), ", ".join(tools) if tools else "?")
    for item in tool_outputs[:5]:
        logger.info("%s ????: %s", colorize(f"[{node}]", "green"), item[:400])
    logger.info("%s LLM??: ? | ??: ???", colorize(f"[{node}]", "yellow"))
    content = _last_content(result)
    emit(node, content)

    parsed = _load_json(content, fallback)

    # ?? Retry on parse failure ??
    for attempt in range(max_retries):
        if parsed is not fallback:
            break
        logger.warning(
            "%s JSON??????%d???...",
            colorize(f"[{node}]", "yellow"),
            attempt + 1,
        )
        retry_prompt = (
            "?????????? JSON?????????? JSON???? markdown????\n"
            f"?????\n{prompt}"
        )
        retry_human = HumanMessage(content=with_memory_context(state, retry_prompt))
        result = agent.invoke({"messages": [retry_human]})
        content = _last_content(result)
        emit(node, f"[retry {attempt + 1}] {content}")
        parsed = _load_json(content, fallback)

    return parsed, content, [human, result["messages"][-1]]


def estimate_tokens(text: str) -> int:
    """Rough token count: ~1 token per 2 CJK chars or 4 ASCII chars."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿" or "぀" <= ch <= "ヿ")
    ascii_chars = len(text) - cjk
    return (cjk // 2) + (ascii_chars // 4)


def check_token_budget(state: ResearchState, additional_tokens: int = 0) -> bool:
    """Check if adding more tokens would exceed budget."""
    budget = state.get("budget", {})
    max_tokens = budget.get("max_tokens", 12000)
    messages_text = " ".join(
        str(getattr(m, "content", "")) for m in state.get("messages", [])
    )
    current = estimate_tokens(messages_text) + additional_tokens
    return current < max_tokens


def bind_agent(node_func, agent, agent_name: str):
    return partial(node_func, agent=agent, agent_name=agent_name)
