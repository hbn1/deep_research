"""Memory reflection node: analyze completed tasks and extract patterns.

Runs after the final report. Accepts memory_manager explicitly
rather than relying on a module-level global.
"""

import logging
from ..state import ResearchState
from ..utils import colorize

logger = logging.getLogger("mult_agents")


def memory_reflect_node(state: ResearchState, agent, agent_name: str,
                        memory_manager=None) -> ResearchState:
    """Extract facts and procedural patterns from the completed research task.

    Args:
        state: Current ResearchState.
        agent: LLM agent instance (unused, kept for signature compatibility).
        agent_name: Agent display name.
        memory_manager: Optional MemoryManager for persistence.
    """
    logger.info("%s starting | agent=%s",
                colorize("[memory_reflect]", "cyan"),
                colorize(agent_name, "magenta"))

    query = state.get("query", "")
    final_report = state.get("final", "")

    if not final_report:
        logger.info("%s skipping - no final output",
                    colorize("[memory_reflect]", "yellow"))
        return {"memory_reflect_done": True}

    if memory_manager is None:
        logger.info("%s skipping - no memory manager configured",
                    colorize("[memory_reflect]", "yellow"))
        return {"memory_reflect_done": True}

    try:
        extracted = memory_manager._extractor.extract_from_turn(
            query=query,
            answer=final_report[:3000],
        )
        facts = extracted.get("facts", [])
        patterns = extracted.get("procedural", [])

        for pat in patterns:
            memory_manager.procedural.learn_pattern(
                user_id=state.get("user_id", "default_user"),
                trigger=pat.get("trigger", f"Research task: {query[:80]}"),
                action=pat.get("action", "Research completed successfully"),
                context=query[:200],
                importance=0.6,
                thread_id=state.get("tenant_id", "default"),
            )

        logger.info("%s extracted facts=%d patterns=%d",
                    colorize("[memory_reflect]", "green"),
                    len(facts), len(patterns))
        return {
            "memory_reflect_done": True,
            "extracted_facts": facts,
            "extracted_patterns": patterns,
        }
    except Exception as exc:
        logger.warning("%s failed (non-critical): %s",
                       colorize("[memory_reflect]", "yellow"), exc)

    return {"memory_reflect_done": True}
