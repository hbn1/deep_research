"""Workflow orchestration: LangGraph nodes, conditional routing, and execution path."""
import logging
from functools import partial
from langgraph.graph import StateGraph, START, END

from .nodes import (
    prune_context_node,
    intent_node,
    direct_answer_node,
    plan_node,
    web_search_node,
    local_rag_node,
    deep_dive_node,
    analyze_node,
    reflect_node,
    write_node,
    memory_reflect_node,
)
from .utils import bind_agent
from .state import ResearchState


logger = logging.getLogger("mult_agents")


def route_after_intent(state: ResearchState) -> str:
    if state.get("intent") == "direct":
        return "direct_answer"
    return "plan"


def should_continue_research(state: ResearchState) -> str:
    """Decide whether to reflect (re-search) or write.
    
    Dynamic gating: only triggers re-search when missing_info_weight > 7,
    otherwise proceeds to write even if needs_more_research is True.
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)

    if iteration >= max_iter:
        return "write"

    needs_more = state.get("needs_more_research", False)
    missing_weight = state.get("missing_info_weight", 5)
    
    # Only re-search when gaps are critical (weight > 7)
    if needs_more and missing_weight > 7:
        return "reflect"

    return "write"


def route_after_write(state: ResearchState) -> str:
    """After writing, optionally run memory reflection.
    
    Only routes to memory_reflect when memory is actually enabled
    (memory_manager is passed to build_app and not None).
    """
    # memory_reflect is only added to the graph when memory_manager is not None,
    # so this route is only reached when memory is enabled.
    if not state.get("memory_reflect_done", False):
        return "memory_reflect"
    return "end"


def build_app(agents, checkpointer, memory_manager=None):
    """Build the compiled LangGraph workflow.

    Args:
        agents: AgentBundle with pre-built LLM agents.
        checkpointer: LangGraph checkpointer instance.
        memory_manager: Optional MemoryManager for the memory_reflect node.
    """
    workflow = StateGraph(ResearchState)

    workflow.add_node("intent", bind_agent(intent_node, agents.intent_router, "intent_router"))
    workflow.add_node("direct_answer", bind_agent(direct_answer_node, agents.direct_responder, "direct_responder"))
    workflow.add_node("plan", bind_agent(plan_node, agents.planner, "planner"))
    workflow.add_node("web_search", bind_agent(web_search_node, agents.scout_web, "scout_web"))
    workflow.add_node("local_rag", bind_agent(local_rag_node, agents.scout_local, "scout_local"))
    workflow.add_node("deep_dive", bind_agent(deep_dive_node, agents.evidence_judge, "evidence_judge"))
    workflow.add_node("analyze", bind_agent(analyze_node, agents.analyst, "analyst"))
    workflow.add_node("prune_context", bind_agent(prune_context_node, None, "pruner"))
    workflow.add_node("reflect", bind_agent(reflect_node, agents.planner, "planner"))
    workflow.add_node("write", bind_agent(write_node, agents.writer, "writer"))
    
    # Only add memory_reflect when memory is enabled
    _has_memory = memory_manager is not None
    if _has_memory:
        _memory_reflect = partial(memory_reflect_node, memory_manager=memory_manager)
        workflow.add_node("memory_reflect", bind_agent(_memory_reflect, agents.writer, "memory_reflect"))

    workflow.add_edge(START, "intent")
    workflow.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "direct_answer": "direct_answer",
            "plan": "plan",
        },
    )
    workflow.add_edge("plan", "web_search")
    workflow.add_edge("plan", "local_rag")
    workflow.add_edge("web_search", "deep_dive")
    workflow.add_edge("local_rag", "deep_dive")
    workflow.add_edge("deep_dive", "analyze")

    workflow.add_conditional_edges(
        "analyze",
        should_continue_research,
        {
            "reflect": "reflect",
            "write": "prune_context"
        }
    )

    workflow.add_edge("reflect", "web_search")
    workflow.add_edge("prune_context", "write")
    workflow.add_edge("reflect", "local_rag")
    workflow.add_edge("direct_answer", END)

    if _has_memory:
        workflow.add_conditional_edges(
            "write",
            route_after_write,
            {
                "memory_reflect": "memory_reflect",
                "end": END,
            }
        )
        workflow.add_edge("memory_reflect", END)
    else:
        workflow.add_edge("write", END)

    return workflow.compile(checkpointer=checkpointer)
