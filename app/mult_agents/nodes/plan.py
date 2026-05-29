import json, logging, re
from langchain_core.messages import HumanMessage
from ..state import ResearchState
from ..utils import (colorize, emit, collect_tool_calls, with_memory_context, log_inputs, invoke_json_agent, _last_content, _load_json)
from ._common import _extract_query_terms, _estimate_relevance
from .deep_dive import _dedupe_sources

logger = logging.getLogger('mult_agents')

def _default_plan(state: ResearchState) -> dict:
    return {
        "objective": state["query"],
        "sub_questions": [state["query"]],
        "outline": [
            {
                "id": "sec_1",
                "title": "默认大纲",
                "description": "默认生成的大纲",
                "section_type": "mixed",
                "requires_data": False,
                "requires_chart": False,
                "priority": 1,
                "search_queries": [state["query"]],
                "status": "pending",
            }
        ],
        "research_questions": [state["query"]],
        "budget": {"max_rounds": 2, "max_sources": 12, "max_tokens": 12000, "max_seconds": 45},
    }




def _guess_primary_entity(query: str) -> str:
    lowered = query.lower()
    ascii_terms = re.findall(r"[a-z][a-z0-9_-]{2,}", lowered)
    for term in ascii_terms:
        if term not in {"latest", "trend", "news", "agent", "open", "using"}:
            return term
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    for term in chinese_terms:
        if term not in {"帮我", "调查", "最新", "使用趋势", "是什么", "多少", "情况"}:
            return term
    return ""




def _derive_direct_search_queries(query: str) -> list[str]:
    base_query = query.strip()
    if not base_query:
        return []
    entity = _guess_primary_entity(base_query)
    candidates = [base_query]
    if entity:
        candidates.extend(
            [
                f"{entity}是什么",
                f"{entity} GitHub",
                f"{entity} 官方文档",
                f"{entity} 使用趋势",
                f"{entity} AI Agent",
            ]
        )
    else:
        candidates.extend(
            [
                f"{base_query} 是什么",
                f"{base_query} GitHub",
                f"{base_query} 官方文档",
            ]
        )
    deduped: list[str] = []
    for item in candidates:
        text = item.strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped[:6]




def _is_query_grounded(candidate: str, user_query: str) -> bool:
    candidate_terms = set(_extract_query_terms(candidate))
    user_terms = set(_extract_query_terms(user_query))
    if not candidate_terms or not user_terms:
        return False
    if _guess_primary_entity(user_query) and _guess_primary_entity(user_query) in candidate.lower():
        return True
    overlap = candidate_terms & user_terms
    return len(overlap) >= 1




def _derive_search_plan(outline: list[dict], sub_questions: list[str], _research_questions: list[str], query: str) -> list[dict]:
    plan: list[dict] = []
    for direct_query in _derive_direct_search_queries(query):
        plan.append(
            {
                "section_id": "user_query",
                "query": direct_query,
                "source_preference": "hybrid",
                "reason": "围绕用户原始问题生成的直接检索词",
            }
        )
    for section in outline:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "sec")
        for item in section.get("search_queries", []) or []:
            text = str(item).strip()
            if text and _is_query_grounded(text, query):
                plan.append(
                    {
                        "section_id": section_id,
                        "query": text,
                        "source_preference": "hybrid",
                        "reason": f"来自大纲章节 {section_id}",
                    }
                )
    if not plan:
        plan.append({"section_id": "sec_1", "query": query, "source_preference": "hybrid", "reason": "fallback"})
    deduped = _dedupe_sources(plan, ["query"])
    return deduped[:6]




def _build_queries(state: ResearchState, source_preference: str) -> list[dict]:
    """Build search queries from plan or supplementary plan.

    In re-search iterations, supplementary_queries are used as the plan.
    Each query's source_preference is respected:
      - "hybrid" ? dispatched to both web and local
      - "web" / "local" ? only to the matching side
    """
    queries: list[dict] = []

    iteration = state.get("iteration", 0)
    if iteration > 0 and state.get("supplementary_queries"):
        base_plan = state.get("supplementary_queries", [])
    else:
        base_plan = state.get("search_plan", [])

    for item in base_plan:
        if not isinstance(item, dict):
            continue
        pref = str(item.get("source_preference", "hybrid")).strip().lower()
        # Match: same preference or hybrid goes everywhere
        if pref == source_preference or pref == "hybrid":
            query = str(item.get("query", "")).strip()
            if query:
                queries.append(item)
    if not queries:
        queries.append({
            "section_id": "sec_1",
            "query": state["query"],
            "source_preference": source_preference,
            "reason": "fallback",
        })
    return queries[:6]




def plan_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[plan]", "cyan"), colorize(agent_name, "magenta"))
    log_inputs("plan", agent_name, {"query": state["query"]})
    
    # Fast path: simple queries use default plan (skip LLM)
    query = state["query"]
    simple_indicators = ["???", "???", "??", "????", "????", "??"]
    is_simple = any(w in query for w in simple_indicators) and len(query) < 40
    if is_simple:
        logger.info("%s ????: ?????? LLM ??", colorize("[plan]", "green"))
        fb = _default_plan(state)
        sp = _derive_search_plan(fb["outline"], fb["sub_questions"], fb["research_questions"], query)
        return {
            "phase": "planning completed (fast)",
            "plan": query,
            "outline": fb["outline"],
            "sub_questions": fb["sub_questions"],
            "research_questions": fb["research_questions"],
            "search_plan": sp,
            "budget": fb["budget"],
            "messages": [],
            "draft": "",
            "iteration": 0,
        }
    
    fallback = _default_plan(state)
    payload, content, messages = invoke_json_agent(
        state,
        f"?????{query}\n???????????????? JSON?",
        agent,
        agent_name,
        "plan",
        fallback,
    )
    outline = payload.get("outline") if isinstance(payload.get("outline"), list) else fallback["outline"]
    sub_questions = payload.get("sub_questions") if isinstance(payload.get("sub_questions"), list) else fallback["sub_questions"]
    research_questions = payload.get("research_questions") if isinstance(payload.get("research_questions"), list) else fallback["research_questions"]
    budget = payload.get("budget") if isinstance(payload.get("budget"), dict) else fallback["budget"]
    search_plan = _derive_search_plan(outline, sub_questions, research_questions, state["query"])
    plan_summary = payload.get("objective") or state["query"]
    return {
        "phase": "planning completed",
        "plan": plan_summary,
        "outline": outline,
        "sub_questions": sub_questions,
        "research_questions": research_questions,
        "search_plan": search_plan,
        "budget": budget,
        "messages": messages,
        "draft": content,
        "iteration": 0,
    }



