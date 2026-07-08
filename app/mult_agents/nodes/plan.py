import json, logging, re
from datetime import date
from langchain_core.messages import HumanMessage
from ..state import ResearchState
from ..utils import (colorize, emit, collect_tool_calls, with_memory_context, log_inputs, invoke_json_agent, _last_content, _load_json)
from ._common import _extract_query_terms, _estimate_relevance
from .deep_dive import _dedupe_sources

logger = logging.getLogger('mult_agents')

_CURRENT_QUERY_MARKERS = (
    "当前", "最新", "现在", "当下", "近期", "最近", "今年", "市面上", "现状",
    "current", "latest", "recent", "today", "up-to-date",
)


def _current_year() -> int:
    return date.today().year


def _current_date_text() -> str:
    return date.today().isoformat()


def _user_specified_year(query: str) -> bool:
    return bool(re.search(r"20\d{2}", query))


def _needs_current_context(query: str) -> bool:
    lowered = query.lower()
    return any(marker in lowered for marker in _CURRENT_QUERY_MARKERS)


def _augment_query_with_current_context(query: str) -> str:
    if not _needs_current_context(query):
        return query
    return (
        f"{query}\n\n"
        f"当前日期：{_current_date_text()}；当前年份：{_current_year()}。"
        "如果用户要求“当前、最新、现在、当下、今年、近期、市面上、现状”，"
        "规划、子问题和搜索词必须面向当前年份和近 12 个月，"
        "除非用户显式指定旧年份，否则禁止把任务改写成 2024 或其他过去年份。"
    )


def _normalize_current_text(value: str, user_query: str) -> str:
    if not _needs_current_context(user_query) or _user_specified_year(user_query):
        return value
    current = _current_year()

    def _replace_year(match: re.Match) -> str:
        year = int(match.group(0))
        if 2020 <= year < current:
            return str(current)
        return match.group(0)

    return re.sub(r"20\d{2}", _replace_year, value)


def _normalize_current_payload(value, user_query: str):
    if isinstance(value, str):
        return _normalize_current_text(value, user_query)
    if isinstance(value, list):
        return [_normalize_current_payload(item, user_query) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_current_payload(item, user_query) for key, item in value.items()}
    return value

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
    if "agent" in lowered or "智能体" in query:
        return "Agent智能体"
    ascii_terms = re.findall(r"[a-z][a-z0-9_-]{2,}", lowered)
    for term in ascii_terms:
        if term not in {"latest", "trend", "news", "open", "using", "current", "recent"}:
            return term
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    for term in chinese_terms:
        if term not in {"帮我", "调查", "最新", "使用趋势", "是什么", "多少", "情况"}:
            return term
    return ""




def _derive_direct_search_queries(query: str) -> list[str]:
    """Generate direct search queries from user query.
    
    Uses simple heuristics: the original query + entity-focused variants.
    No longer generates corrupted Chinese suffixes.
    """
    base_query = query.strip()
    if not base_query:
        return []
    entity = _guess_primary_entity(base_query)
    current_year = _current_year()
    lowered = base_query.lower()
    candidates = []
    if _needs_current_context(base_query):
        if "agent" in lowered or "智能体" in base_query:
            candidates.extend([
                f"{current_year} Agent 智能体 市场 最新趋势",
                f"{current_year} AI Agent platforms enterprise adoption LangChain AutoGen CrewAI OpenAI Agents SDK",
                f"{current_year} Agent 智能体 企业落地 案例 产品 对比",
            ])
        elif entity:
            candidates.extend([
                f"{current_year} {entity} 最新趋势 现状",
                f"{current_year} {entity} 市场 对比 分析",
            ])
    candidates.append(base_query)
    if entity and len(entity) >= 3:
        # Generate clean, language-appropriate search variants
        candidates.extend([
            f"{current_year} {entity} 最新 overview",
            f"{current_year} {entity} comparison",
            f"{entity} best practices",
        ])
    # Deduplicate preserving order
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result[:4]  # Limit to 4 queries max

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
    if _needs_current_context(query):
        return _dedupe_sources(plan, ["query"])[:3]
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
    return deduped[:4]




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
    return queries[:4]




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
    query = _augment_query_with_current_context(query)
    payload, content, messages = invoke_json_agent(
        state,
        f"?????{query}\n???????????????? JSON?",
        agent,
        agent_name,
        "plan",
        fallback,
    )
    payload = _normalize_current_payload(payload, state["query"])
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



