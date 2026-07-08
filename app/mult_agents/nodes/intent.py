import json, logging, re
from ..state import ResearchState
from ..utils import (colorize, emit, collect_tool_calls, with_memory_context, log_inputs, invoke_json_agent, _last_content, _load_json)

logger = logging.getLogger('mult_agents')


def _looks_like_simple_arithmetic_query(query: str) -> bool:
    stripped = query.strip()
    if not stripped or len(stripped) > 80:
        return False
    normalized = stripped.lower()
    normalized = re.sub(r"^(answer only|calculate)\s*:?\s*", "", normalized)
    normalized = re.sub(r"^(what is|what's)\s+", "", normalized)
    normalized = re.sub(r"^(please\s+)?compute\s+", "", normalized)
    normalized = re.sub(r"^(请计算|计算|算一下|帮我算一下)\s*[:：]?\s*", "", normalized)
    normalized = normalized.replace("等于多少", "").replace("是多少", "")
    normalized = normalized.strip().rstrip("=?？。")
    if not normalized:
        return False
    if not re.fullmatch(r"[0-9\s+\-*/%().]+", normalized):
        return False
    return any(op in normalized for op in "+-*/%")


def is_confident_direct_query(query: str) -> bool:
    """Return True when routing can be decided without an LLM call."""
    stripped = query.strip()
    q = stripped.lower()
    if not q:
        return True
    if _looks_like_simple_arithmetic_query(stripped):
        return True

    direct_patterns = [
        r'^(hi|hello|hey)\b',
        r'^who are you\b',
        r'^what can you do\b',
        r'^(thanks|thank you|thx)\b',
        r'^(bye|goodbye)\b',
        r'^weather\b',
        r'^what time\b',
        r'^calculate\s',
        r'^translate\b',
    ]
    if any(re.search(pattern, q) for pattern in direct_patterns):
        return True

    direct_cn = [
        "你好", "您好", "谢谢", "再见", "你是谁", "你能做什么",
        "介绍一下你自己", "天气", "时间", "日期", "现在几点",
    ]
    return len(stripped) <= 12 and any(token in stripped for token in direct_cn)


def detect_intent(query: str) -> str:
    """Rule-based intent detection. Fast path: skip LLM when confident.

    Strategy: only route multiagent for queries that clearly need
    multi-source research, evidence gathering, or deep analysis.
    Simple factual Q&A, greetings, definitions go direct.
    """
    q = query.strip()
    if not q:
        return "direct"
    if _looks_like_simple_arithmetic_query(q):
        return "direct"

    # ---- Direct (simple) signals (ASCII-safe patterns) ----
    direct_patterns = [
        r'^(hi|hello|hey)\b',
        r'^who are you',
        r'^what can you do',
        r'^(thanks|thank you|thx)',
        r'^(bye|goodbye)',
        r'^weather',
        r'^what time',
        r'^calculate\s',
        r'^translate',
    ]
    for pat in direct_patterns:
        if re.search(pat, q.lower()):
            return "direct"

    # ---- Direct: Chinese simple signals (substring match) ----
    direct_cn = [
        "你好",    # ??
        "谢谢",    # ??
        "再见",    # ??
        "你是谁",  # ???
        "你能做什么",  # ?????
        "介绍一下你自己",  # ???????
        "天气怎么样",  # 天气
        "天气",       # 天气
        "时间",       # 时间
        "现在几点",      # ????
    ]
    # ---- Simple Chinese queries (weather, time, date) ----
    simple_cn_patterns = [
        "天气", "时间", "日期", "几点", "星期几",
    ]
    if any(p in q for p in simple_cn_patterns):
        return "direct"

    # Ultra-short greetings (<=4 chars)
    if len(q) <= 4:
        for phrase in direct_cn:
            if phrase in q:
                return "direct"
        return "direct"

    # Research keywords (substring match, not regex)
    research_keywords = [
        "分析",     # ??
        "对比",     # ??
        "趋势",     # ??
        "研究",     # ??
        "调查",     # ??
        "报告",     # ??
        "方案",     # ??
        "盘点",     # ??
        "架构",     # ??
        "有哪些",   # ???
        "哪些",     # ??
        "推荐",     # ??
        "主流",     # ??
        "最新",     # ??
        "当前",     # ??
        "最近",     # ??
        "近期",     # ??
        "排名",     # ??
        "榜单",     # ??
        "评价",     # ??
        "评测",     # ??
        "选型",     # ??
        "落地",     # ??
        "实践",     # ??
        "案例",     # ??
        "代码",     # ??
        "实现",     # ??
        "设计",     # ??
        "原理",     # ??
        "流程",     # ??
        "versus",
        " vs ",               # English comparison
    ]
    has_any_research = any(kw in q for kw in research_keywords)

    # No research keyword at all -> direct
    if not has_any_research:
        return "direct"

    # ---- Strong multiagent signals ----
    strong_signals = [
        "调研报告",     # ????
        "行业分析",     # ????
        "竞品分析",     # ????
        "市场调查",     # ????
        "趋势分析",     # ????
        "深度研究",     # ????
        "比较分析",     # ????
        "对比评测",     # ????
        "技术选型",     # ????
        "架构对比",     # ????
        "方案对比",     # ????
        "最新进展",     # ????
        "最新趋势",     # ????
        "重大新闻",     # ????
        "有哪些",           # ???
        "推荐",                 # ??
        "排名",                 # ??
        "榜单",                 # ??
    ]
    for signal in strong_signals:
        if signal in q:
            return "multiagent"

    # ---- Moderate signals: need >=2 ----
    moderate_signals = [
        "分析",     # ??
        "对比",     # ??
        "趋势",     # ??
        "研究",     # ??
        "调查",     # ??
        "报告",     # ??
        "方案",     # ??
        "盘点",     # ??
        "架构",     # ??
        "主流",     # ??
        "最新",     # ??
        "当前",     # ??
        "评测",     # ??
        "选型",     # ??
        "案例",     # ??
        "实现",     # ??
        "设计",     # ??
        "原理",     # ??
        "流程",     # ??
        "评价",     # ??
        "排名",     # ??
        " versus ",
        " vs ",               # English comparison
    ]
    moderate_count = sum(1 for w in moderate_signals if w in q)

    # Time markers amplify moderate signals
    time_markers = [
        r'20\d{2}', "今年", "去年",      # ??, ??
        "最近", "近期", "当前",   # ??, ??, ??
        "最新",                                    # ??
    ]
    has_time = any(
        re.search(m, q) if m.startswith("20") else m in q
        for m in time_markers
    )

    source_markers = ["GitHub", "github", "官方", "文档", "来源", "出处"]
    has_source = any(m in q for m in source_markers)

    if moderate_count >= 2:
        return "multiagent"
    if moderate_count >= 1 and (has_time or has_source):
        return "multiagent"

    # Remaining: if has moderate signal AND enough substance
    # Substance = length > 12, OR contains English technical terms
    has_english_terms = bool(re.search(r'[A-Za-z][A-Za-z0-9_-]{2,}', q))
    if moderate_count >= 1 and (len(q) > 12 or has_english_terms):
        return "multiagent"

    # Default: direct for simple/ambiguous queries
    return "direct"




def intent_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s ?? | agent=%s", colorize("[intent]", "cyan"), colorize(agent_name, "magenta"))
    rule_route = detect_intent(state["query"])
    # Trust rule-based detection to skip unnecessary LLM call
    if rule_route == "multiagent":
        logger.info("%s ??: multiagent (????? LLM)", colorize("[intent]", "green"))
        return {"intent": "multiagent", "draft": "", "messages": []}
    if rule_route == "direct" and is_confident_direct_query(state["query"]):
        logger.info("%s route: direct (confident rule, skip LLM)", colorize("[intent]", "green"))
        return {"intent": "direct", "draft": "", "messages": []}
    # Only call LLM for ambiguous cases (rule says direct but might be wrong)
    prompt = (
        f"?????{state['query']}\n"
        "??? JSON?{\"route\":\"direct|multiagent\",\"reason\":\"...\"}"
    )
    payload, content, messages = invoke_json_agent(
        state,
        prompt,
        agent,
        agent_name,
        "intent",
        {"route": "direct", "reason": "rule"},
    )
    route = str(payload.get("route", "direct")).strip().lower()
    if route not in {"direct", "multiagent"}:
        route = "direct"
    logger.info("%s ??: %s", colorize("[intent]", "green"), route)
    return {"intent": route, "draft": content, "messages": messages}



