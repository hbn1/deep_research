import json, logging
from datetime import date
from langchain_core.messages import HumanMessage
from ..state import ResearchState
from ..utils import (colorize, emit, collect_tool_calls, with_memory_context, log_inputs, invoke_json_agent, _last_content, _load_json)

logger = logging.getLogger('mult_agents')

def reflect_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[reflect]", "cyan"), colorize(agent_name, "magenta"))
    
    missing_gaps = state.get("missing_gaps", [])
    log_inputs("reflect", agent_name, {"missing_gaps": str(missing_gaps)})
    
    fallback = {
        "reflection_summary": "默认补搜",
        "supplementary_queries": [{"section_id": "gap_1", "query": state["query"], "source_preference": "hybrid", "reason": "fallback"}]
    }
    
    prompt = (
        f"当前日期：{date.today().isoformat()}；当前年份：{date.today().year}。如原问题要求当前、最新、近期或市面上，补搜词必须面向当前年份和近12个月。\n\n"
        f"分析师指出当前证据不足以完全回答问题，存在以下信息缺口：\n{json.dumps(missing_gaps, ensure_ascii=False)}\n\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"已执行过的搜索计划：\n{json.dumps(state.get('search_plan', []), ensure_ascii=False)}\n"
        f"已执行过的补搜计划：\n{json.dumps(state.get('supplementary_queries', []), ensure_ascii=False)}\n\n"
        "请生成新的补搜计划以填补缺口。"
    )
    
    payload, content, messages = invoke_json_agent(
        state,
        prompt,
        agent,
        agent_name,
        "reflect",
        fallback,
    )
    
    return {
        "iteration": state.get("iteration", 0) + 1,
        "supplementary_queries": payload.get("supplementary_queries", fallback["supplementary_queries"]),
        "messages": messages,
    }


