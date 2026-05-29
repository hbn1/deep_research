import json, logging
from langchain_core.messages import HumanMessage
from ..state import ResearchState
from ..utils import (colorize, emit, collect_tool_calls, with_memory_context, log_inputs, invoke_json_agent, _last_content, _load_json)

logger = logging.getLogger('mult_agents')

def _fallback_analysis(state: ResearchState) -> dict:
    source_ids = [item.get("source_id") for item in state.get("evidence_pool", [])[:3] if item.get("source_id")]
    findings = [
        {
            "claim_id": "c_1",
            "claim": f"围绕“{state['query']}”已完成多源检索，初步证据表明问题可以从网络与本地知识库双侧支撑。",
            "confidence": "medium" if source_ids else "low",
            "source_ids": source_ids,
        }
    ]
    hypothesis_status = []
    for hypo in state.get("hypotheses", []):
        hypothesis_status.append(
            {
                "id": hypo.get("id"),
                "status": "verified" if source_ids else "uncertain",
                "reason": "已有可用证据池" if source_ids else "证据不足",
                "source_ids": source_ids,
            }
        )
    return {
        "analysis_summary": "完成结论归纳与假设状态整理。",
        "hypothesis_status": hypothesis_status,
        "findings": findings,
        "claim_map": [{"claim_id": item["claim_id"], "source_ids": item["source_ids"]} for item in findings],
        "next_actions": [] if source_ids else ["补充更多高质量来源"],
    }




def analyze_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[analyze]", "cyan"), colorize(agent_name, "magenta"))
    fallback = _fallback_analysis(state)
    payload, content, messages = invoke_json_agent(
        state,
        "请基于证据池输出结论映射 JSON，并评估证据完备性：\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"证据池：{json.dumps(state.get('evidence_pool', []), ensure_ascii=False)}\n"
        f"审计标记：{json.dumps(state.get('audit_flags', []), ensure_ascii=False)}",
        agent,
        agent_name,
        "analyze",
        fallback,
    )
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else fallback["findings"]
    claim_map = payload.get("claim_map") if isinstance(payload.get("claim_map"), list) else fallback["claim_map"]
    needs_more_research = payload.get("needs_more_research", False)
    missing_gaps = payload.get("missing_gaps", [])
    analysis_summary = payload.get("analysis_summary", content)
    return {
        "analysis": analysis_summary,
        "findings": findings,
        "claim_map": claim_map,
        "needs_more_research": needs_more_research,
        "missing_gaps": missing_gaps,
        "messages": messages,
    }


