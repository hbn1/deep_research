import json, logging
from langchain_core.messages import HumanMessage
from ..state import ResearchState
from ..tools import bocha_web_search_records, search_knowledge_base_records
from ..utils import (colorize, emit, collect_tool_calls, with_memory_context, log_inputs, invoke_json_agent, _last_content, _load_json)
from ._common import (_minimal_record_filter, _assign_source_ids, _format_raw_records, _summarize_records, _normalize_source_ids, _finalize_query_traces)

logger = logging.getLogger('mult_agents')

def _filter_local_records(query: str, records: list[dict]) -> tuple[list[dict], dict]:
    kept = []
    stats = {"raw_count": len(records), "kept_count": 0, "dropped_irrelevant": 0, "dropped_missing_doc": 0, "dropped_empty": 0}
    for record in records:
        title = str(record.get("title", ""))
        snippet = str(record.get("snippet", ""))
        doc_id = str(record.get("doc_id", "")).strip()
        if not snippet:
            stats["dropped_empty"] += 1
            continue
        relevance = _estimate_relevance(query, f"{title}\n{snippet}")
        record["relevance_score"] = relevance
        if not doc_id and relevance < 0.35:
            stats["dropped_missing_doc"] += 1
            continue
        if relevance < 0.2:
            stats["dropped_irrelevant"] += 1
            continue
        kept.append(record)
    stats["kept_count"] = len(kept)
    return kept, stats




def _fallback_local_evidence(records: list[dict]) -> dict:
    evidence = []
    for record in records:
        evidence.append(
            {
                "source_id": record.get("source_id"),
                "doc_id": record.get("doc_id", ""),
                "title": record.get("title", "") or record.get("source_id", ""),
                "snippet": record.get("snippet", ""),
                "source_type": "local",
                "reliability_hint": "internal",
                "supports": [],
                "supports_questions": record.get("supports_questions", []),
                "notes": "",
            }
        )
    return {"summary": "完成本地知识库证据采集。", "evidence": evidence, "gaps": []}




def local_rag_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[local_rag]", "cyan"), colorize(agent_name, "magenta"))
    queries = _build_queries(state, "local")
    raw_records = []
    query_traces = state.get("local_rag_trace", [])
    
    iteration = state.get("iteration", 0)
    prefix = f"LOC{iteration+1}"
    
    for query_index, item in enumerate(queries, 1):
        records = search_knowledge_base_records(str(item.get("query", "")), limit=4)
        records = _assign_source_ids(records, f"{prefix}_{query_index}")
        for record in records:
            record["section_id"] = item.get("section_id")
            record["search_query"] = item.get("query")
        raw_records.extend(records)
        query_traces.append(
            {
                "iteration": iteration,
                "plan_step": query_index,
                "query": str(item.get("query", "")),
                "section_id": item.get("section_id"),
                "reason": item.get("reason", ""),
                "source_preference": item.get("source_preference", "local"),
                "raw_count": len(records),
                "raw_records": _summarize_records(records),
            }
        )
    raw_records = _dedupe_sources(raw_records, ["doc_id", "snippet"])
    raw_records = _minimal_record_filter(raw_records, ["snippet", "title", "doc_id"])
    
    local_retrieval_stats = state.get("local_retrieval_stats", {})
    local_retrieval_stats["query_count"] = local_retrieval_stats.get("query_count", 0) + len(queries)
    local_retrieval_stats["raw_count"] = local_retrieval_stats.get("raw_count", 0) + len(raw_records)
    
    log_inputs("local_rag", agent_name, {"query_count": str(len(queries)), "raw_count": str(len(raw_records))})
    if not raw_records:
        logger.info("%s 无可用本地证据，跳过本地上下文注入", colorize("[local_rag]", "yellow"))
        return {
            "local_rag": "未检索到可用本地知识库证据，已跳过本地上下文注入。",
            "local_evidence": state.get("local_evidence", []),
            "local_retrieval_stats": local_retrieval_stats,
            "local_rag_trace": query_traces,
        }
    # Programmatic evidence construction (skip LLM filtering - same as web_search_node)
    logger.info("[local_rag_node] ????????? (?? LLM) | raw_records=%s", len(raw_records))
    evidence = _fallback_local_evidence(raw_records)["evidence"]
    allowed_source_ids = {str(item.get("source_id")) for item in raw_records if item.get("source_id")}
    evidence = _prune_evidence_to_allowed_sources(evidence, allowed_source_ids)
    evidence = _enrich_evidence_from_raw(evidence, raw_records)
    content = "???????"
    messages = []
    
    local_retrieval_stats["kept_count"] = local_retrieval_stats.get("kept_count", 0) + len(evidence)
    local_retrieval_stats["dropped_count"] = local_retrieval_stats.get("dropped_count", 0) + max(len(raw_records) - len(evidence), 0)
    
    kept_ids = {str(item.get("source_id")) for item in evidence if item.get("source_id")}
    query_traces = _finalize_query_traces(
        query_traces,
        kept_ids,
        [],  # programmatic filtering (no payload)
        "",  # programmatic filtering
    )
    
    existing_evidence = state.get("local_evidence", [])
    return {
        "local_rag": content,  # programmatic filtering
        "local_evidence": existing_evidence + evidence,
        "local_retrieval_stats": local_retrieval_stats,
        "local_rag_trace": query_traces,
        "messages": messages,
    }



