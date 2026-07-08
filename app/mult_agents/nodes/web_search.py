import json, logging
from langchain_core.messages import HumanMessage
from ..state import ResearchState
from ..tools import bocha_web_search_records, search_knowledge_base_records
from ..utils import (colorize, emit, collect_tool_calls, with_memory_context, log_inputs, invoke_json_agent, _last_content, _load_json)
from ._common import (_minimal_record_filter, _assign_source_ids, _format_raw_records, _summarize_records, _normalize_source_ids, _finalize_query_traces)
from ._common import _estimate_relevance, _prune_evidence_to_allowed_sources, _enrich_evidence_from_raw
from .plan import _build_queries
from .deep_dive import _dedupe_sources, _is_official_domain

logger = logging.getLogger('mult_agents')


def _resolve_search_node_timeout(config) -> float:
    from os import getenv

    override = getenv("SEARCH_NODE_TIMEOUT", "").strip()
    if override:
        try:
            return max(1.0, float(override))
        except ValueError:
            logger.warning("Invalid SEARCH_NODE_TIMEOUT=%s, using derived timeout", override)
    fetch_budget = config.fetch_timeout if getattr(config, "fetch_enabled", True) else 0.0
    rewrite_budget = 6.0 if getattr(config, "rewrite_enabled", False) else 2.0
    return max(3.0, min(float(config.request_timeout) + float(fetch_budget) + rewrite_budget, 25.0))


def _is_bad_web_domain(domain: str) -> bool:
    value = domain.lower()
    blocked = ["datasheet", "bdtic", "doc88", "elecfans", "down"]
    return any(item in value for item in blocked)




def _filter_web_records(query: str, records: list[dict]) -> tuple[list[dict], dict]:
    kept = []
    stats = {"raw_count": len(records), "kept_count": 0, "dropped_irrelevant": 0, "dropped_domain": 0, "dropped_empty": 0}
    for record in records:
        title = str(record.get("title", ""))
        snippet = str(record.get("snippet", ""))
        domain = str(record.get("domain", ""))
        if not title and not snippet:
            stats["dropped_empty"] += 1
            continue
        if _is_bad_web_domain(domain):
            stats["dropped_domain"] += 1
            continue
        relevance = _estimate_relevance(query, f"{title}\n{snippet}")
        record["relevance_score"] = relevance
        if relevance < 0.2 and not _is_official_domain(domain):
            stats["dropped_irrelevant"] += 1
            continue
        kept.append(record)
    stats["kept_count"] = len(kept)
    return kept, stats


def _filter_web_records_any(queries: list[str], records: list[dict]) -> tuple[list[dict], dict]:
    kept = []
    stats = {"raw_count": len(records), "kept_count": 0, "dropped_irrelevant": 0, "dropped_domain": 0, "dropped_empty": 0}
    query_texts = [q for q in queries if str(q).strip()] or [""]
    for record in records:
        title = str(record.get("title", ""))
        snippet = str(record.get("snippet", ""))
        domain = str(record.get("domain", ""))
        if not title and not snippet:
            stats["dropped_empty"] += 1
            continue
        if _is_bad_web_domain(domain):
            stats["dropped_domain"] += 1
            continue
        relevance = max(_estimate_relevance(q, f"{title}\n{snippet}") for q in query_texts)
        record["relevance_score"] = relevance
        if relevance < 0.2 and not _is_official_domain(domain):
            stats["dropped_irrelevant"] += 1
            continue
        kept.append(record)
    stats["kept_count"] = len(kept)
    return kept, stats




def _fallback_web_evidence(records: list[dict]) -> dict:
    evidence = []
    for record in records:
        evidence.append(
            {
                "source_id": record.get("source_id"),
                "title": record.get("title"),
                "url": record.get("url", ""),
                "snippet": record.get("snippet", ""),
                "domain": record.get("domain", ""),
                "source_type": "web",
                "reliability_hint": "official" if _is_official_domain(record.get("domain", "")) else "unknown",
                "supports": [],
                "supports_questions": record.get("supports_questions", []),
                "notes": "",
            }
        )
    return {"summary": "Web evidence collection complete", "evidence": evidence, "gaps": []}




def web_search_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """Enterprise-grade web search: parallel queries, multi-backend, cache, fetch, rewrite, rerank."""
    logger.info("%s ?? | agent=%s", colorize("[web_search]", "cyan"), colorize(agent_name, "magenta"))
    queries = _build_queries(state, "web")
    logger.info("[web_search_node] ???? | ????=%s | queries=%s",
                len(queries), [q.get("query", "") for q in queries])

    iteration = state.get("iteration", 0)
    prefix = f"WEB{iteration+1}"
    query_traces = state.get("web_search_trace", [])

    # ?? Extract query texts ??
    query_texts = [str(item.get("query", "")) for item in queries if str(item.get("query", "")).strip()]
    if not query_texts:
        query_texts = [state["query"]]

    # ?? Parallel search with enterprise engine ??
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
    from ..search import get_search_config, search as enterprise_search

    all_raw_records: list[dict] = []
    tenant_id = str(state.get("tenant_id", "default_tenant") or "default_tenant")
    query_index_map: dict[str, int] = {}
    for idx, item in enumerate(queries, 1):
        qtext = str(item.get("query", ""))
        if qtext:
            query_index_map[qtext] = idx

    search_config = get_search_config()
    search_timeout = _resolve_search_node_timeout(search_config)
    with ThreadPoolExecutor(max_workers=max(1, min(len(query_texts), 6))) as executor:
        futures = {}
        for qtext in query_texts:
            futures[executor.submit(
                enterprise_search, qtext,
                enable_cache=search_config.cache_enabled,
                enable_rewrite=search_config.rewrite_enabled,
                enable_fetch=search_config.fetch_enabled,
                enable_rerank=True,
                tenant_id=tenant_id,
            )] = qtext

        try:
            for future in as_completed(futures, timeout=search_timeout):
                qtext = futures[future]
                query_idx = query_index_map.get(qtext, 1)
                try:
                    records = future.result()
                except Exception as exc:
                    logger.warning("[web_search_node] search failed for query=%s: %s", qtext[:60], exc)
                    records = []

                # Assign source_ids
                records = _assign_source_ids(records, f"{prefix}_{query_idx}")
                for record in records:
                    record["section_id"] = query_idx
                    record["search_query"] = qtext

                logger.info("[web_search_node] ???? | query=%s | results=%s", qtext[:60], len(records))
                query_traces.append({
                    "iteration": iteration,
                    "plan_step": query_idx,
                    "query": qtext,
                    "section_id": query_idx,
                    "source_preference": "web",
                    "raw_count": len(records),
                    "raw_records": _summarize_records(records),
                })
                all_raw_records.extend(records)
        except TimeoutError:
            logger.warning("[web_search_node] ?????? (%.1fs)???????", search_timeout)

    # ?? Deduplicate and filter ??
    all_raw_records = _dedupe_sources(all_raw_records, ["url", "title"])
    all_raw_records = _minimal_record_filter(all_raw_records, ["title", "snippet", "url"])

    # ?? Programmatic pre-filter before LLM ??
    all_raw_records, filter_stats = _filter_web_records_any(query_texts, all_raw_records)

    logger.info("[web_search_node] ????? | ????????=%s", len(all_raw_records))

    web_retrieval_stats = state.get("web_retrieval_stats", {})
    web_retrieval_stats["query_count"] = web_retrieval_stats.get("query_count", 0) + len(query_texts)
    web_retrieval_stats["raw_count"] = web_retrieval_stats.get("raw_count", 0) + len(all_raw_records)

    log_inputs("web_search", agent_name, {"query_count": str(len(query_texts)), "raw_count": str(len(all_raw_records))})

    if not all_raw_records:
        logger.info("%s ?????????????????", colorize("[web_search]", "yellow"))
        return {
            "web_search": "??????????????????????",
            "web_evidence": state.get("web_evidence", []),
            "web_retrieval_stats": web_retrieval_stats,
            "web_search_trace": query_traces,
        }

    # ?? Programmatic evidence construction (skip redundant LLM filtering) ??
    # enterprise_search already did rewrite + multi-backend + rerank.
    # We construct evidence directly without an extra LLM round-trip.
    logger.info("[web_search_node] ?????????? LLM ???| raw_records=%s", len(all_raw_records))
    evidence = _fallback_web_evidence(all_raw_records)["evidence"]
    content = "???????"
    messages = []

    # ?? Safety: prune to allowed source_ids, enrich missing fields ??
    allowed_source_ids = {str(item.get("source_id")) for item in all_raw_records if item.get("source_id")}
    evidence = _prune_evidence_to_allowed_sources(evidence, allowed_source_ids)
    evidence = _enrich_evidence_from_raw(evidence, all_raw_records)

    web_retrieval_stats["kept_count"] = web_retrieval_stats.get("kept_count", 0) + len(evidence)
    web_retrieval_stats["dropped_count"] = web_retrieval_stats.get("dropped_count", 0) + max(len(all_raw_records) - len(evidence), 0)

    kept_ids = {str(item.get("source_id")) for item in evidence if item.get("source_id")}
    query_traces = _finalize_query_traces(
        query_traces, kept_ids,
        [],  # programmatic filtering (no payload)
        "",  # programmatic filtering
    )

    existing_evidence = state.get("web_evidence", [])
    logger.info("[web_search_node] ???? | ????=%s | ????=%s",
                len(evidence), len(existing_evidence) + len(evidence))
    return {
        "web_search": content,
        "web_evidence": existing_evidence + evidence,
        "web_retrieval_stats": web_retrieval_stats,
        "web_search_trace": query_traces,
        "messages": messages,
    }



