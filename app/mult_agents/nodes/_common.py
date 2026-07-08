import json, logging, re

logger = logging.getLogger('mult_agents')

def _minimal_record_filter(records: list[dict], required_any: list[str]) -> list[dict]:
    kept: list[dict] = []
    for record in records:
        if any(str(record.get(field, "")).strip() for field in required_any):
            kept.append(record)
    return kept




def _assign_source_ids(records: list[dict], prefix: str) -> list[dict]:
    assigned: list[dict] = []
    for index, record in enumerate(records, 1):
        item = dict(record)
        item["source_id"] = f"{prefix}-{index}"
        assigned.append(item)
    return assigned




def _format_raw_records(records: list[dict], source_type: str) -> str:
    if not records:
        return "[]"
    lines = []
    for record in records[:40]:
        locator = record.get("url") or record.get("doc_id") or ""
        lines.append(
            json.dumps(
                {
                    "source_id": record.get("source_id"),
                    "title": record.get("title"),
                    "url": record.get("url", ""),
                    "doc_id": record.get("doc_id", ""),
                    "snippet": str(record.get("snippet", ""))[:500],
                    "source_type": source_type,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)




def _summarize_records(records: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for record in records[:5]:
        summary.append(
            {
                "source_id": record.get("source_id"),
                "title": record.get("title", ""),
                "locator": record.get("url") or record.get("doc_id") or "",
                "snippet": str(record.get("snippet", ""))[:160],
            }
        )
    return summary




def _normalize_source_ids(values) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized




def _finalize_query_traces(query_traces: list[dict], kept_ids: set[str], rejected_ids: list[str], reject_reason: str) -> list[dict]:
    normalized_rejected = set(_normalize_source_ids(rejected_ids))
    finalized: list[dict] = []
    for trace in query_traces:
        raw_items = [item for item in trace.get("raw_records", []) if isinstance(item, dict)]
        kept_records = [item for item in raw_items if str(item.get("source_id", "")).strip() in kept_ids]
        rejected_records = [
            item
            for item in raw_items
            if str(item.get("source_id", "")).strip() in normalized_rejected or str(item.get("source_id", "")).strip() not in kept_ids
        ]
        trace_item = dict(trace)
        trace_item["raw_source_ids"] = _normalize_source_ids(item.get("source_id") for item in raw_items)
        trace_item["kept_source_ids"] = _normalize_source_ids(item.get("source_id") for item in kept_records)
        trace_item["rejected_source_ids"] = _normalize_source_ids(item.get("source_id") for item in rejected_records)
        trace_item["kept_count"] = len(trace_item["kept_source_ids"])
        trace_item["rejected_count"] = len(trace_item["rejected_source_ids"])
        trace_item["kept_records"] = kept_records[:3]
        trace_item["rejected_records"] = rejected_records[:3]
        if reject_reason:
            trace_item["reject_reason"] = reject_reason
        finalized.append(trace_item)
    return finalized




def _enrich_evidence_from_raw(evidence: list[dict], raw_records: list[dict]) -> list[dict]:
    """从原始记录中补充 evidence 中可能丢失的 url、domain 等字段"""
    raw_lookup = {str(r.get("source_id", "")).strip(): r for r in raw_records if r.get("source_id")}
    enriched = []
    for ev in evidence:
        item = dict(ev)
        sid = str(item.get("source_id", "")).strip()
        raw = raw_lookup.get(sid, {})
        # 补充 url（如 LLM 没有保留）
        if not item.get("url") and raw.get("url"):
            item["url"] = raw["url"]
        # 补充 domain
        if not item.get("domain") and raw.get("domain"):
            item["domain"] = raw["domain"]
        # 补充 title（如 LLM 没有保留）
        if not item.get("title") and raw.get("title"):
            item["title"] = raw["title"]
        enriched.append(item)
    return enriched




def _prune_evidence_to_allowed_sources(evidence: list[dict], allowed_source_ids: set[str]) -> list[dict]:
    kept: list[dict] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        if source_id and source_id in allowed_source_ids:
            kept.append(item)
    return kept




def _extract_query_terms(query: str) -> list[str]:
    lowered = query.lower()
    terms: list[str] = []
    phrase_hints = []
    if "agent" in lowered:
        phrase_hints.append("agent")
    if "智能体" in query:
        phrase_hints.append("智能体")
    if "人工智能" in query:
        phrase_hints.append("人工智能")

    generic_fragments = [
        "写一份", "当前", "市面上", "最新", "现在", "当下", "近期", "最近",
        "今年", "分析报告", "调研报告", "研究报告", "分析", "报告", "的",
    ]
    stopwords = {
        "什么", "如何", "以及", "一个", "关于", "这个", "那个", "进行", "基于",
        "附带", "来源", "清单", "当前", "最新", "现在", "当下", "近期", "最近",
        "今年", "市面上", "分析", "报告", "写一份", "生成", "调研", "研究",
        "latest", "current", "recent", "today", "overview", "comparison",
        "performance", "best", "practices",
    }

    for hint in phrase_hints:
        if hint not in terms:
            terms.append(hint)

    for part in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_-]{3,}", lowered):
        if re.fullmatch(r"[A-Za-z0-9_-]{3,}", part):
            if part not in stopwords and part not in terms:
                terms.append(part)
            continue
        cleaned = part
        for fragment in generic_fragments:
            cleaned = cleaned.replace(fragment, " ")
        sub_terms = re.findall(r"[\u4e00-\u9fff]{2,}", cleaned)
        if not sub_terms and cleaned.strip():
            sub_terms = [part]
        for sub_term in sub_terms:
            if sub_term in stopwords:
                continue
            if sub_term not in terms:
                terms.append(sub_term)
    if terms:
        return terms[:12]

    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_-]{3,}", query.lower())
    terms = []
    stopwords = {"什么", "如何", "以及", "一个", "关于", "这个", "那个", "进行", "基于", "附带", "来源", "清单"}
    for part in parts:
        if part in stopwords:
            continue
        terms.append(part)
    return terms[:12]




def _estimate_relevance(query: str, text: str) -> float:
    terms = _extract_query_terms(query)
    if not terms:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / max(len(terms), 1)



