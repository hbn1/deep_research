import json, logging, re
from langchain_core.messages import HumanMessage
from ..state import ResearchState
from ..utils import (colorize, emit, collect_tool_calls, with_memory_context, log_inputs, invoke_json_agent, _last_content, _load_json)
from ._common import _enrich_evidence_from_raw, _prune_evidence_to_allowed_sources

logger = logging.getLogger('mult_agents')

def _is_official_domain(domain: str) -> bool:
    """Check if domain belongs to government, education, or recognized authority.

    Uses suffix matching only (no substring) to avoid false positives
    like 'government-blog.com' or 'gov-news.net'.
    """
    value = domain.lower().strip()
    if not value:
        return False
    if value.endswith((".gov.cn", ".gov", ".edu", ".edu.cn", ".mil", ".ac.cn")):
        return True
    if value.endswith((".int", ".go.jp", ".gov.uk", ".gov.au", ".gov.sg", ".europa.eu")):
        return True
    return False




def _score_evidence(record: dict) -> tuple[float, str]:
    """Score evidence reliability based on source type, domain authority, and content signals."""
    source_type = record.get("source_type")
    if source_type == "local":
        return 0.92, "???????????????"

    domain = str(record.get("domain", "")).lower()

    if _is_official_domain(domain):
        base_score = 0.88
        reason = "?????????"
    elif any(domain.endswith(suffix) for suffix in [
        "reuters.com", "bloomberg.com", "people.com.cn", "xinhuanet.com",
        "bbc.com", "bbc.co.uk", "economist.com", "nature.com", "science.org",
        "who.int", "worldbank.org", "imf.org", "un.org", "wto.org",
    ]):
        base_score = 0.82
        reason = "??????/????"
    elif any(word in domain for word in ["news", "finance"]):
        base_score = 0.70
        reason = "??????"
    elif domain:
        base_score = 0.55
        reason = "???????"
    else:
        base_score = 0.40
        reason = "????????"

    snippet = str(record.get("snippet", "")).strip()
    title = str(record.get("title", "")).strip()
    score = base_score

    if len(snippet) > 200:
        score = min(score + 0.04, 0.95)
    elif len(snippet) < 30:
        score = max(score - 0.05, 0.30)

    if len(title) < 5:
        score = max(score - 0.03, 0.30)

    if not record.get("url"):
        score = max(score - 0.05, 0.30)

    return round(score, 4), reason




def _normalize_text_for_dedup(text: str) -> str:
    """Normalize text for near-duplicate detection."""
    import unicodedata
    text = unicodedata.normalize("NFKC", str(text))
    cleaned = "".join(
        ch.lower() if ch.isascii() and ch.isalnum() else (ch if "一" <= ch <= "鿿" else " ")
        for ch in text
    )
    return " ".join(cleaned.split())




def _estimate_similarity(a: str, b: str) -> float:
    """Simple bigram overlap similarity."""
    if not a or not b:
        return 0.0
    a_bigrams = {a[i : i + 2] for i in range(len(a) - 1)}
    b_bigrams = {b[i : i + 2] for i in range(len(b) - 1)}
    if not a_bigrams or not b_bigrams:
        return 0.0
    return len(a_bigrams & b_bigrams) / len(a_bigrams | b_bigrams)




def _dedupe_by_content(items: list[dict], threshold: float = 0.55) -> list[dict]:
    """Remove near-duplicate evidence items by content similarity."""
    if len(items) <= 1:
        return items[:]
    keep: list[dict] = []
    for item in items:
        text_a = _normalize_text_for_dedup(
            str(item.get("snippet", "")) + " " + str(item.get("title", ""))
        )
        is_dup = False
        for kept in keep:
            text_b = _normalize_text_for_dedup(
                str(kept.get("snippet", "")) + " " + str(kept.get("title", ""))
            )
            if _estimate_similarity(text_a, text_b) >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(item)
    return keep




def _detect_cross_verification(evidence_pool: list[dict]) -> list[dict]:
    """Boost reliability scores when multiple independent sources support the same claim."""
    if len(evidence_pool) < 2:
        return evidence_pool[:]
    result = [dict(item) for item in evidence_pool]
    for i, item in enumerate(result):
        item_questions = set(item.get("supports_questions", []) or [])
        if not item_questions:
            continue
        corroboration_count = 0
        for j, other in enumerate(result):
            if i == j:
                continue
            other_questions = set(other.get("supports_questions", []) or [])
            if item_questions & other_questions:
                corroboration_count += 1
        if corroboration_count >= 2:
            boost = min(0.06, 0.02 * corroboration_count)
            item["reliability_score"] = round(
                min(item.get("reliability_score", 0.5) + boost, 0.97), 4
            )
            item.setdefault("cross_verified", True)
            item.setdefault("cross_verification_count", corroboration_count)
    return result




def _dedupe_sources(items: list[dict], key_fields: list[str]) -> list[dict]:
    seen = set()
    results = []
    for item in items:
        key = tuple(str(item.get(field, "")).strip() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results




def _fallback_audit(state: ResearchState) -> dict:
    """Robust fallback audit: score, cross-verify, dedup, and detect conflicts."""
    evidence_pool: list[dict] = []
    source_index: list[dict] = []
    audit_flags: list[dict] = []

    all_records = state.get("web_evidence", []) + state.get("local_evidence", [])

    for record in all_records:
        score, reason = _score_evidence(record)
        normalized = dict(record)
        normalized["reliability_score"] = score
        normalized["reliability_reason"] = reason
        normalized["source_label"] = (
            record.get("title") or record.get("doc_id")
            or record.get("url") or record.get("source_id")
        )
        normalized.setdefault("supports", [])
        normalized.setdefault("refutes", [])
        normalized.setdefault("supports_questions", [])
        evidence_pool.append(normalized)

    evidence_pool.sort(key=lambda x: x.get("reliability_score", 0), reverse=True)
    evidence_pool = _dedupe_by_content(evidence_pool, threshold=0.55)
    evidence_pool = _detect_cross_verification(evidence_pool)

    for item in evidence_pool:
        sid = str(item.get("source_id", ""))
        score = item.get("reliability_score", 0)
        locator = item.get("url") or item.get("doc_id") or ""
        if score < 0.45:
            audit_flags.append({
                "type": "low_confidence",
                "target": sid,
                "reason": f"??????? ({score:.2f}): {item.get('reliability_reason', '')}",
            })
        else:
            source_index.append({
                "source_id": sid,
                "label": item["source_label"],
                "locator": locator or "???????",
                "source_type": item.get("source_type", "source"),
            })

    question_groups: dict[str, list[dict]] = {}
    for item in evidence_pool:
        for q in (item.get("supports_questions") or []):
            q = str(q)
            question_groups.setdefault(q, []).append(item)

    for question, items in question_groups.items():
        if len(items) < 2:
            continue
        scores = [it.get("reliability_score", 0) for it in items]
        if max(scores) - min(scores) > 0.35:
            audit_flags.append({
                "type": "divergent_quality",
                "target": question,
                "reason": f"????????????? (range={max(scores)-min(scores):.2f})???????",
            })

    for hypo in state.get("hypotheses", []):
        hypo_id = hypo.get("id")
        related = [
            item for item in evidence_pool
            if hypo_id in item.get("supports", [])
            or hypo_id in item.get("refutes", [])
        ]
        if not related:
            audit_flags.append({
                "type": "missing_evidence",
                "target": hypo_id,
                "reason": "????????",
            })

    return {
        "summary": "??????????????????",
        "evidence_pool": evidence_pool,
        "audit_flags": audit_flags,
        "source_index": _dedupe_sources(source_index, ["source_id"]),
    }




def deep_dive_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[deep_dive]", "cyan"), colorize(agent_name, "magenta"))
    if not state.get("web_evidence") and not state.get("local_evidence"):
        logger.info("%s 等待检索结果", colorize("[deep_dive]", "yellow"))
        return {}
    # Fast path: when total evidence <= 6 items, skip LLM and use programmatic audit
    total_evidence = len(state.get("web_evidence", [])) + len(state.get("local_evidence", []))
    if total_evidence <= 6:
        logger.info("%s ?????????(%d?)??? LLM ??", colorize("[deep_dive]", "green"), total_evidence)
        fb = _fallback_audit(state)
        return {
            "deep_dive": fb["summary"],
            "audit": fb["summary"],
            "evidence_pool": fb["evidence_pool"],
            "audit_flags": fb["audit_flags"],
            "source_index": fb["source_index"],
            "messages": [],
        }
    fallback = _fallback_audit(state)
    payload, content, messages = invoke_json_agent(
        state,
        "请对 web 与 local 证据进行评分、去重、冲突审计，并只输出 JSON。\n"
        f"问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"web_evidence：{json.dumps(state.get('web_evidence', []), ensure_ascii=False)}\n"
        f"local_evidence：{json.dumps(state.get('local_evidence', []), ensure_ascii=False)}",
        agent,
        agent_name,
        "deep_dive",
        fallback,
    )
    payload_pool = payload.get("evidence_pool") if isinstance(payload.get("evidence_pool"), list) else []
    raw_evidence = state.get("web_evidence", []) + state.get("local_evidence", [])
    allowed_source_ids = {str(item.get("source_id", "")).strip() for item in raw_evidence if item.get("source_id")}
    evidence_pool = []
    for item in payload_pool:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("source_id", "")).strip()
        if sid and sid in allowed_source_ids:
            evidence_pool.append(item)
    if not evidence_pool:
        evidence_pool = fallback["evidence_pool"]
    # Post-process: content dedup + cross-verification
    evidence_pool.sort(key=lambda x: x.get("reliability_score", 0), reverse=True)
    evidence_pool = _dedupe_by_content(evidence_pool, threshold=0.55)
    evidence_pool = _detect_cross_verification(evidence_pool)
    existing_ids = {str(item.get("source_id", "")).strip() for item in evidence_pool if isinstance(item, dict)}
    for record in raw_evidence:
        sid = str(record.get("source_id", "")).strip()
        if not sid or sid in existing_ids:
            continue
        score, reason = _score_evidence(record)
        evidence_pool.append(
            {
                "source_id": sid,
                "source_type": record.get("source_type", "source"),
                "title": record.get("title") or sid,
                "url": record.get("url", ""),
                "doc_id": record.get("doc_id", ""),
                "snippet": record.get("snippet", ""),
                "supports_questions": record.get("supports_questions", []),
                "reliability_score": score,
                "reliability_reason": reason,
                "source_label": record.get("title") or record.get("doc_id") or record.get("url") or sid,
            }
        )
        existing_ids.add(sid)
    audit_flags = payload.get("audit_flags") if isinstance(payload.get("audit_flags"), list) else fallback["audit_flags"]
    source_index = []
    for item in evidence_pool:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("source_id", "")).strip()
        if not sid:
            continue
        source_index.append(
            {
                "source_id": sid,
                "label": item.get("title") or item.get("source_label") or sid,
                "locator": item.get("url") or item.get("doc_id") or "",
                "source_type": item.get("source_type", "source"),
            }
        )
    source_index = _dedupe_sources(source_index, ["source_id"])
    return {
        "deep_dive": payload.get("summary", content),
        "audit": payload.get("summary", content),
        "evidence_pool": evidence_pool,
        "audit_flags": audit_flags,
        "source_index": source_index,
        "messages": messages,
    }



