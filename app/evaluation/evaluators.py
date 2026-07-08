"""Deterministic evaluators for agent regression runs."""

from __future__ import annotations

import re
from typing import Any

from .schemas import EvalCase, EvalCaseResult, EvalIssue, EvalMetricResult, EvalTrace

CURRENT_YEAR = "2026"
CITATION_RE = re.compile(r"\[((?:[A-Z]{2,}|WEB|LOC|SRC|DOC)\d+_\d+-\d+)\]")


def evaluate_case(
    case: EvalCase,
    state: dict[str, Any],
    latency_ms: int,
    error: str | None,
    default_threshold: int,
) -> EvalCaseResult:
    trace = build_trace(state)
    metrics: list[EvalMetricResult] = []
    issues: list[EvalIssue] = []

    add_metric(metrics, issues, final_answer_metric(trace, error))
    if case.expected_route:
        add_metric(metrics, issues, route_metric(case, trace))
    if case.required_keywords:
        add_metric(metrics, issues, required_keywords_metric(case, trace))
    if case.forbidden_claims:
        add_metric(metrics, issues, forbidden_claims_metric(case, trace))
    if case.freshness_required:
        add_metric(metrics, issues, freshness_metric(trace))
    if case.min_evidence_count is not None:
        add_metric(metrics, issues, evidence_count_metric(case, trace))
    if case.min_citations is not None or trace.citations.get("used"):
        add_metric(metrics, issues, citation_metric(case, trace))
    if case.max_latency_ms:
        add_metric(metrics, issues, latency_metric(case, latency_ms))

    if error:
        issues.append(
            EvalIssue(
                stage="workflow",
                evaluator="WorkflowExecution",
                severity="error",
                message=error,
            )
        )
    for flag in as_list(state.get("audit_flags")):
        if isinstance(flag, dict) and flag.get("type") == "model_error":
            issues.append(
                EvalIssue(
                    stage=str(flag.get("stage") or "workflow"),
                    evaluator="ModelProvider",
                    severity="error",
                    message=str(flag.get("message") or "Model provider call failed"),
                )
            )

    threshold = case.score_threshold if case.score_threshold is not None else default_threshold
    weighted_total = sum(metric.score * metric.weight for metric in metrics if metric.weight > 0)
    weight_sum = sum(metric.weight for metric in metrics if metric.weight > 0) or 1
    score = round(weighted_total / weight_sum)

    hard_fail = any(issue.severity == "error" for issue in issues)
    passed = score >= threshold and not hard_fail
    failed_evaluators = [metric.name for metric in metrics if not metric.passed]
    suspected = sorted({issue.stage for issue in issues if issue.stage})

    return EvalCaseResult(
        case=case,
        status="passed" if passed else "failed",
        passed=passed,
        score=score,
        threshold=threshold,
        latency_ms=latency_ms,
        suspected_stages=suspected,
        failed_evaluators=failed_evaluators,
        metrics=metrics,
        issues=issues,
        trace=trace,
        error=error,
    )


def add_metric(
    metrics: list[EvalMetricResult],
    issues: list[EvalIssue],
    metric_and_issue: tuple[EvalMetricResult, EvalIssue | None],
) -> None:
    metric, issue = metric_and_issue
    metrics.append(metric)
    if issue:
        issues.append(issue)


def final_answer_metric(trace: EvalTrace, error: str | None) -> tuple[EvalMetricResult, EvalIssue | None]:
    if error:
        return metric("FinalAnswer", "workflow", 0, False, 3.0, "Workflow raised an exception"), None
    final = trace.final.strip()
    passed = bool(final)
    score = 100 if passed else 0
    issue = None
    if not passed:
        issue = EvalIssue(
            stage="write",
            evaluator="FinalAnswer",
            message="No final answer was produced.",
        )
    return metric("FinalAnswer", "write", score, passed, 3.0, "Final answer exists" if passed else "Missing final answer"), issue


def route_metric(case: EvalCase, trace: EvalTrace) -> tuple[EvalMetricResult, EvalIssue | None]:
    expected = (case.expected_route or "").strip().lower()
    actual = (trace.route or trace.intent or "").strip().lower()
    passed = actual == expected
    reason = f"Expected route={expected}, actual route={actual or 'unknown'}"
    issue = None
    if not passed:
        issue = EvalIssue(
            stage="intent",
            evaluator="RouteEvaluator",
            message=reason,
        )
    return metric("RouteEvaluator", "intent", 100 if passed else 0, passed, 1.5, reason), issue


def required_keywords_metric(case: EvalCase, trace: EvalTrace) -> tuple[EvalMetricResult, EvalIssue | None]:
    final_lower = trace.final.lower()
    missing = [word for word in case.required_keywords if word.lower() not in final_lower]
    passed = not missing
    score = round(100 * (len(case.required_keywords) - len(missing)) / max(len(case.required_keywords), 1))
    reason = "All required keywords are present" if passed else f"Missing keywords: {', '.join(missing)}"
    issue = None
    if not passed:
        issue = EvalIssue(
            stage="write",
            evaluator="RequiredKeywords",
            message=reason,
        )
    return metric("RequiredKeywords", "write", score, passed, 1.0, reason), issue


def forbidden_claims_metric(case: EvalCase, trace: EvalTrace) -> tuple[EvalMetricResult, EvalIssue | None]:
    final_lower = trace.final.lower()
    hits = [claim for claim in case.forbidden_claims if claim.lower() in final_lower]
    passed = not hits
    reason = "No forbidden claims found" if passed else f"Forbidden claims found: {', '.join(hits)}"
    issue = None
    if hits:
        issue = EvalIssue(
            stage="write",
            evaluator="ForbiddenClaims",
            message=reason,
            evidence=hits[0],
        )
    return metric("ForbiddenClaims", "write", 100 if passed else 0, passed, 2.0, reason), issue


def freshness_metric(trace: EvalTrace) -> tuple[EvalMetricResult, EvalIssue | None]:
    haystack = " ".join([trace.final, trace.plan, trace.analysis])
    passed = CURRENT_YEAR in haystack
    reason = f"Freshness check expects current year marker {CURRENT_YEAR}"
    issue = None
    if not passed:
        issue = EvalIssue(
            stage="plan",
            evaluator="Freshness",
            message=reason,
        )
    return metric("Freshness", "plan", 100 if passed else 40, passed, 1.0, reason), issue


def evidence_count_metric(case: EvalCase, trace: EvalTrace) -> tuple[EvalMetricResult, EvalIssue | None]:
    required = case.min_evidence_count or 0
    count = trace.counts.get("evidence_pool", 0)
    passed = count >= required
    score = 100 if passed else round(100 * count / max(required, 1))
    reason = f"Evidence count {count}/{required}"
    issue = None
    if not passed:
        issue = EvalIssue(
            stage="deep_dive",
            evaluator="EvidenceCount",
            message=reason,
        )
    return metric("EvidenceCount", "deep_dive", score, passed, 1.0, reason), issue


def citation_metric(case: EvalCase, trace: EvalTrace) -> tuple[EvalMetricResult, EvalIssue | None]:
    used = trace.citations.get("used", [])
    missing = trace.citations.get("missing", [])
    required = case.min_citations or 0
    enough = len(used) >= required
    valid = not missing
    passed = enough and valid
    score = 100
    if not enough:
        score = min(score, round(100 * len(used) / max(required, 1)))
    if missing:
        score = min(score, 30)
    reason_parts = [f"citations={len(used)}/{required}"]
    if missing:
        reason_parts.append(f"missing ids: {', '.join(missing)}")
    reason = "; ".join(reason_parts)
    issue = None
    if not passed:
        issue = EvalIssue(
            stage="write",
            evaluator="CitationValidity",
            message=reason,
            evidence=", ".join(missing) if missing else None,
        )
    return metric("CitationValidity", "write", score, passed, 2.0, reason), issue


def latency_metric(case: EvalCase, latency_ms: int) -> tuple[EvalMetricResult, EvalIssue | None]:
    limit = case.max_latency_ms or 1
    passed = latency_ms <= limit
    score = 100 if passed else max(0, round(100 * limit / latency_ms))
    reason = f"Latency {latency_ms}ms / limit {limit}ms"
    issue = None
    if not passed:
        issue = EvalIssue(
            stage="workflow",
            evaluator="Latency",
            severity="warning",
            message=reason,
        )
    return metric("Latency", "workflow", score, passed, 0.5, reason), issue


def metric(name: str, stage: str, score: int, passed: bool, weight: float, reason: str) -> EvalMetricResult:
    return EvalMetricResult(
        name=name,
        stage=stage,
        score=max(0, min(100, score)),
        passed=passed,
        weight=weight,
        reason=reason,
    )


def build_trace(state: dict[str, Any]) -> EvalTrace:
    source_index = compact_records(state.get("source_index", []), limit=20)
    evidence_pool = compact_records(state.get("evidence_pool", []), limit=20)
    web_evidence = compact_records(state.get("web_evidence", []), limit=12)
    local_evidence = compact_records(state.get("local_evidence", []), limit=12)
    final = str(state.get("final") or state.get("draft") or "")
    intent = str(state.get("intent") or "")
    route = intent if intent in {"direct", "multiagent"} else "unknown"

    valid_source_ids = collect_source_ids(source_index, evidence_pool, web_evidence, local_evidence)
    used_citations = sorted(set(CITATION_RE.findall(final)))
    missing_citations = [source_id for source_id in used_citations if source_id not in valid_source_ids]

    return EvalTrace(
        route=route,
        intent=intent,
        plan=clip_text(state.get("plan", "")),
        analysis=clip_text(state.get("analysis", "")),
        final=clip_text(final, max_chars=12000),
        search_plan=compact_records(state.get("search_plan", []), limit=8),
        findings=compact_records(state.get("findings", []), limit=12),
        source_index=source_index,
        web_evidence=web_evidence,
        local_evidence=local_evidence,
        evidence_pool=evidence_pool,
        counts={
            "web_evidence": len(as_list(state.get("web_evidence"))),
            "local_evidence": len(as_list(state.get("local_evidence"))),
            "evidence_pool": len(as_list(state.get("evidence_pool"))),
            "source_index": len(as_list(state.get("source_index"))),
            "findings": len(as_list(state.get("findings"))),
        },
        citations={
            "used": used_citations,
            "missing": missing_citations,
            "available": sorted(valid_source_ids),
        },
        needs_more_research=bool(state.get("needs_more_research", False)),
        missing_gaps=[str(item) for item in as_list(state.get("missing_gaps"))[:8]],
        iteration=int(state.get("iteration", 0) or 0),
    )


def collect_source_ids(*groups: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for group in groups:
        for item in group:
            for key in ("source_id", "id", "citation_id"):
                value = item.get(key)
                if value:
                    ids.add(str(value))
    return ids


def compact_records(value: Any, limit: int) -> list[dict[str, Any]]:
    records = as_list(value)[:limit]
    compacted: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            compacted.append({"value": clip_text(record)})
            continue
        compacted.append(
            {
                key: clip_text(val)
                for key, val in record.items()
                if key
                in {
                    "source_id",
                    "id",
                    "title",
                    "url",
                    "snippet",
                    "content",
                    "locator",
                    "source",
                    "claim",
                    "finding",
                    "question",
                    "query",
                    "score",
                    "confidence",
                }
            }
        )
    return compacted


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def clip_text(value: Any, max_chars: int = 900) -> str:
    text = value if isinstance(value, str) else str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
