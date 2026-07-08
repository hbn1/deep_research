"""Schemas shared by the evaluation runner, storage layer, and API."""

from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    category: str = "general"
    expected_route: str | None = None
    required_keywords: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    freshness_required: bool = False
    min_citations: int | None = Field(default=None, ge=0)
    min_evidence_count: int | None = Field(default=None, ge=0)
    max_latency_ms: int | None = Field(default=None, ge=1)
    max_iterations: int | None = Field(default=None, ge=1, le=6)
    score_threshold: int | None = Field(default=None, ge=0, le=100)
    notes: str | None = None


class EvalRunRequest(BaseModel):
    dataset_id: str = Field(default="smoke", min_length=1)
    case_ids: list[str] | None = None
    max_cases: int | None = Field(default=None, ge=1, le=200)
    max_iterations: int | None = Field(default=1, ge=1, le=6)
    enable_memory: bool = False
    user_id: str = Field(default="eval_user", min_length=1)
    tenant_id: str = Field(default="eval_tenant", min_length=1)
    score_threshold: int = Field(default=70, ge=0, le=100)


class EvalDatasetInfo(BaseModel):
    id: str
    path: str
    case_count: int


class EvalIssue(BaseModel):
    stage: str
    evaluator: str
    severity: str = "error"
    message: str
    evidence: str | None = None


class EvalMetricResult(BaseModel):
    name: str
    stage: str
    score: int = Field(ge=0, le=100)
    passed: bool
    weight: float = 1.0
    reason: str


class EvalTrace(BaseModel):
    route: str
    intent: str
    plan: str = ""
    analysis: str = ""
    final: str = ""
    search_plan: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    source_index: list[dict[str, Any]] = Field(default_factory=list)
    web_evidence: list[dict[str, Any]] = Field(default_factory=list)
    local_evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_pool: list[dict[str, Any]] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    citations: dict[str, list[str]] = Field(default_factory=dict)
    needs_more_research: bool = False
    missing_gaps: list[str] = Field(default_factory=list)
    iteration: int = 0


class EvalCaseResult(BaseModel):
    case: EvalCase
    status: str
    passed: bool
    score: int = Field(ge=0, le=100)
    threshold: int = Field(ge=0, le=100)
    latency_ms: int
    suspected_stages: list[str] = Field(default_factory=list)
    failed_evaluators: list[str] = Field(default_factory=list)
    metrics: list[EvalMetricResult] = Field(default_factory=list)
    issues: list[EvalIssue] = Field(default_factory=list)
    trace: EvalTrace
    error: str | None = None


class EvalRunSummary(BaseModel):
    run_id: str
    dataset_id: str
    status: str
    started_at: str
    completed_at: str | None = None
    total_cases: int
    passed_cases: int = 0
    failed_cases: int = 0
    pass_rate: float = 0
    average_score: float = 0
    average_latency_ms: float = 0
    result_dir: str
    cases: list[dict[str, Any]] = Field(default_factory=list)


class EvalRunDetail(BaseModel):
    summary: EvalRunSummary
    cases: list[EvalCaseResult]
