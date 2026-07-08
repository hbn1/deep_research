"""Run evaluation datasets against the live DeepResearch workflow."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from .dataset_loader import EvalDatasetLoader
from .evaluators import evaluate_case
from .schemas import EvalCaseResult, EvalRunRequest, EvalRunSummary
from .store import EvalResultStore
from observability.langsmith import trace_run


class EvaluationRunner:
    def __init__(
        self,
        workflow_service,
        dataset_loader: EvalDatasetLoader | None = None,
        store: EvalResultStore | None = None,
    ):
        self.workflow_service = workflow_service
        self.dataset_loader = dataset_loader or EvalDatasetLoader()
        self.store = store or EvalResultStore()

    async def run(self, request: EvalRunRequest) -> EvalRunSummary:
        summary: EvalRunSummary | None = None
        async for event in self.iter_run(request):
            if event.get("type") == "summary":
                summary = EvalRunSummary.model_validate(event["summary"])
        if summary is None:
            raise RuntimeError("Evaluation finished without a summary")
        return summary

    async def iter_run(self, request: EvalRunRequest) -> AsyncIterator[dict[str, Any]]:
        cases = self.dataset_loader.load_cases(
            dataset_id=request.dataset_id,
            case_ids=request.case_ids,
            max_cases=request.max_cases,
        )
        if not cases:
            raise ValueError(f"No evaluation cases selected for dataset: {request.dataset_id}")

        run_id = self._new_run_id(request.dataset_id)
        started_at = now_iso()
        with trace_run(
            "evaluation.run",
            inputs={
                "dataset_id": request.dataset_id,
                "case_ids": request.case_ids,
                "max_cases": request.max_cases,
            },
            metadata={
                "run_id": run_id,
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
                "total_cases": len(cases),
            },
            tags=("evaluation",),
        ) as run_span:
            yield {
                "run_id": run_id,
                "type": "run_start",
                "dataset_id": request.dataset_id,
                "total_cases": len(cases),
            }

            results: list[EvalCaseResult] = []
            for index, case in enumerate(cases, start=1):
                yield {
                    "type": "case_start",
                    "run_id": run_id,
                    "case_id": case.id,
                    "index": index,
                    "total": len(cases),
                    "query": case.query,
                }

                with trace_run(
                    "evaluation.case",
                    inputs={"query": case.query},
                    metadata={
                        "run_id": run_id,
                        "case_id": case.id,
                        "dataset_id": request.dataset_id,
                        "category": case.category,
                        "expected_route": case.expected_route,
                    },
                    tags=("evaluation", "case"),
                ) as case_span:
                    start = time.perf_counter()
                    state: dict[str, Any] = {}
                    error: str | None = None
                    try:
                        state = await self.workflow_service.run_state(
                            query=case.query,
                            user_id=request.user_id,
                            thread_id=self._thread_id(run_id, case.id),
                            tenant_id=request.tenant_id,
                            max_iterations=case.max_iterations or request.max_iterations,
                            enable_memory=request.enable_memory,
                            persist=False,
                        )
                    except Exception as exc:
                        error = str(exc)

                    latency_ms = round((time.perf_counter() - start) * 1000)
                    result = evaluate_case(
                        case=case,
                        state=state,
                        latency_ms=latency_ms,
                        error=error,
                        default_threshold=request.score_threshold,
                    )
                    case_span.end(
                        outputs={
                            "status": result.status,
                            "passed": result.passed,
                            "score": result.score,
                            "latency_ms": result.latency_ms,
                            "failed_evaluators": result.failed_evaluators,
                            "suspected_stages": result.suspected_stages,
                        }
                    )
                results.append(result)
                yield {
                    "type": "case_result",
                    "run_id": run_id,
                    "case_id": case.id,
                    "result": result.model_dump(mode="json"),
                }

            summary = self._build_summary(
                run_id=run_id,
                dataset_id=request.dataset_id,
                started_at=started_at,
                results=results,
            )
            summary = self.store.save_run(summary, results)
            run_span.end(
                outputs={
                    "status": summary.status,
                    "total_cases": summary.total_cases,
                    "passed_cases": summary.passed_cases,
                    "failed_cases": summary.failed_cases,
                    "pass_rate": summary.pass_rate,
                    "average_score": summary.average_score,
                }
            )
            yield {
                "type": "summary",
                "run_id": run_id,
                "summary": summary.model_dump(mode="json"),
            }

    @staticmethod
    def _build_summary(
        run_id: str,
        dataset_id: str,
        started_at: str,
        results: list[EvalCaseResult],
    ) -> EvalRunSummary:
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        failed = total - passed
        avg_score = sum(result.score for result in results) / total if total else 0
        avg_latency = sum(result.latency_ms for result in results) / total if total else 0
        return EvalRunSummary(
            run_id=run_id,
            dataset_id=dataset_id,
            status="completed",
            started_at=started_at,
            completed_at=now_iso(),
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            pass_rate=(passed / total * 100) if total else 0,
            average_score=avg_score,
            average_latency_ms=avg_latency,
            result_dir="",
            cases=[
                {
                    "case_id": result.case.id,
                    "query": result.case.query,
                    "status": result.status,
                    "score": result.score,
                    "latency_ms": result.latency_ms,
                    "failed_evaluators": result.failed_evaluators,
                    "suspected_stages": result.suspected_stages,
                }
                for result in results
            ],
        )

    @staticmethod
    def _new_run_id(dataset_id: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        safe_dataset = "".join(ch for ch in dataset_id if ch.isalnum() or ch in {"-", "_"}) or "eval"
        return f"{stamp}-{safe_dataset}-{suffix}"

    @staticmethod
    def _thread_id(run_id: str, case_id: str) -> str:
        safe_case = "".join(ch for ch in case_id if ch.isalnum() or ch in {"-", "_"}) or "case"
        return f"eval_{run_id}_{safe_case}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
