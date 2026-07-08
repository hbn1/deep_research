"""File-backed storage for evaluation run results."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schemas import EvalCaseResult, EvalRunDetail, EvalRunSummary


class EvalResultStore:
    def __init__(self, base_dir: Path | None = None):
        env_dir = os.getenv("EVAL_RESULTS_DIR", "").strip()
        if base_dir is not None:
            self.base_dir = base_dir
        elif env_dir:
            self.base_dir = Path(env_dir)
        else:
            self.base_dir = Path(__file__).resolve().parents[2] / "eval_results"

    def save_run(self, summary: EvalRunSummary, cases: list[EvalCaseResult]) -> EvalRunSummary:
        run_dir = self.base_dir / summary.run_id
        cases_dir = run_dir / "cases"
        cases_dir.mkdir(parents=True, exist_ok=True)

        summary = summary.model_copy(update={"result_dir": str(run_dir)})
        self._write_json(run_dir / "summary.json", summary.model_dump(mode="json"))
        self._write_markdown(run_dir / "summary.md", summary, cases)

        with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
            for result in cases:
                payload = result.model_dump(mode="json")
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._write_json(cases_dir / f"{result.case.id}.json", payload)

        return summary

    def list_runs(self) -> list[EvalRunSummary]:
        if not self.base_dir.exists():
            return []
        runs: list[EvalRunSummary] = []
        for path in sorted(self.base_dir.glob("*/summary.json"), reverse=True):
            try:
                runs.append(EvalRunSummary.model_validate(self._read_json(path)))
            except Exception:
                continue
        return runs

    def get_run(self, run_id: str) -> EvalRunDetail:
        run_dir = self._run_dir(run_id)
        summary = EvalRunSummary.model_validate(self._read_json(run_dir / "summary.json"))
        cases: list[EvalCaseResult] = []
        jsonl_path = run_dir / "cases.jsonl"
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    raw = raw.strip()
                    if raw:
                        cases.append(EvalCaseResult.model_validate(json.loads(raw)))
        return EvalRunDetail(summary=summary, cases=cases)

    def get_case(self, run_id: str, case_id: str) -> EvalCaseResult:
        case_path = self._run_dir(run_id) / "cases" / f"{Path(case_id).name}.json"
        if not case_path.exists():
            raise FileNotFoundError(f"Evaluation case result not found: {case_id}")
        return EvalCaseResult.model_validate(self._read_json(case_path))

    def _run_dir(self, run_id: str) -> Path:
        safe_id = Path(run_id).name
        if safe_id != run_id or not safe_id:
            raise FileNotFoundError("Invalid run_id")
        run_dir = self.base_dir / safe_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Evaluation run not found: {run_id}")
        return run_dir

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_markdown(path: Path, summary: EvalRunSummary, cases: list[EvalCaseResult]) -> None:
        lines = [
            f"# Eval Run {summary.run_id}",
            "",
            f"- Dataset: `{summary.dataset_id}`",
            f"- Status: `{summary.status}`",
            f"- Total cases: {summary.total_cases}",
            f"- Passed: {summary.passed_cases}",
            f"- Failed: {summary.failed_cases}",
            f"- Pass rate: {summary.pass_rate:.1f}%",
            f"- Average score: {summary.average_score:.1f}",
            f"- Average latency: {summary.average_latency_ms:.0f}ms",
            "",
            "| Case | Status | Score | Latency | Failed evaluators | Suspected stages |",
            "|---|---:|---:|---:|---|---|",
        ]
        for result in cases:
            failed = ", ".join(result.failed_evaluators) or "-"
            stages = ", ".join(result.suspected_stages) or "-"
            lines.append(
                f"| `{result.case.id}` | {result.status} | {result.score} | "
                f"{result.latency_ms}ms | {failed} | {stages} |"
            )

        failed_cases = [case for case in cases if not case.passed]
        if failed_cases:
            lines.extend(["", "## Failed Case Details", ""])
            for result in failed_cases:
                lines.append(f"### {result.case.id}")
                lines.append("")
                lines.append(f"Query: {result.case.query}")
                lines.append("")
                for issue in result.issues:
                    lines.append(f"- `{issue.stage}` / `{issue.evaluator}`: {issue.message}")
                lines.append("")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
