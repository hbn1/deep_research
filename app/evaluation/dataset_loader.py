"""Load JSONL evaluation datasets from the workspace."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schemas import EvalCase, EvalDatasetInfo


class EvalDatasetLoader:
    def __init__(self, base_dir: Path | None = None):
        env_dir = os.getenv("EVAL_DATASETS_DIR", "").strip()
        if base_dir is not None:
            self.base_dir = base_dir
        elif env_dir:
            self.base_dir = Path(env_dir)
        else:
            self.base_dir = Path(__file__).resolve().parents[2] / "eval_datasets"

    def list_datasets(self) -> list[EvalDatasetInfo]:
        if not self.base_dir.exists():
            return []

        datasets: list[EvalDatasetInfo] = []
        for path in sorted(self.base_dir.glob("*.jsonl")):
            datasets.append(
                EvalDatasetInfo(
                    id=path.stem,
                    path=str(path),
                    case_count=self._count_jsonl(path),
                )
            )
        return datasets

    def load_cases(
        self,
        dataset_id: str,
        case_ids: list[str] | None = None,
        max_cases: int | None = None,
    ) -> list[EvalCase]:
        path = self._dataset_path(dataset_id)
        if not path.exists():
            raise ValueError(f"Evaluation dataset not found: {dataset_id}")

        selected = set(case_ids or [])
        cases: list[EvalCase] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                try:
                    case = EvalCase.model_validate(json.loads(raw))
                except Exception as exc:
                    raise ValueError(f"Invalid case in {path.name}:{line_no}: {exc}") from exc
                if selected and case.id not in selected:
                    continue
                cases.append(case)
                if max_cases and len(cases) >= max_cases:
                    break

        if selected:
            missing = sorted(selected - {case.id for case in cases})
            if missing:
                raise ValueError(f"Cases not found in {dataset_id}: {', '.join(missing)}")
        return cases

    def _dataset_path(self, dataset_id: str) -> Path:
        safe_id = Path(dataset_id).name
        if safe_id != dataset_id or not safe_id:
            raise ValueError("dataset_id must be a simple dataset name")
        return self.base_dir / f"{safe_id}.jsonl"

    @staticmethod
    def _count_jsonl(path: Path) -> int:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                stripped = raw.strip()
                if stripped and not stripped.startswith("#"):
                    count += 1
        return count
