import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.dependencies import require_admin_access
from backend.service import WorkflowService, get_workflow_service
from evaluation.dataset_loader import EvalDatasetLoader
from evaluation.runner import EvaluationRunner
from evaluation.schemas import EvalRunRequest
from evaluation.store import EvalResultStore


router = APIRouter(prefix="/api/v1/evals", tags=["evals"], dependencies=[Depends(require_admin_access)])

_dataset_loader = EvalDatasetLoader()
_result_store = EvalResultStore()


def _runner(workflow_service: WorkflowService) -> EvaluationRunner:
    return EvaluationRunner(
        workflow_service=workflow_service,
        dataset_loader=_dataset_loader,
        store=_result_store,
    )


@router.get("/datasets")
def list_eval_datasets():
    return {"datasets": [item.model_dump(mode="json") for item in _dataset_loader.list_datasets()]}


@router.post("/run")
async def run_eval(
    payload: EvalRunRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
):
    try:
        summary = await _runner(workflow_service).run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return summary.model_dump(mode="json")


@router.post("/run/stream")
async def stream_eval(
    payload: EvalRunRequest,
    workflow_service: WorkflowService = Depends(get_workflow_service),
) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in _runner(workflow_service).iter_run(payload):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/runs")
def list_eval_runs():
    return {"runs": [item.model_dump(mode="json") for item in _result_store.list_runs()]}


@router.get("/runs/{run_id}")
def get_eval_run(run_id: str):
    try:
        detail = _result_store.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return detail.model_dump(mode="json")


@router.get("/runs/{run_id}/cases/{case_id}")
def get_eval_case(run_id: str, case_id: str):
    try:
        result = _result_store.get_case(run_id, case_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result.model_dump(mode="json")
