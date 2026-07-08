from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.dependencies import require_admin_access
from backend.schemas.rag import (
    RagDocumentsResponse,
    RagSearchRequest,
    RagSearchResponse,
    RagStatusResponse,
    RagUploadResponse,
)
from backend.service import RagDocumentService, get_rag_document_service
from backend.service.rag_document_service import RagDocumentError


router = APIRouter(prefix="/api/v1/rag", tags=["rag"], dependencies=[Depends(require_admin_access)])


@router.post("/documents", response_model=RagUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Form("default_tenant"),
    user_id: str = Form("default_user"),
    thread_id: str = Form(""),
    service: RagDocumentService = Depends(get_rag_document_service),
) -> RagUploadResponse:
    content = await file.read()
    try:
        record = await run_in_threadpool(
            service.ingest_document,
            filename=file.filename or "document",
            content_type=file.content_type or "",
            content=content,
            tenant_id=tenant_id,
            user_id=user_id,
            thread_id=thread_id,
        )
    except RagDocumentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return RagUploadResponse(document=record)


@router.get("/documents", response_model=RagDocumentsResponse)
def list_documents(
    tenant_id: str = Query("default_tenant"),
    limit: int = Query(50, ge=1, le=200),
    service: RagDocumentService = Depends(get_rag_document_service),
) -> RagDocumentsResponse:
    return RagDocumentsResponse(documents=service.list_documents(tenant_id=tenant_id, limit=limit))


@router.get("/status", response_model=RagStatusResponse)
def rag_status(
    tenant_id: str = Query("default_tenant"),
    service: RagDocumentService = Depends(get_rag_document_service),
) -> RagStatusResponse:
    try:
        return RagStatusResponse.model_validate(service.status(tenant_id=tenant_id))
    except RagDocumentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/search", response_model=RagSearchResponse)
async def search_rag(
    payload: RagSearchRequest,
    service: RagDocumentService = Depends(get_rag_document_service),
) -> RagSearchResponse:
    try:
        results = await run_in_threadpool(
            service.search,
            query=payload.query,
            tenant_id=payload.tenant_id,
            limit=payload.limit,
        )
    except RagDocumentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG search failed: {exc}") from exc
    return RagSearchResponse(query=payload.query, tenant_id=payload.tenant_id, results=results)
