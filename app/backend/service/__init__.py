from functools import lru_cache

from backend.config import AppSettings
from .rag_document_service import RagDocumentService
from .workflow_service import WorkflowService


@lru_cache(maxsize=1)
def get_workflow_service() -> WorkflowService:
    settings = AppSettings()
    return WorkflowService(config_path=settings.config_path)


@lru_cache(maxsize=1)
def get_rag_document_service() -> RagDocumentService:
    settings = AppSettings()
    return RagDocumentService(
        config_path=settings.config_path,
        upload_dir=settings.rag_upload_path(),
        max_upload_bytes=settings.rag_max_upload_mb * 1024 * 1024,
        allowed_extensions=settings.rag_allowed_extension_set(),
        max_tenant_storage_bytes=settings.rag_max_tenant_storage_mb * 1024 * 1024,
        validate_file_signatures=settings.rag_validate_file_signatures,
    )


__all__ = [
    "WorkflowService",
    "get_workflow_service",
    "RagDocumentService",
    "get_rag_document_service",
]
