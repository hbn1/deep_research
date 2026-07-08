from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from uuid import uuid4

from backend.schemas.rag import RagDocumentRecord
from mult_agents.config import AppConfig
from mult_agents.main import configure_dashscope_endpoint
from mult_agents.rag.core import RAGConfig, RAGManager
from mult_agents.tools import init_rag_system
from observability.langsmith import trace_run


class RagDocumentError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class RagDocumentService:
    """Runtime document ingestion service for local RAG."""

    def __init__(
        self,
        config_path: str,
        upload_dir: str | Path,
        max_upload_bytes: int,
        allowed_extensions: set[str],
        max_tenant_storage_bytes: int = 0,
        validate_file_signatures: bool = True,
    ):
        self._config_path = config_path
        self._upload_dir = Path(upload_dir)
        self._max_upload_bytes = max_upload_bytes
        self._allowed_extensions = {item.lower() for item in allowed_extensions}
        self._max_tenant_storage_bytes = max_tenant_storage_bytes
        self._validate_file_signatures = validate_file_signatures
        self._lock = threading.RLock()
        self._manifest_path = self._upload_dir / "manifest.jsonl"

    def ingest_document(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
        tenant_id: str,
        user_id: str,
        thread_id: str = "",
    ) -> RagDocumentRecord:
        with trace_run(
            "rag.ingest_document",
            run_type="tool",
            inputs={
                "filename": filename,
                "content_type": content_type,
                "size_bytes": len(content),
            },
            metadata={"tenant_id": tenant_id, "user_id": user_id, "thread_id": thread_id},
            tags=("rag", "ingestion"),
        ) as span:
            record = self._ingest_document(
                filename=filename,
                content_type=content_type,
                content=content,
                tenant_id=tenant_id,
                user_id=user_id,
                thread_id=thread_id,
            )
            span.end(
                outputs={
                    "doc_id": record.doc_id,
                    "filename": record.filename,
                    "chunks": record.chunks,
                    "tenant_id": record.tenant_id,
                    "collection": record.collection,
                }
            )
            return record

    def _ingest_document(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
        tenant_id: str,
        user_id: str,
        thread_id: str = "",
    ) -> RagDocumentRecord:
        safe_filename = self._safe_filename(filename)
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in self._allowed_extensions:
            allowed = ", ".join(sorted(self._allowed_extensions))
            raise RagDocumentError(f"Unsupported file type {suffix or '<none>'}; allowed: {allowed}", 415)
        if not content:
            raise RagDocumentError("Uploaded file is empty.", 400)
        if len(content) > self._max_upload_bytes:
            limit_mb = self._max_upload_bytes / 1024 / 1024
            raise RagDocumentError(f"File is too large. Limit is {limit_mb:.0f} MB.", 413)
        if self._validate_file_signatures and not self._has_expected_signature(suffix, content):
            raise RagDocumentError(f"File content does not match declared type {suffix}.", 415)

        runtime_config = AppConfig.from_file(self._config_path)
        configure_dashscope_endpoint(runtime_config)

        doc_id = f"doc_{int(time.time())}_{uuid4().hex[:10]}"
        normalized_tenant = self._safe_id(tenant_id or runtime_config.tenant_id)
        normalized_user = self._safe_id(user_id or runtime_config.user_id)
        self._enforce_tenant_storage_limit(normalized_tenant, len(content))
        source = f"rag://{normalized_tenant}/{doc_id}/{safe_filename}"
        stored_path = self._upload_dir / normalized_tenant / f"{doc_id}{suffix}"
        uploaded_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        metadata = {
            "doc_id": doc_id,
            "source": source,
            "filename": safe_filename,
            "tenant_id": normalized_tenant,
            "user_id": normalized_user,
            "thread_id": thread_id or "",
            "content_type": content_type or "",
            "uploaded_at": uploaded_at,
            "file_ext": suffix,
        }

        rag_config = RAGConfig(
            milvus_host=runtime_config.milvus_host,
            milvus_port=runtime_config.milvus_port,
            collection_name=runtime_config.milvus_collection,
        )

        with self._lock:
            try:
                stored_path.parent.mkdir(parents=True, exist_ok=True)
                stored_path.write_bytes(content)
                init_rag_system(runtime_config.api_key, rag_config, tenant_id=normalized_tenant)
                rag = RAGManager.get(runtime_config.api_key, rag_config, tenant_id=normalized_tenant)
                chunks = rag.ingest_bytes(content, source, metadata=metadata)
            except RagDocumentError:
                raise
            except Exception as exc:
                stored_path.unlink(missing_ok=True)
                raise RagDocumentError(f"RAG ingestion failed: {exc}", 502) from exc

            if chunks <= 0:
                stored_path.unlink(missing_ok=True)
                raise RagDocumentError(
                    "Document text extraction produced no chunks. Check whether the file has selectable text.",
                    422,
                )

            record = RagDocumentRecord(
                doc_id=doc_id,
                filename=safe_filename,
                source=source,
                tenant_id=normalized_tenant,
                user_id=normalized_user,
                content_type=content_type or "",
                size_bytes=len(content),
                chunks=chunks,
                collection=runtime_config.milvus_collection,
                stored_path=str(stored_path),
                uploaded_at=uploaded_at,
            )
            self._append_manifest(record)
            return record

    def list_documents(self, tenant_id: str = "", limit: int = 50) -> list[RagDocumentRecord]:
        docs = self._read_manifest()
        if tenant_id:
            docs = [item for item in docs if item.tenant_id == tenant_id]
        return list(reversed(docs))[:limit]

    def status(self, tenant_id: str = "default_tenant") -> dict:
        runtime_config = AppConfig.from_file(self._config_path)
        configure_dashscope_endpoint(runtime_config)
        normalized_tenant = self._safe_id(tenant_id or runtime_config.tenant_id)
        rag_config = RAGConfig(
            milvus_host=runtime_config.milvus_host,
            milvus_port=runtime_config.milvus_port,
            collection_name=runtime_config.milvus_collection,
        )
        rag = RAGManager.get_or_none(
            tenant_id=normalized_tenant,
            collection_name=runtime_config.milvus_collection,
        )
        stats: dict = {}
        if rag is not None:
            stats = rag.get_collection_stats()
        return {
            "tenant_id": normalized_tenant,
            "collection": runtime_config.milvus_collection,
            "milvus_host": runtime_config.milvus_host,
            "milvus_port": runtime_config.milvus_port,
            "configured_enabled": runtime_config.enable_milvus,
            "runtime_initialized": rag is not None,
            "stats": stats,
            "documents": self.list_documents(normalized_tenant, limit=20),
        }

    def search(self, query: str, tenant_id: str = "default_tenant", limit: int = 5) -> list[dict]:
        with trace_run(
            "rag.search",
            run_type="retriever",
            inputs={"query": query, "limit": limit},
            metadata={"tenant_id": tenant_id},
            tags=("rag", "retrieval"),
        ) as span:
            runtime_config = AppConfig.from_file(self._config_path)
            configure_dashscope_endpoint(runtime_config)
            normalized_tenant = self._safe_id(tenant_id or runtime_config.tenant_id)
            rag_config = RAGConfig(
                milvus_host=runtime_config.milvus_host,
                milvus_port=runtime_config.milvus_port,
                collection_name=runtime_config.milvus_collection,
            )
            rag = RAGManager.get(runtime_config.api_key, rag_config, tenant_id=normalized_tenant)
            results = rag.search_records(query, k=limit)
            span.end(outputs={"result_count": len(results), "tenant_id": normalized_tenant})
            return results

    def _append_manifest(self, record: RagDocumentRecord) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self._manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def _read_manifest(self) -> list[RagDocumentRecord]:
        if not self._manifest_path.exists():
            return []
        records: list[RagDocumentRecord] = []
        for line in self._manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(RagDocumentRecord.model_validate(json.loads(line)))
            except Exception:
                continue
        return records

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path(filename or "document").name.strip()
        name = re.sub(r"[^\w.\-\u4e00-\u9fff ]+", "_", name, flags=re.UNICODE).strip(" .")
        return name or f"document_{uuid4().hex[:8]}"

    @staticmethod
    def _safe_id(value: str) -> str:
        text = (value or "default_tenant").strip()
        text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)
        return text or "default_tenant"

    def _enforce_tenant_storage_limit(self, tenant_id: str, upload_size: int) -> None:
        if self._max_tenant_storage_bytes <= 0:
            return
        used = sum(item.size_bytes for item in self._read_manifest() if item.tenant_id == tenant_id)
        if used + upload_size > self._max_tenant_storage_bytes:
            limit_mb = self._max_tenant_storage_bytes / 1024 / 1024
            raise RagDocumentError(f"Tenant storage quota exceeded. Limit is {limit_mb:.0f} MB.", 413)

    @staticmethod
    def _has_expected_signature(suffix: str, content: bytes) -> bool:
        if suffix == ".pdf":
            return content.lstrip().startswith(b"%PDF")
        if suffix == ".docx":
            return content.startswith(b"PK")
        return False
