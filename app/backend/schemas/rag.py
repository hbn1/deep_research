from pydantic import BaseModel, Field


class RagDocumentRecord(BaseModel):
    doc_id: str
    filename: str
    source: str
    tenant_id: str
    user_id: str
    content_type: str
    size_bytes: int
    chunks: int
    collection: str
    stored_path: str
    uploaded_at: str


class RagUploadResponse(BaseModel):
    status: str = "indexed"
    document: RagDocumentRecord


class RagDocumentsResponse(BaseModel):
    documents: list[RagDocumentRecord]


class RagStatusResponse(BaseModel):
    tenant_id: str
    collection: str
    milvus_host: str
    milvus_port: int
    configured_enabled: bool
    runtime_initialized: bool
    stats: dict = Field(default_factory=dict)
    documents: list[RagDocumentRecord] = Field(default_factory=list)


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    tenant_id: str = "default_tenant"
    limit: int = Field(default=5, ge=1, le=20)


class RagSearchResponse(BaseModel):
    query: str
    tenant_id: str
    results: list[dict]
