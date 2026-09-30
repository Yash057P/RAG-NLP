"""Pydantic schemas shared by the API layer and serialised to the frontend.

The TypeScript types in ``frontend/src/types/api.ts`` mirror these one-to-one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StageStatus = Literal["pending", "running", "success", "error", "cached", "skipped"]
DocumentStatus = Literal["pending", "processing", "ready", "error"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------- pipeline ---
class PipelineStage(BaseModel):
    """One node of the RAG pipeline visualisation."""

    model_config = ConfigDict(populate_by_name=True)

    key: str = Field(..., description="Stable identifier, e.g. 'embedding'")
    label: str = Field(..., description="Human readable name, e.g. 'Embeddings'")
    description: str = ""
    status: StageStatus = "pending"
    duration_ms: float | None = None
    detail: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class PipelineTrace(BaseModel):
    """Ordered list of stages executed for one ingestion or QA operation."""

    id: str
    operation: Literal["ingestion", "query", "rebuild"]
    status: Literal["running", "success", "error"] = "success"
    stages: list[PipelineStage] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    started_at: datetime = Field(default_factory=utc_now)


# -------------------------------------------------------------- documents ---
class DocumentRecord(BaseModel):
    """Registry entry describing one uploaded document."""

    id: str
    filename: str
    extension: str
    size_bytes: int
    size_display: str
    status: DocumentStatus = "pending"
    uploaded_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    chunk_count: int = 0
    page_count: int | None = None
    char_count: int = 0
    word_count: int = 0
    fingerprint: str = ""
    content_type: str | None = None
    stored_filename: str = Field(
        default="", description="Basename inside the uploads/ directory"
    )
    error: str | None = None
    processing_ms: float = 0.0
    embedding_model: str = ""
    stages: list[PipelineStage] = Field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]
    total: int


class FailedUpload(BaseModel):
    filename: str
    reason: str
    code: str = "upload_failed"


class UploadResponse(BaseModel):
    created: list[DocumentRecord] = Field(default_factory=list)
    failed: list[FailedUpload] = Field(default_factory=list)
    pipeline: PipelineTrace | None = None
    message: str = ""


class DeleteResponse(BaseModel):
    deleted_id: str
    filename: str
    chunks_removed: int
    message: str


class RebuildResponse(BaseModel):
    pipeline: PipelineTrace
    documents_processed: int
    total_chunks: int
    message: str


# ------------------------------------------------------------- retrieval ---
class RetrievedChunk(BaseModel):
    """A single chunk returned by the similarity search."""

    id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    snippet: str
    score: float = Field(..., description="Cosine similarity in [-1, 1]")
    distance: float | None = None
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    token_estimate: int = 0
    is_reference: bool = Field(
        default=False,
        description="True for bibliography entries; their score is demoted at retrieval",
    )


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    top_k: int
    total_candidates: int = 0
    top_score: float = 0.0
    average_score: float = 0.0
    grounding_coverage: float = 0.0
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0
    embedding_model: str = ""
    vector_store_backend: str = ""


class Source(BaseModel):
    """Citation shown next to the answer."""

    document_id: str
    filename: str
    chunk_index: int
    page: int | None = None
    snippet: str
    score: float


# ------------------------------------------------------------------ chat ---
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class LLMInfo(BaseModel):
    provider: str
    model: str
    available: bool = True
    fallback_used: bool = False
    detail: str | None = None


class AnswerResponse(BaseModel):
    id: str
    question: str
    answer: str
    grounded: bool = Field(
        ..., description="False when the question could not be supported by the corpus"
    )
    abstained: bool = Field(
        ..., description="True when the guarded no-answer message was returned"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_label: str
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    sources: list[Source] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    retrieval: RetrievalResult
    llm: LLMInfo
    prompt: str = ""
    pipeline: PipelineTrace | None = None
    response_time_ms: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)


class HistoryEntry(BaseModel):
    id: str
    question: str
    answer: str
    confidence: float
    confidence_label: str
    sources: list[Source] = Field(default_factory=list)
    response_time_ms: float
    created_at: datetime
    llm_model: str = ""
    top_score: float = 0.0


class HistoryResponse(BaseModel):
    entries: list[HistoryEntry] = Field(default_factory=list)
    total: int


# ----------------------------------------------------------------- stats ---
class StatsResponse(BaseModel):
    total_documents: int = 0
    documents_ready: int = 0
    documents_error: int = 0
    total_chunks: int = 0
    total_size_bytes: int = 0
    total_size_display: str = "0 B"
    embedding_model: str = ""
    embedding_provider: str = ""
    embedding_dimension: int = 0
    vector_store_status: str = "unknown"
    vector_store_backend: str = "unknown"
    collection_name: str = ""
    questions_asked: int = 0
    average_response_time_ms: float = 0.0
    fastest_response_ms: float | None = None
    slowest_response_ms: float | None = None
    last_question_at: datetime | None = None
    last_indexed_at: datetime | None = None
    llm_provider: str = ""
    llm_model: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    top_k: int = 0


# ---------------------------------------------------------------- system ---
class ComponentHealth(BaseModel):
    name: str
    status: Literal["ok", "degraded", "error", "unknown"]
    detail: str = ""


class PipelineStageDefinition(BaseModel):
    key: str
    label: str
    description: str
    icon: str = ""


class SystemStatus(BaseModel):
    status: Literal["ok", "degraded", "error"] = "ok"
    app_name: str
    version: str
    environment: str
    llm: ComponentHealth
    embeddings: ComponentHealth
    vector_store: ComponentHealth
    pipeline_stages: list[PipelineStageDefinition] = Field(default_factory=list)
    example_questions: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    message: str
    success: bool = True


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


# ------------------------------------------------------------ enumeration ---
class StageKey(str, Enum):
    """Canonical key order for the eight stage RAG pipeline diagram."""

    INGEST = "ingest"
    EXTRACTION = "extraction"
    CLEANING = "cleaning"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    VECTOR_STORE = "vector_store"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"

    @property
    def display(self) -> str:
        return STAGE_DEFINITIONS[self][0]


STAGE_DEFINITIONS: dict[StageKey, tuple[str, str, str]] = {
    StageKey.INGEST: ("ingest", "Document Upload", "Receive PDF / DOCX / TXT files"),
    StageKey.EXTRACTION: ("extraction", "Text Extraction", "Parse raw file into text"),
    StageKey.CLEANING: ("cleaning", "Text Cleaning", "Normalise whitespace and boilerplate"),
    StageKey.CHUNKING: ("chunking", "Chunking", "Split into 500 char chunks (100 overlap)"),
    StageKey.EMBEDDING: ("embedding", "Embeddings", "all-MiniLM-L6-v2 sentence vectors"),
    StageKey.VECTOR_STORE: ("vector_store", "Vector Database", "Persist vectors in ChromaDB"),
    StageKey.RETRIEVAL: ("retrieval", "Retrieval", "Top-5 cosine similarity search"),
    StageKey.GENERATION: ("generation", "Answer Generation", "LLM answers strictly from context"),
}
