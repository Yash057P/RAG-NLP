"""Aggregate statistics for the analytics dashboard."""

from __future__ import annotations

from fastapi import APIRouter

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.models.schemas import StatsResponse
from backend.services.analytics import get_analytics
from backend.services.embedding_service import get_embedder
from backend.services.llm_factory import get_llm_router
from backend.services.registry import get_registry
from backend.services.vector_store import get_vector_store
from backend.utils.files import human_size

logger = get_logger(__name__)

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Everything the dashboard cards need, in a single request."""
    registry = get_registry()
    analytics = get_analytics()
    embedder = get_embedder()
    store = get_vector_store()
    llm = get_llm_router().info()

    document_totals = registry.totals()
    store_status = store.status()
    timing = analytics.stats()

    return StatsResponse(
        total_documents=document_totals["total_documents"],
        documents_ready=document_totals["documents_ready"],
        documents_error=document_totals["documents_error"],
        total_chunks=store.count(),
        total_size_bytes=document_totals["total_size_bytes"],
        total_size_display=human_size(document_totals["total_size_bytes"]),
        embedding_model=embedder.name,
        embedding_provider=embedder.provider,
        embedding_dimension=embedder.dimension,
        vector_store_status=store_status.get("status", "unknown"),
        vector_store_backend=store.backend,
        collection_name=store_status.get("collection", settings.CHROMA_COLLECTION),
        questions_asked=timing["questions_asked"],
        average_response_time_ms=timing["average_response_time_ms"],
        fastest_response_ms=timing["fastest_response_ms"],
        slowest_response_ms=timing["slowest_response_ms"],
        last_question_at=timing["last_question_at"],
        last_indexed_at=document_totals["last_indexed_at"],
        llm_provider=llm["provider"],
        llm_model=llm["model"],
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        top_k=settings.TOP_K,
    )
