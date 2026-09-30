"""System health, pipeline metadata and destructive maintenance operations."""

from __future__ import annotations

import shutil
import time

from fastapi import APIRouter

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.models.schemas import (
    ComponentHealth,
    HealthResponse,
    MessageResponse,
    SystemStatus,
)
from backend.prompts.templates import EXAMPLE_QUESTIONS
from backend.services.analytics import get_analytics
from backend.services.embedding_service import embedder_info
from backend.services.llm_factory import get_llm_router
from backend.services.pipeline import stage_definitions
from backend.services.rag_service import reset_rag_service
from backend.services.registry import get_registry
from backend.services.vector_store import dispose_vector_store, get_vector_store

logger = get_logger(__name__)

router = APIRouter(prefix="/system", tags=["system"])

_STARTED_AT = time.time()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        uptime_seconds=round(time.time() - _STARTED_AT, 2),
    )


@router.get("/status", response_model=SystemStatus)
async def status_report() -> SystemStatus:
    """Component health plus the static pipeline definition for the diagram."""
    embeddings = embedder_info()
    llm_info = get_llm_router().info()
    store = get_vector_store().status()

    llm_health = ComponentHealth(
        name=f"LLM ({llm_info['provider']})",
        status="ok" if llm_info["available"] else "degraded",
        detail=str(llm_info.get("detail") or llm_info["model"]),
    )
    embedding_health = ComponentHealth(
        name=f"Embeddings ({embeddings['provider']})",
        status="ok" if embeddings["available"] else "error",
        detail=f"{embeddings['model']} - {embeddings['detail']}",
    )
    store_health = ComponentHealth(
        name=f"Vector store ({store.get('backend', 'unknown')})",
        status="ok" if store.get("status") in ("ready", "empty") else "error",
        detail=(
            f"{store.get('chunks', 0)} chunks in '{store.get('collection', '')}' "
            f"({store.get('metric', 'cosine')})"
            if store.get("status") != "error"
            else str(store.get("error", "unknown error"))
        ),
    )

    statuses = {llm_health.status, embedding_health.status, store_health.status}
    overall = "error" if "error" in statuses else ("degraded" if "degraded" in statuses else "ok")

    return SystemStatus(
        status=overall,  # type: ignore[arg-type]
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        llm=llm_health,
        embeddings=embedding_health,
        vector_store=store_health,
        pipeline_stages=stage_definitions(),
        example_questions=EXAMPLE_QUESTIONS,
        config=settings.public_dict(),
    )


@router.post("/reset", response_model=MessageResponse)
async def reset() -> MessageResponse:
    """Wipe uploads, the vector store, the registry and the question history."""
    store = get_vector_store()
    cleared_chunks = store.count()
    store.reset()

    documents = get_registry().clear()
    history = get_analytics().clear()

    # ChromaDB holds its SQLite file open. The handle has to go *before* the
    # directory is removed, otherwise the next rebuild writes to an unlinked
    # file and every write fails with "attempt to write a readonly database".
    dispose_vector_store()
    reset_rag_service()

    for path in (settings.UPLOAD_DIR, settings.VECTOR_STORE_DIR):
        shutil.rmtree(path, ignore_errors=True)
    settings.ensure_directories()

    logger.info(
        "System reset: %d documents, %d chunks, %d history entries",
        documents,
        cleared_chunks,
        history,
    )
    return MessageResponse(
        message=(
            f"Reset complete: removed {documents} document(s), {cleared_chunks} "
            f"chunk(s) and {history} history entry(ies)."
        )
    )
