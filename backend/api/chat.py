"""Question answering endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.models.schemas import (
    AskRequest,
    AnswerResponse,
    HistoryResponse,
    MessageResponse,
)
from backend.services.analytics import get_analytics
from backend.services.rag_service import get_rag_service

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(payload: AskRequest) -> AnswerResponse:
    """Run the retrieval-augmented generation loop for one question."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be blank.",
        )
    return get_rag_service().answer(question, payload.top_k)


@router.get("/history", response_model=HistoryResponse)
async def history(limit: int | None = Query(default=None, ge=1, le=500)) -> HistoryResponse:
    """Previously asked questions, newest first (drives the export button)."""
    entries = get_analytics().history(limit)
    return HistoryResponse(entries=entries, total=len(entries))


@router.delete("/history", response_model=MessageResponse)
async def clear_history() -> MessageResponse:
    """Clear the question history and reset the analytics counters."""
    removed = get_analytics().clear()
    logger.info("Cleared %d history entries", removed)
    return MessageResponse(message=f"Cleared {removed} question(s) from the history.")


@router.get("/examples", response_model=list[str])
async def examples() -> list[str]:
    """Suggested starter questions."""
    from backend.prompts.templates import EXAMPLE_QUESTIONS

    return EXAMPLE_QUESTIONS


@router.get("/config", response_model=dict)
async def chat_config() -> dict:
    """Retrieval settings the UI needs to describe the current configuration."""
    return {
        "top_k": settings.TOP_K,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "grounding_threshold": settings.GROUNDING_THRESHOLD,
        "min_similarity": settings.MIN_SIMILARITY,
    }
