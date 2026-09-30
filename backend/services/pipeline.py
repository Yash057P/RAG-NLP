"""Helpers for building the eight stage RAG pipeline trace shown in the GUI."""

from __future__ import annotations

import time

from backend.models.schemas import (
    STAGE_DEFINITIONS,
    PipelineStage,
    PipelineStageDefinition,
    PipelineTrace,
    StageKey,
    StageStatus,
)
from backend.utils.files import generate_id

INGESTION_STAGES = (
    StageKey.INGEST,
    StageKey.EXTRACTION,
    StageKey.CLEANING,
    StageKey.CHUNKING,
    StageKey.EMBEDDING,
    StageKey.VECTOR_STORE,
)

QUERY_STAGES = (StageKey.RETRIEVAL, StageKey.GENERATION)

ALL_STAGES = INGESTION_STAGES + QUERY_STAGES


def stage_definitions() -> list[PipelineStageDefinition]:
    """Static stage metadata used to render the diagram before anything runs."""
    return [
        PipelineStageDefinition(
            key=key.value,
            label=STAGE_DEFINITIONS[key][1],
            description=STAGE_DEFINITIONS[key][2],
        )
        for key in ALL_STAGES
    ]


def _blank_stage(key: StageKey, status: StageStatus = "pending") -> PipelineStage:
    _, label, description = STAGE_DEFINITIONS[key]
    return PipelineStage(
        key=key.value, label=label, description=description, status=status
    )


class TraceRecorder:
    """Small builder that records per-stage timings and metrics."""

    def __init__(self, operation: str, *, pre_marked: dict[StageKey, StageStatus] | None = None) -> None:
        self.trace = PipelineTrace(
            id=generate_id("trace_"),
            operation=operation,  # type: ignore[arg-type]
            status="running",
            started_at=_now(),
        )
        pre_marked = pre_marked or {}
        for key in ALL_STAGES:
            self.trace.stages.append(_blank_stage(key, pre_marked.get(key, "pending")))
        self._started = time.perf_counter()

    # -- stage updates ------------------------------------------------------
    def mark(
        self,
        key: StageKey,
        status: StageStatus,
        *,
        duration_ms: float | None = None,
        detail: str | None = None,
        **metrics: object,
    ) -> PipelineStage:
        for stage in self.trace.stages:
            if stage.key == key.value:
                stage.status = status
                if duration_ms is not None:
                    stage.duration_ms = round(duration_ms, 2)
                if detail is not None:
                    stage.detail = detail
                if metrics:
                    stage.metrics.update({k: v for k, v in metrics.items() if v is not None})
                return stage
        stage = _blank_stage(key, status)
        stage.duration_ms = round(duration_ms, 2) if duration_ms is not None else None
        stage.detail = detail
        stage.metrics.update({k: v for k, v in metrics.items() if v is not None})
        self.trace.stages.append(stage)
        return stage

    def mark_all(
        self, keys: tuple[StageKey, ...], status: StageStatus, detail: str | None = None
    ) -> None:
        for key in keys:
            self.mark(key, status, detail=detail)

    def stage(self, key: StageKey) -> PipelineStage:
        for candidate in self.trace.stages:
            if candidate.key == key.value:
                return candidate
        return self.mark(key, "pending")

    # -- completion ---------------------------------------------------------
    def finish(self, status: str = "success") -> PipelineTrace:
        self.trace.total_duration_ms = round((time.perf_counter() - self._started) * 1000, 2)
        self.trace.status = status  # type: ignore[assignment]
        return self.trace


def query_trace(
    *,
    total_documents: int,
    total_chunks: int,
    candidates: int,
    returned: int,
    top_score: float,
    retrieval_ms: float,
    embedding_ms: float,
    search_ms: float,
    generation_ms: float,
    llm_model: str,
    grounded: bool,
    total_ms: float,
) -> PipelineTrace:
    """A query trace where the ingestion stages are shown as already cached."""
    recorder = TraceRecorder(
        "query",
        pre_marked={key: "cached" for key in INGESTION_STAGES},
    )
    recorder.mark(
        StageKey.RETRIEVAL,
        "success",
        duration_ms=retrieval_ms,
        detail=f"{returned} of {candidates} chunks returned",
        top_k_returned=returned,
        total_candidates=candidates,
        top_score=round(top_score, 4),
        query_embedding_ms=round(embedding_ms, 2),
        similarity_search_ms=round(search_ms, 2),
    )
    recorder.mark(
        StageKey.GENERATION,
        "success" if grounded else "skipped",
        duration_ms=generation_ms,
        detail=llm_model if grounded else "abstained - no supporting context",
        llm=llm_model,
    )
    recorder.trace.stages[0].detail = f"{total_documents} document(s) indexed"
    recorder.trace.stages[5].detail = f"{total_chunks} chunks searchable"
    trace = recorder.finish("success")
    # This recorder is built *after* retrieval and generation have already run,
    # so its own elapsed time is meaningless. The caller measures the real
    # wall-clock total and hands it in instead.
    trace.total_duration_ms = round(total_ms, 2)
    return trace


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
