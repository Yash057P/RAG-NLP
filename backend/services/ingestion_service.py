"""Ingestion orchestration: run the six ingestion stages for a file.

``ingest_one`` never raises for an expected failure - it returns a
:class:`DocumentRecord` with ``status="error"`` and the failing stage marked, so
the UI can show exactly which step broke.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from backend.core.config import settings
from backend.core.errors import (
    EmptyDocumentError,
    ExtractionError,
    RAGError,
    UnsupportedFileTypeError,
)
from backend.core.logging import get_logger
from backend.models.schemas import (
    DocumentRecord,
    FailedUpload,
    PipelineTrace,
    StageKey,
    utc_now,
)
from backend.services.document_processor import process_file
from backend.services.embedding_service import get_embedder
from backend.services.pipeline import INGESTION_STAGES, TraceRecorder
from backend.services.registry import get_registry
from backend.services.vector_store import get_vector_store
from backend.utils.files import (
    content_fingerprint,
    file_extension,
    generate_id,
    human_size,
    safe_filename,
    split_stored_filename,
)

logger = get_logger(__name__)


def _chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}::{index:05d}"


def _validate_extension(extension: str) -> None:
    allowed = {ext.lower() for ext in settings.ALLOWED_EXTENSIONS}
    if extension not in allowed:
        raise UnsupportedFileTypeError(
            f"'.{extension}' files are not supported. "
            f"Allowed types: {', '.join('.' + ext for ext in sorted(allowed))}",
            details={"extension": extension, "allowed": sorted(allowed)},
        )


def save_upload(source_path: Path, original_filename: str) -> tuple[Path, str]:
    """Copy an uploaded temp file into ``uploads/`` under a collision-free name."""
    stored = safe_filename(original_filename)
    target = settings.UPLOAD_DIR / stored
    if target.exists():
        target = settings.UPLOAD_DIR / f"{Path(stored).stem}_{generate_id()[:6]}{Path(stored).suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source_path.read_bytes())
    return target, target.name


def ingest_one(
    *,
    stored_path: Path,
    original_filename: str,
    size_bytes: int,
    content_type: str | None = None,
    fingerprint: str = "",
    document_id: str | None = None,
    skip_ingest_stage: bool = False,
) -> tuple[DocumentRecord, PipelineTrace]:
    """Process, embed and index a single document. Returns record + trace."""
    extension = file_extension(original_filename)
    doc_id = document_id or generate_id("doc_")
    filename = safe_filename(original_filename)

    record = DocumentRecord(
        id=doc_id,
        filename=filename,
        extension=extension,
        size_bytes=size_bytes,
        size_display=human_size(size_bytes),
        status="processing",
        content_type=content_type,
        fingerprint=fingerprint or content_fingerprint(stored_path.read_bytes()),
        # Recorded here (and not by the caller) so rebuilds and single-file
        # reindexing can find - and delete - the original file again.
        stored_filename=stored_path.name,
    )
    recorder = TraceRecorder("ingestion")

    # -- stage 1: ingest ---------------------------------------------------
    if skip_ingest_stage:
        recorder.mark(
            StageKey.INGEST,
            "cached",
            detail=f"restored from disk ({record.size_display})",
        )
    else:
        recorder.mark(
            StageKey.INGEST,
            "success",
            detail=f"{filename} ({record.size_display})",
            size_bytes=size_bytes,
            extension=extension,
        )

    # -- stages 2-4: extraction, cleaning, chunking -----------------------
    try:
        _validate_extension(extension)
        processing = process_file(stored_path, extension)
    except RAGError as exc:
        failed_stage = (
            StageKey.CHUNKING
            if isinstance(exc, EmptyDocumentError)
            else StageKey.EXTRACTION
        )
        recorder.mark(failed_stage, "error", detail=exc.message)
        for key in INGESTION_STAGES:
            recorder.stage(key)  # ensure ordering is intact
        return _finalize_error(record, recorder, exc.message), recorder.finish("error")
    except Exception as exc:  # noqa: BLE001
        message = f"Unexpected error while processing the file: {exc}"
        logger.exception("Ingestion failed for %s", original_filename)
        recorder.mark(StageKey.EXTRACTION, "error", detail=message)
        return _finalize_error(record, recorder, message), recorder.finish("error")

    timings = processing.timing_ms
    recorder.mark(
        StageKey.EXTRACTION,
        "success",
        duration_ms=timings["extraction"],
        detail=f"{processing.raw_char_count:,} characters via {processing.extraction_method}",
        method=processing.extraction_method,
        raw_characters=processing.raw_char_count,
        **({"pages": processing.page_count} if processing.page_count else {}),
    )

    removed = processing.raw_char_count - processing.clean_char_count
    removal_pct = (
        round(removed / processing.raw_char_count * 100, 2)
        if processing.raw_char_count
        else 0.0
    )
    recorder.mark(
        StageKey.CLEANING,
        "success",
        duration_ms=timings["cleaning"],
        detail=f"{processing.clean_char_count:,} characters kept ({removal_pct}% noise removed)",
        clean_characters=processing.clean_char_count,
        noise_removed_pct=removal_pct,
        words=processing.word_count,
    )

    chunks = processing.chunks
    sizes = [len(chunk.text) for chunk in chunks] or [0]
    recorder.mark(
        StageKey.CHUNKING,
        "success",
        duration_ms=timings["chunking"],
        detail=f"{len(chunks)} chunks of {settings.CHUNK_SIZE} chars (overlap {settings.CHUNK_OVERLAP})",
        chunk_count=len(chunks),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        min_chunk_chars=min(sizes),
        max_chunk_chars=max(sizes),
        average_chunk_chars=round(sum(sizes) / len(sizes), 1),
    )

    # -- stage 5: embeddings ----------------------------------------------
    embedder = get_embedder()
    record.embedding_model = embedder.name
    chunk_texts = [chunk.text for chunk in chunks]
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    for chunk in chunks:
        ids.append(_chunk_id(doc_id, chunk.index))
        metadata: dict[str, Any] = {
            "document_id": doc_id,
            "filename": filename,
            "extension": extension,
            "chunk_index": chunk.index,
            "char_count": len(chunk.text),
        }
        # ChromaDB only stores str/int/float/bool metadata, so an unknown page
        # is omitted rather than written as None.
        if chunk.page is not None:
            metadata["page"] = chunk.page
        if chunk.char_start is not None:
            metadata["char_start"] = chunk.char_start
        if chunk.char_end is not None:
            metadata["char_end"] = chunk.char_end
        metadatas.append(metadata)

    try:
        started = time.perf_counter()
        vectors: list[list[float]] = []
        batch = max(settings.EMBEDDING_BATCH_SIZE, 1)
        for offset in range(0, len(chunk_texts), batch):
            vectors.extend(embedder.embed_documents(chunk_texts[offset : offset + batch]))
        embedding_ms = (time.perf_counter() - started) * 1000

        recorder.mark(
            StageKey.EMBEDDING,
            "success",
            duration_ms=embedding_ms,
            detail=f"{len(vectors)} vectors from {embedder.name}",
            chunks_embedded=len(vectors),
            model=embedder.name,
            provider=embedder.provider,
            dimension=embedder.dimension,
        )

        # -- stage 6: vector store ----------------------------------------
        started = time.perf_counter()
        store = get_vector_store()
        stored = store.add(ids, vectors, chunk_texts, metadatas)
        store_ms = (time.perf_counter() - started) * 1000
        recorder.mark(
            StageKey.VECTOR_STORE,
            "success",
            duration_ms=store_ms,
            detail=f"{stored} chunks written to {store.backend}",
            backend=store.backend,
            chunks_written=stored,
            total_chunks=store.count(),
            collection=settings.CHROMA_COLLECTION,
        )
    except Exception as exc:  # noqa: BLE001
        message = f"Embedding or indexing failed: {exc}"
        recorder.mark(StageKey.EMBEDDING, "error", detail=message)
        return _finalize_error(record, recorder, message), recorder.finish("error")

    # -- success ------------------------------------------------------------
    record.status = "ready"
    record.chunk_count = len(chunks)
    record.page_count = processing.page_count
    record.char_count = processing.clean_char_count
    record.word_count = processing.word_count
    record.error = None
    record.updated_at = utc_now()
    record.processing_ms = sum(stage.duration_ms or 0.0 for stage in recorder.trace.stages)
    record.stages = recorder.trace.stages

    trace = recorder.finish("success")
    record.stages = trace.stages
    logger.info(
        "Indexed %s -> %d chunks in %.0fms", filename, len(chunks), record.processing_ms
    )
    return record, trace


def _finalize_error(
    record: DocumentRecord, recorder: TraceRecorder, message: str
) -> DocumentRecord:
    record.status = "error"
    record.error = message
    record.updated_at = utc_now()
    trace = recorder.trace
    record.stages = trace.stages
    return record


def register(record: DocumentRecord) -> DocumentRecord:
    return get_registry().add(record)


def build_failed_upload(record: DocumentRecord) -> FailedUpload:
    return FailedUpload(
        filename=record.filename,
        reason=record.error or "Processing failed",
        code="processing_failed",
    )


def rebuild_index() -> tuple[PipelineTrace, int, int]:
    """Wipe the vector store and re-ingest every file still in ``uploads/``."""
    recorder = TraceRecorder("rebuild", pre_marked={key: "pending" for key in INGESTION_STAGES})
    registry = get_registry()
    store = get_vector_store()

    recorder.mark(
        StageKey.VECTOR_STORE,
        "success",
        duration_ms=0.0,
        detail=f"cleared {store.count()} vectors from {store.backend}",
        cleared=True,
    )
    store.reset()
    registry.clear()

    total_chunks = 0
    processed = 0

    for path in sorted(settings.UPLOAD_DIR.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        extension = file_extension(path.name)
        if extension not in {ext.lower() for ext in settings.ALLOWED_EXTENSIONS}:
            continue
        # The file on disk is "{document_id}_{original}"; restore both so the
        # rebuild keeps the same ids and shows the original filename in the UI.
        document_id, original_name = split_stored_filename(path.name)
        try:
            new_record, _ = ingest_one(
                stored_path=path,
                original_filename=original_name,
                size_bytes=path.stat().st_size,
                fingerprint=content_fingerprint(path.read_bytes()),
                document_id=document_id,
                skip_ingest_stage=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Rebuild failed for %s: %s", path.name, exc)
            continue
        registry.add(new_record)
        processed += 1
        if new_record.status == "ready":
            total_chunks += new_record.chunk_count

    recorder.mark(
        StageKey.INGEST,
        "cached",
        detail=f"{processed} file(s) discovered in uploads/",
    )
    recorder.mark(
        StageKey.VECTOR_STORE,
        "success",
        detail=f"{total_chunks} chunks across {processed} document(s)",
        backend=store.backend,
        total_chunks=total_chunks,
    )
    logger.info("Rebuild complete: %d documents, %d chunks", processed, total_chunks)
    return recorder.finish("success"), processed, total_chunks
