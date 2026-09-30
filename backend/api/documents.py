"""Document upload, listing, deletion and index rebuild."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi import BackgroundTasks

from backend.core.config import settings
from backend.core.errors import (
    DocumentNotFoundError,
    FileTooLargeError,
    RAGError,
)
from backend.core.logging import get_logger
from backend.models.schemas import (
    DeleteResponse,
    DocumentListResponse,
    DocumentRecord,
    FailedUpload,
    PipelineTrace,
    RebuildResponse,
    RetrievedChunk,
    UploadResponse,
)
from backend.services.embedding_service import get_embedder
from backend.services.ingestion_service import (
    build_failed_upload,
    ingest_one,
    rebuild_index,
)
from backend.services.registry import get_registry
from backend.services.rag_service import _to_retrieved
from backend.services.vector_store import get_vector_store
from backend.utils.files import content_fingerprint, file_extension, generate_id, safe_filename

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Documents that fit comfortably in memory for chunk preview.
PREVIEW_LIMIT = 5


async def _read_upload(upload: UploadFile) -> bytes:
    """Read an upload, refusing anything above the configured size limit."""
    limit = settings.max_upload_bytes
    buffer = bytearray()
    while True:
        block = await upload.read(1024 * 1024)
        if not block:
            break
        buffer.extend(block)
        if len(buffer) > limit:
            raise FileTooLargeError(
                f"'{upload.filename}' exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit."
            )
    return bytes(buffer)


def _stored_path(document_id: str, filename: str) -> Path:
    return settings.UPLOAD_DIR / f"{document_id}_{safe_filename(filename)}"


@router.get("", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """Every registered document, newest first."""
    documents = get_registry().list()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    """Upload one or more documents and run them through the RAG pipeline."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No files were provided."
        )

    registry = get_registry()
    created: list[DocumentRecord] = []
    failed: list[FailedUpload] = []
    trace: PipelineTrace | None = None

    for upload in files:
        filename = upload.filename or "unnamed"
        extension = file_extension(filename)

        if extension not in {ext.lower() for ext in settings.ALLOWED_EXTENSIONS}:
            failed.append(
                FailedUpload(
                    filename=filename,
                    reason=(
                        f"Unsupported type '.{extension}'. Allowed: "
                        f"{', '.join('.' + e for e in settings.ALLOWED_EXTENSIONS)}"
                    ),
                    code="unsupported_file_type",
                )
            )
            continue

        try:
            data = await _read_upload(upload)
        except RAGError as exc:
            failed.append(FailedUpload(filename=filename, reason=exc.message, code=exc.code))
            continue

        if not data:
            failed.append(
                FailedUpload(
                    filename=filename, reason="File is empty.", code="empty_file"
                )
            )
            continue

        fingerprint = content_fingerprint(data)

        # Re-uploading the same named file replaces the previous copy so the
        # index never contains two identical documents.
        existing = registry.find_by_fingerprint(fingerprint)
        if existing and existing.filename == safe_filename(filename):
            logger.info("Replacing previously uploaded %s", existing.filename)
            get_vector_store().delete_document(existing.id)
            registry.remove(existing.id)

        document_id = generate_id("doc_")
        stored = _stored_path(document_id, filename)
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_bytes(data)

        record, file_trace = ingest_one(
            stored_path=stored,
            original_filename=filename,
            size_bytes=len(data),
            content_type=upload.content_type,
            fingerprint=fingerprint,
            document_id=document_id,
        )
        registry.add(record)

        if record.status == "ready":
            created.append(record)
            trace = trace or file_trace
            logger.info(
                "Upload %s -> %d chunks (%s)",
                record.filename,
                record.chunk_count,
                get_embedder().name,
            )
        else:
            failed.append(build_failed_upload(record))

    if created and not failed:
        message = f"Indexed {len(created)} document(s)."
    elif created:
        message = f"Indexed {len(created)} document(s); {len(failed)} failed."
    elif failed:
        message = f"No documents were indexed; {len(failed)} failed."
    else:
        message = "Nothing to do."

    return UploadResponse(created=created, failed=failed, pipeline=trace, message=message)


@router.get("/{document_id}", response_model=DocumentRecord)
async def get_document(document_id: str) -> DocumentRecord:
    """Metadata for a single document."""
    return get_registry().get(document_id)


@router.get("/{document_id}/chunks", response_model=list[RetrievedChunk])
async def get_document_chunks(
    document_id: str, limit: int = PREVIEW_LIMIT
) -> list[RetrievedChunk]:
    """A preview of the first stored chunks, for inspecting the chunking step."""
    get_registry().get(document_id)  # 404 if unknown
    chunks = get_vector_store().get_chunks(document_id, limit=max(1, min(limit, 25)))
    return [_to_retrieved(chunk) for chunk in chunks]


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str) -> DeleteResponse:
    """Remove a document from the registry, the vector store and disk."""
    registry = get_registry()
    record = registry.get(document_id)  # 404 if unknown

    removed = get_vector_store().delete_document(document_id)
    registry.remove(document_id)

    if record.stored_filename:
        path = settings.UPLOAD_DIR / record.stored_filename
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - permissions
            logger.warning("Could not delete %s: %s", path, exc)

    logger.info("Deleted %s (%d chunks)", record.filename, removed)
    return DeleteResponse(
        deleted_id=document_id,
        filename=record.filename,
        chunks_removed=removed,
        message=f"Deleted '{record.filename}' and {removed} chunk(s).",
    )


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild(background: BackgroundTasks) -> RebuildResponse:
    """Wipe the vector store and re-ingest every file in ``uploads/``."""
    logger.info("Rebuild requested")
    trace, processed, total_chunks = rebuild_index()
    return RebuildResponse(
        pipeline=trace,
        documents_processed=processed,
        total_chunks=total_chunks,
        message=f"Rebuilt the index from {processed} file(s) into {total_chunks} chunks.",
    )


@router.post("/{document_id}/reindex", response_model=DocumentRecord)
async def reindex_document(document_id: str) -> DocumentRecord:
    """Re-process a single document with the current pipeline settings."""
    registry = get_registry()
    record = registry.get(document_id)

    source = (
        settings.UPLOAD_DIR / record.stored_filename
        if record.stored_filename
        else None
    )
    if source is None or not source.exists():
        raise DocumentNotFoundError(
            f"The uploaded file for '{record.filename}' is no longer on disk."
        )

    get_vector_store().delete_document(document_id)
    new_record, _ = ingest_one(
        stored_path=source,
        original_filename=record.filename,
        size_bytes=record.size_bytes,
        content_type=record.content_type,
        fingerprint=record.fingerprint,
        document_id=document_id,
        skip_ingest_stage=True,
    )
    registry.add(new_record)
    return new_record