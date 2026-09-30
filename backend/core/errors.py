"""Application level exceptions and their FastAPI handlers."""

from __future__ import annotations

from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse

from backend.core.logging import get_logger

logger = get_logger(__name__)


class RAGError(Exception):
    """Base class for every error the application raises deliberately."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class UnsupportedFileTypeError(RAGError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_file_type"


class FileTooLargeError(RAGError):
    status_code = 413
    code = "file_too_large"


class EmptyDocumentError(RAGError):
    status_code = 422
    code = "empty_document"


class DocumentNotFoundError(RAGError):
    status_code = 404
    code = "document_not_found"


class ExtractionError(RAGError):
    status_code = 422
    code = "extraction_failed"


class LLMUnavailableError(RAGError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "llm_unavailable"


class VectorStoreError(RAGError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "vector_store_error"


def register_exception_handlers(app: Any) -> None:
    """Attach handlers for :class:`RAGError` and unexpected exceptions."""

    @app.exception_handler(RAGError)
    async def _rag_error_handler(_: Request, exc: RAGError) -> JSONResponse:
        logger.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred. Check the server logs.",
                    "details": {"exception": type(exc).__name__},
                }
            },
        )
