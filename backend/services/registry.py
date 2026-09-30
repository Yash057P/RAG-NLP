"""Persistent registry of uploaded documents (JSON on disk).

ChromaDB holds the vectors; this registry holds the human-facing metadata
(status, sizes, timings, per-document pipeline trace) that the dashboard and
the document list render.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from backend.core.config import settings
from backend.core.errors import DocumentNotFoundError
from backend.core.logging import get_logger
from backend.models.schemas import DocumentRecord

logger = get_logger(__name__)


class DocumentRegistry:
    """Thread-safe, JSON-backed collection of :class:`DocumentRecord`."""

    def __init__(self, path: Any | None = None) -> None:
        self._path = path or settings.REGISTRY_PATH
        self._lock = threading.RLock()
        self._records: dict[str, DocumentRecord] = {}
        self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        path = self._path
        if not path.exists():
            logger.debug("No document registry at %s, starting empty", path)
            return
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read registry %s: %s", path, exc)
            return

        for item in payload.get("documents", []):
            try:
                self._records[item["id"]] = DocumentRecord.model_validate(item)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed registry entry: %s", exc)
        logger.info("Loaded %d document(s) from registry", len(self._records))

    def _flush(self) -> None:
        path = self._path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [
                record.model_dump(mode="json") for record in self._sorted()
            ]
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
        tmp.replace(path)

    def _sorted(self) -> list[DocumentRecord]:
        return sorted(self._records.values(), key=lambda r: r.uploaded_at, reverse=True)

    # -- reads --------------------------------------------------------------
    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return self._sorted()

    def get(self, document_id: str) -> DocumentRecord:
        with self._lock:
            record = self._records.get(document_id)
        if record is None:
            raise DocumentNotFoundError(
                f"No document with id '{document_id}' is registered."
            )
        return record

    def find_by_fingerprint(self, fingerprint: str) -> DocumentRecord | None:
        with self._lock:
            for record in self._records.values():
                if record.fingerprint == fingerprint:
                    return record
        return None

    def by_status(self, status: str) -> list[DocumentRecord]:
        with self._lock:
            return [record for record in self._sorted() if record.status == status]

    # -- writes -------------------------------------------------------------
    def add(self, record: DocumentRecord) -> DocumentRecord:
        with self._lock:
            self._records[record.id] = record
            self._flush()
        return record

    def update(self, document_id: str, **fields: Any) -> DocumentRecord:
        with self._lock:
            record = self.get(document_id)
            for key, value in fields.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            self._flush()
        return record

    def remove(self, document_id: str) -> DocumentRecord:
        with self._lock:
            record = self.get(document_id)
            del self._records[document_id]
            self._flush()
        return record

    def clear(self) -> int:
        with self._lock:
            count = len(self._records)
            self._records.clear()
            self._flush()
        return count

    # -- aggregates ---------------------------------------------------------
    def totals(self) -> dict[str, Any]:
        with self._lock:
            records = self._sorted()
        indexed = [r for r in records if r.status == "ready"]
        return {
            "total_documents": len(records),
            "documents_ready": len(indexed),
            "documents_error": sum(1 for r in records if r.status == "error"),
            "registered_chunks": sum(r.chunk_count for r in indexed),
            "total_size_bytes": sum(r.size_bytes for r in records),
            "last_indexed_at": (
                max((r.updated_at for r in indexed), default=None)
            ),
        }


_registry: DocumentRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> DocumentRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = DocumentRegistry()
    return _registry
