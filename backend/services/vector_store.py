"""Vector storage and similarity search.

Two interchangeable backends sit behind :class:`BaseVectorStore`:

* :class:`LangChainChromaVectorStore` - the default, using LangChain's
  ``langchain_chroma.Chroma`` integration (with the persistent HNSW index).
* :class:`ChromaVectorStore` - same persistent collection, driven directly
  through the ``chromadb`` client when the LangChain wrapper is unavailable.
* :class:`NumpyVectorStore` - a dependency-free, on-disk cosine index used
  when ChromaDB cannot be initialised at all.

Both normalise vectors, return cosine *similarity* in ``[-1, 1]`` (higher is
better) and support filtering by ``document_id`` so individual documents can be
deleted or the whole index rebuilt.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ScoredChunk:
    """One retrieved chunk with its similarity score."""

    id: str
    text: str
    score: float
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        return str(self.metadata.get("document_id", ""))

    @property
    def filename(self) -> str:
        return str(self.metadata.get("filename", "unknown"))


@runtime_checkable
class BaseVectorStore(Protocol):
    backend: str
    location: str

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int: ...

    def query(
        self, embedding: list[float], top_k: int, document_id: str | None = None
    ) -> list[ScoredChunk]: ...

    def delete_document(self, document_id: str) -> int: ...

    def count(self) -> int: ...

    def count_by_document(self) -> dict[str, int]: ...

    def get_chunks(self, document_id: str, limit: int = 10) -> list[ScoredChunk]: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...

    def status(self) -> dict[str, Any]: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop values ChromaDB cannot store.

    Chroma only accepts str/int/float/bool metadata, so ``None`` (used for an
    unknown page number) and empty strings are removed rather than passed
    through and rejected at write time.
    """
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _chunks_from_payload(result: dict[str, Any]) -> list[ScoredChunk]:
    """Build :class:`ScoredChunk` objects from a Chroma ``get()`` payload."""
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        ScoredChunk(
            id=ids[index],
            text=documents[index] if index < len(documents) else "",
            score=0.0,
            distance=1.0,
            metadata=dict(metadatas[index]) if index < len(metadatas) and metadatas[index] else {},
        )
        for index in range(len(ids))
    ]


# ------------------------------------------------------------------ chroma ---
class ChromaVectorStore:
    """ChromaDB persistent collection using cosine distance."""

    backend = "chroma"

    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        self._persist_directory = persist_directory
        self.location = str(persist_directory)
        self._collection_name = collection_name
        self._lock = threading.RLock()
        self._client: Any = None
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = self._make_client(persist_directory)
        self._collection = self._make_collection()

    # -- construction -------------------------------------------------------
    @staticmethod
    def _make_client(persist_directory: Path) -> Any:
        try:
            import chromadb  # type: ignore[import-untyped]
            from chromadb.config import Settings as ChromaSettings  # type: ignore

            return chromadb.PersistentClient(
                path=str(persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        except TypeError:
            # Newer/older clients that do not accept ``settings``.
            import chromadb  # type: ignore[import-untyped]

            return chromadb.PersistentClient(path=str(persist_directory))

    def _make_collection(self) -> Any:
        try:
            return self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:  # noqa: BLE001 - newer API uses ``configuration``
            logger.debug("Retrying collection creation without legacy metadata")
        return self._client.get_or_create_collection(name=self._collection_name)

    def _reopen(self) -> None:
        """(Re)create the client and collection, e.g. after the directory moved."""
        with self._lock:
            self._persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = self._make_client(self._persist_directory)
            self._collection = self._make_collection()

    def close(self) -> None:
        """Release Chroma's SQLite/HNSW file handles.

        The persistent client holds an open handle on ``chroma.sqlite3``; if that
        handle outlives its directory the next write fails with
        ``attempt to write a readonly database``. Closing the client drops the
        handle so the directory can safely be deleted or rebuilt.
        """
        with self._lock:
            client, self._client = self._client, None
            self._collection = None
        if client is None:
            return
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001 - older clients lack close()
            logger.debug("Chroma client close() ignored: %s", exc)

    def _require_client(self) -> Any:
        if self._client is None:  # pragma: no cover - defensive
            self._reopen()
        return self._client

    # -- writes -------------------------------------------------------------
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        if not ids:
            return 0
        with self._lock:
            clean_metadata = [sanitize_metadata(meta) for meta in metadatas]
            try:
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=clean_metadata,
                )
            except Exception as exc:  # noqa: BLE001 - duplicate ids
                if "id" not in str(exc).lower():
                    raise
                logger.debug("Re-adding existing ids, updating instead")
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=clean_metadata,
                )
        return len(ids)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            existing = self._collection.get(
                where={"document_id": document_id}, include=[]
            )
            removed = len(existing.get("ids", []))
            if removed:
                self._collection.delete(where={"document_id": document_id})
        return removed

    # -- reads --------------------------------------------------------------
    def query(
        self, embedding: list[float], top_k: int, document_id: str | None = None
    ) -> list[ScoredChunk]:
        with self._lock:
            total = self._collection.count()
            if total == 0:
                return []
            where = {"document_id": document_id} if document_id else None
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, total),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[ScoredChunk] = []
        for index, chunk_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            # Cosine distance in [0, 2] -> similarity in [-1, 1].
            chunks.append(
                ScoredChunk(
                    id=chunk_id,
                    text=docs[index] if index < len(docs) else "",
                    score=1.0 - distance,
                    distance=distance,
                    metadata=dict(metas[index]) if index < len(metas) and metas[index] else {},
                )
            )
        return chunks

    def count(self) -> int:
        with self._lock:
            return int(self._collection.count())

    def count_by_document(self) -> dict[str, int]:
        with self._lock:
            result = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in result.get("metadatas") or []:
            if not metadata:
                continue
            key = str(metadata.get("document_id", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_chunks(self, document_id: str, limit: int = 10) -> list[ScoredChunk]:
        """Fetch raw chunks for a document, in insertion order."""
        with self._lock:
            result = self._collection.get(
                where={"document_id": document_id},
                limit=limit,
                include=["documents", "metadatas"],
            )
        return _chunks_from_payload(result)

    def reset(self) -> None:
        """Drop every chunk and return to a writable, empty index."""
        # Dropping the client *and* the collection is deliberate: a client that
        # survives an ``rmtree`` of its directory keeps writing to an unlinked
        # file and fails with "attempt to write a readonly database".
        self.close()
        self._reopen()
        with self._lock:
            try:
                self._require_client().delete_collection(self._collection_name)
            except Exception:  # noqa: BLE001 - collection may not exist yet
                logger.debug("Collection %s did not exist during reset", self._collection_name)
            self._collection = self._make_collection()

    def status(self) -> dict[str, Any]:
        try:
            count = self.count()
            return {
                "backend": self.backend,
                "status": "ready" if count else "empty",
                "collection": self._collection_name,
                "location": self.location,
                "chunks": count,
                "metric": "cosine",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "backend": self.backend,
                "status": "error",
                "collection": self._collection_name,
                "location": self.location,
                "chunks": 0,
                "error": str(exc),
            }


# ---------------------------------------------------- langchain + chroma ---
class _LangChainEmbeddingAdapter:
    """Adapts this module's :class:`Embedder` to LangChain's interface."""

    def __init__(self, embedder: object) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)  # type: ignore[attr-defined]

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)  # type: ignore[attr-defined]


class LangChainChromaVectorStore:
    """ChromaDB through LangChain's ``langchain_chroma.Chroma`` integration.

    Reads go through LangChain's ``similarity_search_with_score`` and deletions
    through its metadata ``where`` filter. Writes use the underlying collection
    handle so the vectors already computed in the embedding stage are stored
    as-is instead of being recomputed.
    """

    backend = "chroma (langchain)"

    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        self._persist_directory = persist_directory
        self.location = str(persist_directory)
        self._collection_name = collection_name
        self._lock = threading.RLock()
        persist_directory.mkdir(parents=True, exist_ok=True)

        self._store = self._make_store()
        self._collection = self._collection_handle(self._store)

    # -- construction -------------------------------------------------------
    def _make_store(self) -> Any:
        from langchain_chroma import Chroma  # type: ignore[import-untyped]

        from backend.services.embedding_service import get_embedder

        return Chroma(
            collection_name=self._collection_name,
            embedding_function=_LangChainEmbeddingAdapter(get_embedder()),
            persist_directory=self.location,
            collection_metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _collection_handle(store: Any) -> Any:
        """Return LangChain's underlying Chroma collection handle.

        ``_collection`` is a property that raises ``ValueError`` (not
        ``AttributeError``) once the collection has been dropped, so it is
        fetched defensively.
        """
        try:
            handle = getattr(store, "_collection", None)
        except ValueError:  # pragma: no cover - collection dropped upstream
            handle = None
        if handle is None:  # pragma: no cover - defensive
            raise RuntimeError("langchain_chroma did not expose its collection handle")
        return handle

    # -- writes -------------------------------------------------------------
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        if not ids:
            return 0
        payload = {
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
            "metadatas": [sanitize_metadata(meta) for meta in metadatas],
        }
        with self._lock:
            try:
                self._collection.add(**payload)
            except Exception as exc:  # noqa: BLE001 - ids already present
                if "id" not in str(exc).lower():
                    raise
                logger.debug("Chunk ids already present, upserting instead")
                self._collection.upsert(**payload)
        return len(ids)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            matches = self._store.get(where={"document_id": document_id}, include=[])
            removed = len(matches.get("ids", []))
            if removed:
                self._store.delete(where={"document_id": document_id})
        return removed

    # -- reads --------------------------------------------------------------
    def query(
        self, embedding: list[float], top_k: int, document_id: str | None = None
    ) -> list[ScoredChunk]:
        with self._lock:
            total = self.count()
            if total == 0:
                return []
            where = {"document_id": document_id} if document_id else None
            # LangChain's own similarity_search_* helpers re-embed their argument,
            # but the RAG service has already embedded the question (and timed
            # it for the pipeline trace), so the query goes straight to Chroma.
            # On a cosine index the returned distance equals 1 - similarity.
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, total),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        return [
            ScoredChunk(
                id=ids[index],
                text=documents[index] if index < len(documents) else "",
                score=1.0 - float(distances[index]) if index < len(distances) else 0.0,
                distance=float(distances[index]) if index < len(distances) else 1.0,
                metadata=dict(metadatas[index]) if index < len(metadatas) and metadatas[index] else {},
            )
            for index in range(len(ids))
        ]

    def count(self) -> int:
        with self._lock:
            try:
                return int(self._collection.count())
            except Exception:  # noqa: BLE001 - clients without a count handle
                return len(self._store.get(include=[]).get("ids", []))

    def count_by_document(self) -> dict[str, int]:
        with self._lock:
            result = self._store.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in result.get("metadatas") or []:
            if not metadata:
                continue
            key = str(metadata.get("document_id", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_chunks(self, document_id: str, limit: int = 10) -> list[ScoredChunk]:
        """Fetch raw chunks for a document, in insertion order."""
        with self._lock:
            result = self._store.get(
                where={"document_id": document_id},
                limit=limit,
                include=["documents", "metadatas"],
            )
        return _chunks_from_payload(result)

    def close(self) -> None:
        """Release Chroma's SQLite/HNSW file handles.

        LangChain's ``Chroma`` wrapper owns the persistent client, which keeps an
        open handle on ``chroma.sqlite3``. Closing it lets the directory be
        deleted (or recreated) without later writes failing with
        ``attempt to write a readonly database``.
        """
        with self._lock:
            store, self._store = getattr(self, "_store", None), None
            self._collection = None
        if store is None:
            return
        client = getattr(store, "_client", None)
        if client is None:  # pragma: no cover - defensive
            return
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001 - older clients lack close()
            logger.debug("Chroma client close() ignored: %s", exc)

    def _reopen(self) -> None:
        """(Re)create the LangChain store and refresh the cached handle."""
        with self._lock:
            self._persist_directory.mkdir(parents=True, exist_ok=True)
            self._store = self._make_store()
            self._collection = self._collection_handle(self._store)

    def reset(self) -> None:
        """Drop every chunk and return to a writable, empty index."""
        try:
            self._store.delete_collection()
        except Exception:  # noqa: BLE001 - collection may not exist yet
            logger.debug("delete_collection ignored for %s", self._collection_name)
        # Dropping the client *and* the collection is deliberate: a client that
        # survives an ``rmtree`` of its directory keeps writing to an unlinked
        # file and fails with "attempt to write a readonly database".
        self.close()
        self._reopen()

    def status(self) -> dict[str, Any]:
        try:
            count = self.count()
            return {
                "backend": self.backend,
                "status": "ready" if count else "empty",
                "collection": self._collection_name,
                "location": self.location,
                "chunks": count,
                "metric": "cosine",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "backend": self.backend,
                "status": "error",
                "collection": self._collection_name,
                "location": self.location,
                "chunks": 0,
                "error": str(exc),
            }


# ------------------------------------------------------------------- numpy ---
class NumpyVectorStore:
    """Flat cosine index persisted as ``vectors.npz`` + ``records.json``."""

    backend = "numpy"

    def __init__(self, directory: Path) -> None:
        self.location = str(directory)
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vectors_path = directory / "vectors.npz"
        self._records_path = directory / "records.json"
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._matrix: np.ndarray = np.zeros((0, settings.EMBEDDING_DIMENSION), dtype=np.float32)
        self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        if not self._records_path.exists():
            return
        try:
            payload = json.loads(self._records_path.read_text("utf-8"))
            self._ids = payload.get("ids", [])
            self._documents = payload.get("documents", [])
            self._metadatas = payload.get("metadatas", [])
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read numpy store metadata: %s", exc)
            return

        if self._vectors_path.exists() and self._ids:
            try:
                with np.load(self._vectors_path) as data:
                    matrix = data["embeddings"].astype(np.float32)
                if matrix.shape[0] == len(self._ids):
                    self._matrix = _l2_normalize(matrix)
                    logger.info(
                        "Loaded %d vectors from %s", matrix.shape[0], self._vectors_path
                    )
                    return
            except (OSError, KeyError, ValueError) as exc:
                logger.warning("Could not read numpy store vectors: %s", exc)
        self._matrix = np.zeros((0, settings.EMBEDDING_DIMENSION), dtype=np.float32)

    def _flush(self) -> None:
        np.savez_compressed(
            self._vectors_path,
            embeddings=self._matrix.astype(np.float32),
        )
        self._records_path.write_text(
            json.dumps(
                {
                    "ids": self._ids,
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                },
                ensure_ascii=False,
            ),
            "utf-8",
        )

    # -- writes -------------------------------------------------------------
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        if not ids:
            return 0
        with self._lock:
            incoming = np.asarray(embeddings, dtype=np.float32)
            if incoming.ndim == 1:
                incoming = incoming.reshape(1, -1)
            normalised = _l2_normalize(incoming)

            keep = [i for i, chunk_id in enumerate(ids) if chunk_id not in set(self._ids)]
            if len(keep) < len(ids):
                logger.debug("Skipping %d duplicate chunk ids", len(ids) - len(keep))
            if not keep:
                return 0

            self._ids.extend(ids[i] for i in keep)
            self._documents.extend(documents[i] for i in keep)
            self._metadatas.extend(dict(metadatas[i]) for i in keep)
            self._matrix = (
                normalised
                if self._matrix.shape[0] == 0
                else np.vstack([self._matrix, normalised])
            )
            self._flush()
        return len(keep)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            if self._matrix.shape[0] == 0:
                return 0
            mask = np.array(
                [meta.get("document_id") == document_id for meta in self._metadatas],
                dtype=bool,
            )
            removed = int(mask.sum())
            if not removed:
                return 0
            keep = ~mask
            self._ids = [self._ids[i] for i in np.nonzero(keep)[0]]
            self._documents = [self._documents[i] for i in np.nonzero(keep)[0]]
            self._metadatas = [self._metadatas[i] for i in np.nonzero(keep)[0]]
            self._matrix = (
                self._matrix[keep]
                if keep.any()
                else np.zeros((0, self._matrix.shape[1]), dtype=np.float32)
            )
            self._flush()
        return removed

    # -- reads --------------------------------------------------------------
    def query(
        self, embedding: list[float], top_k: int, document_id: str | None = None
    ) -> list[ScoredChunk]:
        with self._lock:
            if self._matrix.shape[0] == 0:
                return []

            query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            if query.shape[1] != self._matrix.shape[1]:
                # Model changed since the index was built.
                logger.warning(
                    "Embedding dimension mismatch: index has %d, query has %d",
                    self._matrix.shape[1],
                    query.shape[1],
                )
                return []

            candidates = np.nonzero(
                np.array(
                    [
                        True
                        if document_id is None
                        else meta.get("document_id") == document_id
                        for meta in self._metadatas
                    ],
                    dtype=bool,
                )
            )[0]
            if candidates.size == 0:
                return []

            scores = (self._matrix[candidates] @ _l2_normalize(query)[0]).astype(float)
            order = np.argsort(-scores)[:top_k]

            return [
                ScoredChunk(
                    id=self._ids[candidates[position]],
                    text=self._documents[candidates[position]],
                    score=float(scores[position]),
                    distance=1.0 - float(scores[position]),
                    metadata=dict(self._metadatas[candidates[position]]),
                )
                for position in order
            ]

    def count(self) -> int:
        with self._lock:
            return int(self._matrix.shape[0])

    def count_by_document(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for metadata in self._metadatas:
                key = str(metadata.get("document_id", "unknown"))
                counts[key] = counts.get(key, 0) + 1
            return counts

    def get_chunks(self, document_id: str, limit: int = 10) -> list[ScoredChunk]:
        """Fetch raw chunks for a document, in insertion order."""
        with self._lock:
            positions = [
                index
                for index, metadata in enumerate(self._metadatas)
                if metadata.get("document_id") == document_id
            ][:limit]
        return [
            ScoredChunk(
                id=self._ids[index],
                text=self._documents[index],
                score=0.0,
                distance=1.0,
                metadata=dict(self._metadatas[index]),
            )
            for index in positions
        ]

    def reset(self) -> None:
        with self._lock:
            self._ids.clear()
            self._documents.clear()
            self._metadatas.clear()
            self._matrix = np.zeros(
                (0, settings.EMBEDDING_DIMENSION), dtype=np.float32
            )
            if self._dir.exists():
                shutil.rmtree(self._dir, ignore_errors=True)
            self._dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """No-op: the NumPy index keeps no external file handles."""

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": self.backend,
                "status": "ready" if self._matrix.shape[0] else "empty",
                "collection": "numpy_store",
                "location": self.location,
                "chunks": int(self._matrix.shape[0]),
                "dimension": int(self._matrix.shape[1]),
                "metric": "cosine",
            }


# ---------------------------------------------------------------- factory ---
_store: BaseVectorStore | None = None
_store_lock = threading.RLock()


def _build_store() -> BaseVectorStore:
    """Instantiate the configured backend, degrading gracefully in ``auto`` mode."""
    backend = settings.VECTOR_STORE_BACKEND

    if backend in ("auto", "chroma"):
        errors: list[str] = []
        for factory in (LangChainChromaVectorStore, ChromaVectorStore):
            try:
                store = factory(settings.chroma_persist_path, settings.CHROMA_COLLECTION)
                logger.info("Vector store: %s at %s", store.backend, store.location)
                return store
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{factory.__name__}: {exc}")
                logger.debug("Candidate %s failed: %s", factory.__name__, exc)
        if backend == "chroma":
            raise RuntimeError(
                "ChromaDB could not be initialised: " + "; ".join(errors)
            )
        logger.warning(
            "ChromaDB unavailable (%s); using the built-in NumPy cosine index.",
            "; ".join(errors),
        )

    store = NumpyVectorStore(settings.numpy_store_path)
    logger.info("Vector store: %s at %s", store.backend, store.location)
    return store


def get_vector_store() -> BaseVectorStore:
    """Return the process-wide vector store."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = _build_store()
    return _store


def reset_vector_store() -> None:
    """Drop the cached handle so the factory re-runs (used after a rebuild)."""
    global _store
    with _store_lock:
        _store = None


def dispose_vector_store() -> None:
    """Close the active store and drop the cached handle.

    Must be called *before* ``VECTOR_STORE_DIR`` is deleted: ChromaDB keeps its
    SQLite file open, and a client pointing at a removed directory fails every
    subsequent write with ``attempt to write a readonly database``.
    """
    global _store
    with _store_lock:
        store, _store = _store, None
    closer = getattr(store, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception as exc:  # noqa: BLE001 - closing must never fail a reset
        logger.debug("Ignoring error while closing the vector store: %s", exc)
