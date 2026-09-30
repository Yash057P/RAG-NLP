"""Vector store backend tests.

The API suite forces the offline NumPy index, so these tests exercise the
ChromaDB backends directly to keep the persistence paths covered.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from backend.services.vector_store import (
    ChromaVectorStore,
    LangChainChromaVectorStore,
    NumpyVectorStore,
)

pytest.importorskip("chromadb", reason="chromadb is an optional dependency")

VECTOR = [0.11, 0.23, 0.37] * 128  # 384 dimensions
OTHER = [0.31, 0.17, 0.41] * 128

CHROMA_FACTORIES = pytest.mark.parametrize(
    "factory",
    [ChromaVectorStore, LangChainChromaVectorStore],
    ids=["chroma", "chroma-langchain"],
)


def _write(store: object, ids: list[str], vectors: list[list[float]], texts: list[str]) -> int:
    return store.add(  # type: ignore[attr-defined]
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=[{"document_id": f"doc_{i}", "filename": f"{i}.txt"} for i in ids],
    )


@CHROMA_FACTORIES
class TestChromaBackends:
    """Both Chroma wrappers must behave identically from the caller's side."""

    @pytest.fixture()
    def store(self, tmp_path: Path, factory: type):
        instance = factory(tmp_path / "chroma", "rag_test_collection")
        yield instance
        instance.close()

    def test_round_trips_chunks(self, store: object) -> None:
        assert _write(store, ["a", "b"], [VECTOR, OTHER], ["alpha", "beta"]) == 2
        assert store.count() == 2  # type: ignore[attr-defined]
        assert store.count_by_document() == {  # type: ignore[attr-defined]
            "doc_a": 1,
            "doc_b": 1,
        }

        hits = store.query(VECTOR, top_k=1)  # type: ignore[attr-defined]
        assert [chunk.text for chunk in hits] == ["alpha"]
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)

    def test_reset_empties_the_index(self, store: object) -> None:
        _write(store, ["a", "b"], [VECTOR, OTHER], ["alpha", "beta"])
        store.reset()  # type: ignore[attr-defined]
        assert store.count() == 0  # type: ignore[attr-defined]
        assert store.query(VECTOR, top_k=5) == []  # type: ignore[attr-defined]

    def test_survives_writing_after_its_directory_is_deleted(
        self, store: object, tmp_path: Path
    ) -> None:
        """Regression: a live Chroma client keeps SQLite open.

        ``POST /api/system/reset`` deletes ``VECTOR_STORE_DIR`` while the client
        is still around, which made every later write fail with
        ``attempt to write a readonly database`` and broke rebuild.
        """
        _write(store, ["a"], [VECTOR], ["alpha"])
        store.close()  # type: ignore[attr-defined]
        shutil.rmtree(tmp_path / "chroma", ignore_errors=True)
        (tmp_path / "chroma").mkdir(parents=True, exist_ok=True)

        store._reopen()  # type: ignore[attr-defined]  # what the factory does next
        assert _write(store, ["b"], [VECTOR], ["beta"]) == 1
        assert store.count_by_document() == {"doc_b": 1}  # type: ignore[attr-defined]

    def test_reset_then_delete_then_rewrite(self, store: object, tmp_path: Path) -> None:
        """The exact system-reset sequence: reset, rmtree, reopen, re-index."""
        _write(store, ["a", "b"], [VECTOR, OTHER], ["alpha", "beta"])
        store.reset()  # type: ignore[attr-defined]
        store.close()  # type: ignore[attr-defined]
        shutil.rmtree(tmp_path / "chroma", ignore_errors=True)
        (tmp_path / "chroma").mkdir(parents=True, exist_ok=True)

        store._reopen()  # type: ignore[attr-defined]
        assert _write(store, ["c"], [VECTOR], ["gamma"]) == 1
        assert store.status()["status"] == "ready"  # type: ignore[attr-defined]
        assert store.count() == 1  # type: ignore[attr-defined]

    def test_close_is_idempotent(self, store: object) -> None:
        store.close()  # type: ignore[attr-defined]
        store.close()  # type: ignore[attr-defined]


class TestNumpyBackend:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> NumpyVectorStore:
        return NumpyVectorStore(tmp_path / "numpy_store")

    def test_round_trips_and_filters(self, store: NumpyVectorStore) -> None:
        _write(store, ["a", "b"], [VECTOR, OTHER], ["alpha", "beta"])
        assert store.query(VECTOR, top_k=5, document_id="doc_a")[0].text == "alpha"
        assert store.query(VECTOR, top_k=5, document_id="doc_missing") == []

    def test_reset_clears_the_index(self, store: NumpyVectorStore) -> None:
        _write(store, ["a"], [VECTOR], ["alpha"])
        store.reset()
        assert store.count() == 0
        assert store.status()["status"] == "empty"
