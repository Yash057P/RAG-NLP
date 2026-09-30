"""End-to-end API tests against the full upload → ask flow."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from backend.core.config import settings
from backend.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _txt_upload(filename: str, content: str) -> tuple[str, io.BytesIO, str]:
    return (filename, io.BytesIO(content.encode("utf-8")), "text/plain")


SAMPLE_DOC = """
Retrieval Augmented Generation for Question Answering

The objective of the experiment is to build a question answering system that
answers strictly from a set of uploaded documents.

RAG combines a retriever with a large language model. The retriever finds
relevant passages and the generator conditions its answer on those passages.

Advantages of RAG include: grounded answers, up-to-date knowledge without
retraining, and full traceability back to the source document.

The chunking strategy uses 500 character chunks with an overlap of 100
characters so that a sentence spanning a boundary still appears in one chunk.

Embeddings are produced with sentence-transformers/all-MiniLM-L6-v2, a small
bi-encoder that maps text into 384 dimensional vectors. Vectors are stored in
ChromaDB, which persists an HNSW index and answers cosine similarity queries.

A limitation of the system is that retrieval quality depends on the chunk size:
small chunks lose context while large chunks dilute the signal.
"""


class TestSystem:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/api/system/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_status_lists_pipeline_stages(self, client: TestClient) -> None:
        response = client.get("/api/system/status")
        assert response.status_code == 200
        body = response.json()
        assert len(body["pipeline_stages"]) == 8
        assert body["example_questions"]
        assert body["vector_store"]["name"].startswith("Vector store")


class TestDocuments:
    def test_rejects_unsupported_type(self, client: TestClient) -> None:
        response = client.post(
            "/api/documents",
            files=[("files", ("notes.md", io.BytesIO(b"# hi"), "text/markdown"))],
        )
        assert response.status_code == 201
        body = response.json()
        assert body["created"] == []
        assert body["failed"][0]["code"] == "unsupported_file_type"

    def test_upload_and_list(self, client: TestClient) -> None:
        response = client.post(
            "/api/documents", files=[("files", _txt_upload("rag-notes.txt", SAMPLE_DOC))]
        )
        assert response.status_code == 201
        body = response.json()
        assert len(body["created"]) == 1

        record = body["created"][0]
        assert record["status"] == "ready"
        assert record["chunk_count"] > 1
        assert record["extension"] == "txt"
        assert record["size_bytes"] > 0

        stages = {stage["key"] for stage in record["stages"]}
        assert {"ingest", "extraction", "cleaning", "chunking", "embedding", "vector_store"} <= stages

        listing = client.get("/api/documents").json()
        assert listing["total"] >= 1
        assert any(doc["id"] == record["id"] for doc in listing["documents"])

    def test_chunk_preview(self, client: TestClient) -> None:
        document_id = client.get("/api/documents").json()["documents"][0]["id"]
        response = client.get(f"/api/documents/{document_id}/chunks", params={"limit": 3})
        assert response.status_code == 200
        chunks = response.json()
        assert 0 < len(chunks) <= 3
        assert all(chunk["text"].strip() for chunk in chunks)

    def test_unknown_document_is_404(self, client: TestClient) -> None:
        assert client.get("/api/documents/does-not-exist").status_code == 404


class TestChat:
    def test_answers_from_context_with_citations(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat/ask", json={"question": "What is the objective of the experiment?"}
        )
        assert response.status_code == 200
        body = response.json()

        assert body["grounded"] is True
        assert body["abstained"] is False
        assert 0.0 <= body["confidence"] <= 1.0
        assert len(body["retrieved_chunks"]) <= settings.TOP_K
        assert body["retrieval"]["top_score"] > 0
        assert body["sources"], "a grounded answer must cite at least one source"
        assert body["sources"][0]["filename"].endswith(".txt")
        assert body["response_time_ms"] > 0
        assert body["pipeline"]["operation"] == "query"

    def test_abstains_when_corpus_cannot_answer(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat/ask",
            json={"question": "Who won the 1998 FIFA World Cup final in Rio?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is False
        assert "do not contain enough information" in body["answer"]
        assert body["sources"] == []

    def test_blank_question_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/chat/ask", json={"question": "   "}).status_code == 422

    def test_history_records_questions(self, client: TestClient) -> None:
        history = client.get("/api/chat/history").json()
        assert history["total"] >= 2
        assert history["entries"][0]["question"]


class TestStats:
    def test_reports_pipeline_totals(self, client: TestClient) -> None:
        body = client.get("/api/stats").json()
        assert body["total_documents"] >= 1
        assert body["documents_ready"] >= 1
        assert body["total_chunks"] >= 1
        assert body["questions_asked"] >= 2
        assert body["average_response_time_ms"] > 0
        assert body["chunk_size"] == settings.CHUNK_SIZE
        assert body["chunk_overlap"] == settings.CHUNK_OVERLAP
        assert body["embedding_model"]


class TestMaintenance:
    def test_rebuild_is_idempotent(self, client: TestClient) -> None:
        before = client.get("/api/stats").json()["total_chunks"]

        response = client.post("/api/documents/rebuild")
        assert response.status_code == 200
        body = response.json()
        assert body["documents_processed"] >= 1
        assert body["total_chunks"] == before

    def test_rebuild_preserves_filenames_and_ids(self, client: TestClient) -> None:
        """Uploads are stored as "{id}_{name}" - the rebuild must undo that."""
        name, data, mime = _txt_upload("lab report.txt", SAMPLE_DOC)
        upload = client.post("/api/documents", files=[("files", (name, data, mime))])
        assert upload.status_code == 201
        target = upload.json()["created"][0]
        # safe_filename() normalises the space, and that name must survive a
        # rebuild instead of turning into "doc_ab12…lab_report.txt".
        assert target["filename"] == "lab_report.txt"

        assert client.post("/api/documents/rebuild").status_code == 200

        documents = client.get("/api/documents").json()["documents"]
        by_id = {doc["id"]: doc for doc in documents}
        assert target["id"] in by_id
        assert by_id[target["id"]]["filename"] == "lab_report.txt"
        assert all(not doc["filename"].startswith("doc_") for doc in documents)

    def test_delete_removes_chunks(self, client: TestClient) -> None:
        documents = client.get("/api/documents").json()["documents"]
        target = documents[0]
        before = client.get("/api/stats").json()["total_chunks"]

        response = client.delete(f"/api/documents/{target['id']}")
        assert response.status_code == 200
        assert response.json()["chunks_removed"] == target["chunk_count"]

        after = client.get("/api/stats").json()["total_chunks"]
        assert after == before - target["chunk_count"]

    def test_deleted_document_stays_deleted_after_rebuild(self, client: TestClient) -> None:
        """The upload file must be removed too, or a rebuild resurrects it."""
        name, data, mime = _txt_upload("resurrect_me.txt", SAMPLE_DOC)
        upload = client.post("/api/documents", files=[("files", (name, data, mime))])
        assert upload.status_code == 201
        target = upload.json()["created"][0]

        assert client.delete(f"/api/documents/{target['id']}").status_code == 200
        assert not (settings.UPLOAD_DIR / target["stored_filename"]).exists()

        client.post("/api/documents/rebuild")

        remaining = client.get("/api/documents").json()["documents"]
        assert target["id"] not in {doc["id"] for doc in remaining}
        assert "resurrect_me.txt" not in {doc["filename"] for doc in remaining}
