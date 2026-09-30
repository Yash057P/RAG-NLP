"""End-to-end smoke test against the real production stack.

Uses the default configuration - sentence-transformers/all-MiniLM-L6-v2
embeddings, LangChain's ChromaDB integration and the extractive baseline - to
prove the full upload -> ask -> cite flow works outside the test doubles.

    python scripts/verify_stack.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import backend.core.config as config  # noqa: E402


def use_throwaway_storage() -> None:
    """Keep this script out of the real uploads/ and vector_store/."""
    target = REPO_ROOT / ".verify_tmp"
    shutil.rmtree(target, ignore_errors=True)
    config.settings.DATA_DIR = target
    config.settings.UPLOAD_DIR = target / "uploads"
    config.settings.VECTOR_STORE_DIR = target / "vector_store"
    config.settings.REGISTRY_PATH = target / "registry.json"
    config.settings.CHAT_HISTORY_PATH = target / "chat_history.json"
    config.settings.EMBEDDING_PROVIDER = "auto"
    config.settings.VECTOR_STORE_BACKEND = "auto"
    config.settings.LLM_PROVIDER = "extractive"  # no API key needed
    config.settings.ensure_directories()


QUESTIONS = [
    "What is the objective of the experiment?",
    "What are the advantages of RAG?",
    "Which embedding model is used and why?",
    "What are the limitations of this system?",
    "How does the chunking strategy work?",
    "Who won the 1998 FIFA World Cup final?",
]


def main() -> int:
    use_throwaway_storage()

    from fastapi.testclient import TestClient

    from backend.main import app

    samples = sorted((REPO_ROOT / "samples").glob("*"))
    if not samples:
        print("No sample documents found. Run scripts/generate_samples.py first.")
        return 1

    with TestClient(app) as client:
        started = time.perf_counter()
        status = client.get("/api/system/status").json()
        print(f"Startup (incl. model load): {time.perf_counter() - started:.1f}s")
        print(f"  embeddings : {status['embeddings']['detail']}")
        print(f"  vector db  : {status['vector_store']['detail']}")
        print(f"  llm        : {status['llm']['detail']}")
        print(f"  overall    : {status['status']}\n")

        files = [
            ("files", (path.name, path.read_bytes(), "application/octet-stream"))
            for path in samples
        ]
        started = time.perf_counter()
        response = client.post("/api/documents", files=files)
        upload_ms = (time.perf_counter() - started) * 1000
        body = response.json()
        print(f"Upload of {len(files)} files: {upload_ms:.0f}ms -> {body['message']}")
        for record in body["created"]:
            pages = record["page_count"] if record["page_count"] is not None else "-"
            print(
                f"  + {record['filename']:24s} {record['size_display']:>9s} "
                f"{record['chunk_count']:>3d} chunks  "
                f"{str(pages):>3s} pages  {record['embedding_model']}"
            )
        for failure in body["failed"]:
            print(f"  ! {failure['filename']}: {failure['reason']}")
        if body["created"]:
            print("\n  Pipeline stages for the first document:")
            for stage in body["created"][0]["stages"]:
                duration = f"{stage['duration_ms']:8.1f}ms" if stage["duration_ms"] is not None else "     n/a"
                print(f"    {stage['key']:12s} {stage['status']:8s} {duration}  {stage['detail'] or ''}")

        stats = client.get("/api/stats").json()
        print(
            f"\nStats: {stats['total_documents']} docs, {stats['total_chunks']} chunks, "
            f"backend={stats['vector_store_backend']}, dim={stats['embedding_dimension']}"
        )

        print("\n--- Question answering ---")
        for question in QUESTIONS:
            result = client.post("/api/chat/ask", json={"question": question}).json()
            verdict = "GROUNDED " if result["grounded"] else "ABSTAINED"
            print(
                f"\nQ: {question}\n"
                f"   [{verdict}] confidence={result['confidence']:.2f} "
                f"({result['confidence_label']}) in {result['response_time_ms']:.0f}ms"
            )
            print(f"   top_score={result['retrieval']['top_score']} "
                  f"chunks={len(result['retrieved_chunks'])} "
                  f"coverage={result['retrieval']['grounding_coverage']}")
            print(f"   A: {result['answer'][:220]}")
            for source in result["sources"][:2]:
                print(
                    f"     - {source['filename']} chunk {source['chunk_index']}"
                    f"{f' p.{source['page']}' if source['page'] else ''} "
                    f"score={source['score']}"
                )

        print("\n--- Retrieved chunk scores for the first question ---")
        result = client.post("/api/chat/ask", json={"question": QUESTIONS[0]}).json()
        for chunk in result["retrieved_chunks"]:
            print(
                f"  {chunk['score']:.4f}  {chunk['filename']} "
                f"chunk#{chunk['chunk_index']}"
                f"{f' p.{chunk['page']}' if chunk['page'] else ''}: "
                f"{chunk['snippet'][:80]}..."
            )

        history = client.get("/api/chat/history").json()
        print(f"\nHistory: {history['total']} entries")
        final = client.get("/api/stats").json()
        print(
            f"Avg response: {final['average_response_time_ms']:.0f}ms "
            f"(min {final['fastest_response_ms']:.0f} / max {final['slowest_response_ms']:.0f})"
        )

        # Regression guard: /system/reset deletes the Chroma directory while the
        # persistent client is still alive. Without releasing that handle the
        # writes below fail with "attempt to write a readonly database".
        print("\n--- Reset, re-upload, rebuild ---")
        print(f"  {client.post('/api/system/reset').json()['message']}")

        again = client.post("/api/documents", files=files).json()
        for record in again["created"]:
            print(f"  + {record['filename']:24s} {record['chunk_count']:>3d} chunks")
        for failure in again["failed"]:
            print(f"  ! {failure['filename']}: {failure['reason']}")

        rebuild = client.post("/api/documents/rebuild").json()
        print(f"  {rebuild['message']}")

        documents = client.get("/api/documents").json()["documents"]
        failed = [d for d in documents if d["status"] != "ready"]
        for document in documents:
            print(
                f"  {document['status']:6s} {document['filename']:24s} "
                f"{document['chunk_count']:>3d} chunks"
                + (f"  ERROR: {document['error']}" if document["error"] else "")
            )

        expected_docs = len(again["created"])
        expected_chunks = final["total_chunks"]
        if (
            failed
            or len(documents) != expected_docs
            or rebuild["total_chunks"] != expected_chunks
        ):
            print(
                f"  FAIL: expected {expected_docs} ready docs / {expected_chunks} chunks, "
                f"got {len(documents) - len(failed)} ready / {rebuild['total_chunks']} chunks"
            )
            return 1
        print(f"  OK: {len(documents)} documents, {rebuild['total_chunks']} chunks recovered")

        after = client.post("/api/chat/ask", json={"question": QUESTIONS[1]}).json()
        print(
            f"  Re-ask after rebuild: grounded={after['grounded']} "
            f"confidence={after['confidence']:.2f} chunks={len(after['retrieved_chunks'])}"
        )

    shutil.rmtree(REPO_ROOT / ".verify_tmp", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
