#!/usr/bin/env python
"""Generate sample documents for the lab demo.

Creates one PDF (multi-page, so page-level citations are visible), one DOCX and
one TXT file in ``samples/``. The PDF is written with raw PDF syntax to keep the
script dependency-free.

Usage::

    python scripts/generate_samples.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "samples"

# --------------------------------------------------------------------------
# Shared content - a plausible NLP lab report on Retrieval Augmented Generation
# --------------------------------------------------------------------------
TITLE = "Retrieval Augmented Generation for Document Question Answering"

SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Abstract",
        [
            "This report documents a laboratory experiment that implements a "
            "Retrieval Augmented Generation (RAG) system. The system answers "
            "natural language questions using only the documents a user uploads, "
            "and every answer is accompanied by a citation pointing back to the "
            "exact passage it was generated from.",
            "The experiment covers the full pipeline: text extraction, cleaning, "
            "chunking, embedding generation, vector storage, similarity "
            "retrieval and grounded answer generation.",
        ],
    ),
    (
        "1. Objective",
        [
            "The objective of the experiment is to build a question answering "
            "system that can answer questions about a personal document "
            "collection without any prior fine-tuning of a language model.",
            "A secondary objective is to verify that answers are grounded: the "
            "system must refuse to answer when the retrieved context does not "
            "support an answer, rather than hallucinating one.",
            "The objective of the experiment is therefore twofold: demonstrate "
            "the mechanics of retrieval augmented generation, and demonstrate "
            "that its outputs are traceable to their sources.",
        ],
    ),
    (
        "2. System Architecture",
        [
            "The architecture has six stages. First the uploaded file is parsed "
            "into plain text. The text is then cleaned to remove repeated page "
            "headers, fix hyphenation and normalise whitespace.",
            "The cleaned text is split into chunks of 500 characters with an "
            "overlap of 100 characters. The overlap exists so that a sentence "
            "straddling a chunk boundary is still fully contained in one chunk.",
            "Each chunk is encoded into a 384 dimensional vector. The vectors are "
            "stored in a ChromaDB collection configured for cosine similarity.",
        ],
    ),
    (
        "3. Embeddings",
        [
            "The embedding model is sentence-transformers/all-MiniLM-L6-v2. It "
            "is a small bi-encoder that maps a chunk of text into a dense vector "
            "of 384 dimensions, and it is chosen because it is fast on CPU while "
            "still capturing semantic similarity well enough for passage "
            "retrieval.",
            "The same model embeds the user's question, which makes query and "
            "document vectors directly comparable with a cosine similarity.",
        ],
    ),
    (
        "4. Chunking Strategy",
        [
            "The chunking strategy uses 500 character chunks with an overlap of "
            "100 characters. A chunk size of 500 characters keeps enough context "
            "for a meaningful passage while staying small enough that the "
            "embedding is not diluted by unrelated text.",
            "The overlap of 100 characters is roughly one fifth of a chunk. This "
            "prevents a fact that falls exactly on a boundary from being split "
            "across two chunks, where neither half would retrieve it well.",
        ],
    ),
    (
        "5. Vector Database",
        [
            "The vector database used for storage is ChromaDB. It persists an "
            "HNSW index on disk, which makes cosine similarity search fast even "
            "as the collection grows, and it keeps the collection metadata "
            "alongside each vector so that citations can be resolved.",
            "Each vector carries metadata recording the source file name, the "
            "chunk number and the page number. This is what makes it possible to "
            "show the user exactly which passage produced an answer.",
        ],
    ),
    (
        "6. Retrieval",
        [
            "Retrieval works by embedding the question with the same model used "
            "for the documents, then performing a similarity search that returns "
            "the top 5 chunks ranked by cosine similarity.",
            "A similarity score close to 1 means the chunk means almost the same "
            "thing as the question, while a score near 0 means the two are "
            "unrelated. The interface displays these scores so retrieval quality "
            "can be inspected directly.",
        ],
    ),
    (
        "7. Answer Generation",
        [
            "Answer generation sends the retrieved context and the question to a "
            "large language model. The prompt instructs the model to answer only "
            "from the provided context and to reply that the documents do not "
            "contain enough information when the context is insufficient.",
            "Because generation is conditioned on retrieved text, the system "
            "cannot invent facts about documents it has never seen. The cost of "
            "this design is that answer quality is bounded by retrieval quality.",
        ],
    ),
    (
        "8. Advantages of RAG",
        [
            "The advantages of RAG include grounded answers that can be traced "
            "to a source, knowledge that can be updated by editing documents "
            "rather than retraining a model, and transparency because the "
            "retrieved passages can be shown to the user.",
            "RAG also reduces hallucination compared with asking a model "
            "directly, since the model is instructed to rely on the supplied "
            "context.",
        ],
    ),
    (
        "9. Limitations",
        [
            "A limitation of the system is that retrieval quality depends heavily "
            "on the chunk size: chunks that are too small lose context, while "
            "chunks that are too large dilute the signal and reduce precision.",
            "Another limitation is that the system performs lexical and semantic "
            "matching only. It cannot answer questions that require arithmetic "
            "over the documents, or reasoning that spans many chunks at once.",
            "Finally, scanned PDFs without a text layer cannot be processed "
            "because optical character recognition is out of scope for this "
            "experiment.",
        ],
    ),
    (
        "10. Conclusion",
        [
            "The experiment demonstrates that a complete retrieval augmented "
            "generation workflow can be built from standard open source "
            "components and run locally on a laptop CPU.",
            "The key takeaways are that grounding depends on retrieval, that "
            "citations make the system auditable, and that abstaining is a "
            "feature rather than a failure when the corpus does not cover the "
            "question.",
        ],
    ),
]


def _content_lines() -> list[tuple[str, list[str]]]:
    return SECTIONS


def _wrap(text: str, width: int = 92) -> list[str]:
    return textwrap.wrap(text, width=width)


def build_txt() -> Path:
    """Plain-text version of the report."""
    path = SAMPLES_DIR / "rag_lab_report.txt"
    lines: list[str] = [TITLE, "=" * len(TITLE), ""]
    for heading, paragraphs in _content_lines():
        lines.append(heading)
        lines.append("-" * len(heading))
        for paragraph in paragraphs:
            lines.extend(_wrap(paragraph))
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", "utf-8")
    return path


def build_docx() -> Path:
    """DOCX version, including a table so table extraction is exercised."""
    from docx import Document

    path = SAMPLES_DIR / "rag_lecture_notes.docx"
    document = Document()
    document.add_heading("Lecture Notes: RAG Pipelines", level=1)
    document.add_paragraph(
        "These notes accompany the lab on retrieval augmented generation."
    )

    document.add_heading("Pipeline stages", level=2)
    for heading, paragraphs in _content_lines()[1:5]:
        document.add_heading(heading, level=3)
        document.add_paragraph(paragraphs[0])

    document.add_heading("Configuration summary", level=2)
    table = document.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    header = table.rows[0].cells
    header[0].text = "Parameter"
    header[1].text = "Value"
    for name, value in (
        ("Chunk size", "500 characters"),
        ("Chunk overlap", "100 characters"),
        ("Embedding model", "sentence-transformers/all-MiniLM-L6-v2"),
        ("Embedding dimension", "384"),
        ("Vector database", "ChromaDB (cosine)"),
        ("Top-K retrieved chunks", "5"),
    ):
        row = table.add_row().cells
        row[0].text = name
        row[1].text = value

    document.add_paragraph(
        "The advantage of a 100 character overlap is that a fact on a chunk "
        "boundary still appears whole inside at least one chunk."
    )
    document.save(str(path))
    return path


# --------------------------------------------------------------- PDF writer ---
def _pdf_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(pages_per_section: int = 1) -> Path:
    """Write a multi-page PDF using raw PDF syntax (no extra dependencies)."""
    path = SAMPLES_DIR / "rag_lab_report.pdf"

    # Flow the report into fixed-size pages. Each page records the text origin
    # so every content stream is positioned independently.
    pages: list[dict] = []
    lines: list[tuple[int, str]] = []
    y = 740
    top = 740

    def start_page() -> None:
        nonlocal lines, y
        if lines:
            pages.append({"lines": lines, "top": top})
        lines = []
        y = top

    def emit(size: int, text: str, leading: int = 15) -> None:
        nonlocal y
        if y < 60:
            start_page()
        lines.append((size, text))
        y -= leading

    start_page()
    emit(18, TITLE, 26)
    emit(10, "NLP Lab Experiment Report", 20)
    for heading, paragraphs in _content_lines():
        emit(1, "")
        emit(14, heading, 20)
        for paragraph in paragraphs:
            for line in _wrap(paragraph, 88):
                emit(11, line)
        emit(1, "")
    if lines:
        pages.append({"lines": lines, "top": top})

    # --- assemble the PDF ---------------------------------------------------
    objects: list[bytes] = []

    def add_object(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    # Object 1 = catalog, 2 = pages, 3 = font, then per page: page + content.
    catalog_id = 1
    pages_id = 2
    font_id = 3
    objects.extend([b"", b"", b""])
    page_ids: list[tuple[int, int]] = []

    for page in pages:
        stream_parts = [
            "BT",
            f"1 0 0 1 54 {page['top']} Tm",
            "/F1 11 Tf",
            "14 TL",
        ]
        for size, text in page["lines"]:
            if size == 1 and not text:
                stream_parts.append("T*")
                continue
            stream_parts.append(f"/F1 {size} Tf")
            stream_parts.append(f"({_pdf_escape(text)}) Tj")
            stream_parts.append("T*")
        stream_parts.append("ET")
        stream = "\n".join(stream_parts).encode("latin-1", "replace")

        content_id = add_object(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_id = add_object(
            b"<< /Type /Page /Parent "
            + str(pages_id).encode()
            + b" 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 "
            + str(font_id).encode()
            + b" 0 R >> >> /Contents "
            + str(content_id).encode()
            + b" 0 R >>"
        )
        page_ids.append((page_id, content_id))

    objects[catalog_id - 1] = b"<< /Type /Catalog /Pages " + str(pages_id).encode() + b" 0 R >>"
    kids = b" ".join(f"{pid} 0 R".encode() for pid, _ in page_ids)
    objects[pages_id - 1] = (
        b"<< /Type /Pages /Kids ["
        + kids
        + b"] /Count "
        + str(len(page_ids)).encode()
        + b" >>"
    )
    objects[font_id - 1] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    path.write_bytes(bytes(out))
    return path


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    created = [build_txt(), build_docx(), build_pdf()]
    for path in created:
        print(f"  {path.relative_to(REPO_ROOT)}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
