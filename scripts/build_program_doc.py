#!/usr/bin/env python
"""Build the Program section of the lab report.

The report documents the RAG loop, so the document holds one snippet per stage
of that loop and nothing else - no project tree, no settings dump, no
screenshots. Every snippet is pulled from the source tree by name, never by line
range, so the report cannot silently drift from the code it claims to describe.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "RAG-NLP_Program.docx"

CODE_FONT = "Consolas"
CODE_SIZE = Pt(8.5)


# ------------------------------------------------------------------ helpers ---
def definition(relative: str, name: str) -> str:
    """Return the source of a function, method or class, located by name.

    Hard-coded line ranges rot the moment a function above grows by one line, and
    a stale range silently pastes the wrong code into the report. Parsing the
    file instead means the document follows the code it documents.

    ``name`` may be dotted to reach a method, e.g. ``"ExtractiveLLM.generate"``.
    """
    path = REPO_ROOT / relative
    lines = path.read_text("utf-8").splitlines()

    for node in ast.parse(path.read_text("utf-8")).body:
        parts = name.split(".")
        if getattr(node, "name", None) != parts[0]:
            continue
        for part in parts[1:]:
            node = next(
                (c for c in node.body if getattr(c, "name", None) == part), None
            )
            if node is None:
                break
        if node is not None:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise LookupError(f"{name} not found in {relative}")


def between(relative: str, start_anchor: str, stop_anchor: str) -> str:
    """Return the lines from ``start_anchor`` up to, excluding, ``stop_anchor``.

    For blocks the AST cannot delimit: a module-level template, or the head of a
    long function up to the point where the report's topic ends.
    """
    lines = (REPO_ROOT / relative).read_text("utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if start_anchor in line), None)
    stop = next(
        (i for i, line in enumerate(lines) if i > (start or 0) and stop_anchor in line),
        None,
    )
    if start is None or stop is None:
        raise LookupError(f"anchors not found in {relative}: {start_anchor!r}")
    return "\n".join(lines[start:stop]).rstrip()


def _force_font(run, name: str) -> None:
    """Word needs the east-asian font pinned too or it substitutes."""
    element = run._element.rPr.rFonts
    element.set(qn("w:eastAsia"), name)


def _shade(cell, hex_color: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), hex_color)
    properties.append(shading)


def heading(document: Document, text: str, level: int) -> None:
    paragraph = document.add_heading(text, level=level)
    paragraph.paragraph_format.space_before = Pt(14 if level <= 1 else 10)
    paragraph.paragraph_format.space_after = Pt(6)


def body(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.font.size = Pt(10)
    paragraph.paragraph_format.space_after = Pt(6)


def caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    paragraph.paragraph_format.space_after = Pt(3)


def add_code(document: Document, source: str) -> None:
    table = document.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    table.autofit = True

    cell = table.cell(0, 0)
    _shade(cell, "F4F5F7")

    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.0

    for index, line in enumerate(source.split("\n")):
        if index:
            paragraph.add_run().add_break()
        run = paragraph.add_run(line)
        run.font.name = CODE_FONT
        run.font.size = CODE_SIZE
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
        _force_font(run, CODE_FONT)

    document.add_paragraph()


# ------------------------------------------------------------------- stages ---
def snippet_chunking(document: Document) -> None:
    heading(document, "1. Chunking", 1)
    body(
        document,
        "The cleaned text is split into 500-character chunks with 100 characters of "
        "overlap, so a sentence split across a boundary is still retrievable from "
        "one side of it. Each chunk records the character offset it occupies, which "
        "is how a citation can name the page a chunk came from.",
    )
    caption(document, "backend/services/document_processor.py - chunk_text()")
    add_code(document, definition("backend/services/document_processor.py", "chunk_text"))


def snippet_embedding(document: Document) -> None:
    heading(document, "2. Embedding", 1)
    body(
        document,
        "Chunks are embedded with all-MiniLM-L6-v2 into 384-dimensional vectors, "
        "L2-normalised so that a dot product is the cosine similarity. The same "
        "model embeds the question at query time, which is what puts documents and "
        "questions in one comparable space. A document's filename is prepended to "
        "the text sent to the embedder, so a question naming the file can find it; "
        "the stored text stays verbatim, so citations still show real content.",
    )
    caption(document, "backend/services/embedding_service.py")
    add_code(
        document,
        definition(
            "backend/services/embedding_service.py",
            "SentenceTransformerEmbedder.embed_documents",
        )
        + "\n\n"
        + definition(
            "backend/services/embedding_service.py",
            "SentenceTransformerEmbedder.embed_query",
        ),
    )


def snippet_store(document: Document) -> None:
    heading(document, "3. Vector store", 1)
    body(
        document,
        "Vectors are written to a persistent ChromaDB collection configured for "
        "cosine distance. The query goes straight to the collection rather than "
        "through LangChain's similarity-search helper, because the question has "
        "already been embedded by the retrieval stage and the helper would embed it "
        "a second time. Chroma returns a distance, which on a cosine index equals "
        "1 - similarity, so the score is converted back.",
    )
    caption(document, "backend/services/vector_store.py - LangChainChromaVectorStore.query()")
    add_code(
        document,
        definition("backend/services/vector_store.py", "LangChainChromaVectorStore.query"),
    )


def snippet_retrieval(document: Document) -> None:
    heading(document, "4. Retrieval (top 5)", 1)
    body(
        document,
        "The question is embedded and searched against the collection. Chunks "
        "weaker than MIN_SIMILARITY are dropped, so a question with nothing relevant "
        "returns an empty result set rather than the five least-bad chunks. The "
        "store is asked for more candidates than the final count because a demoted "
        "bibliography chunk should let the prose ranked just below the cut take its "
        "place.",
    )
    caption(document, "backend/services/rag_service.py - RAGService.retrieve()")
    add_code(document, definition("backend/services/rag_service.py", "RAGService.retrieve"))


def snippet_gate(document: Document) -> None:
    heading(document, "5. Grounding gate", 1)
    body(
        document,
        "This is the requirement the rest of the system exists to satisfy: the "
        "answer must come from the retrieved context. The question is not simply "
        "handed to the model and trusted. Term coverage - the fraction of the "
        "question's meaningful words that appear in the context the model is shown, "
        "source labels included - is measured first, and below the threshold the "
        "model is never called at all. An unanswerable question therefore costs zero "
        "LLM tokens and cannot produce a hallucination.",
    )
    caption(document, "backend/services/rag_service.py - RAGService.answer()")
    add_code(
        document,
        between(
            "backend/services/rag_service.py",
            "    def answer(",
            "    response = AnswerResponse(",
        ),
    )


def snippet_prompt(document: Document) -> None:
    heading(document, "6. Prompt template", 1)
    body(
        document,
        "The retrieved chunks are rendered as a numbered context block and the model "
        "is instructed to answer only from it, replying with a fixed sentence when "
        "the context does not contain the answer. The template is used exactly as "
        "specified.",
    )
    caption(document, "backend/prompts/templates.py")
    add_code(
        document,
        between(
            "backend/prompts/templates.py",
            "CONTEXT_BLOCK_TEMPLATE",
            "# Emitted when retrieval returns nothing at all.",
        ),
    )


STAGES = (
    snippet_chunking,
    snippet_embedding,
    snippet_store,
    snippet_retrieval,
    snippet_gate,
    snippet_prompt,
)


# -------------------------------------------------------------------- build ---
def build() -> Path:
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)

    for section in document.sections:
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)

    title = document.add_heading("Program", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(
        "RAG Question Answering System - Retrieval Augmented Generation Pipeline\n"
        "FastAPI + LangChain + ChromaDB  |  React + TypeScript dashboard"
    )
    run.font.size = Pt(11)
    run.bold = True

    note = document.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = note.add_run(
        "The implementation of each stage of the RAG loop, extracted verbatim "
        "from the project source tree."
    )
    run.font.size = Pt(9)
    run.italic = True
    run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    for stage in STAGES:
        stage(document)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size:,} bytes)")
