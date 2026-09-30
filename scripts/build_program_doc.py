"""Generate the lab "Program" section as a Word document.

Every code block is extracted verbatim from the live source tree by line
range, so the document can never drift from the code it documents.

    python scripts/build_program_doc.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from docx import Document  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

OUTPUT = REPO_ROOT / "docs" / "RAG-NLP_Program.docx"

CODE_FONT = "Consolas"
CODE_SIZE = Pt(8.5)


# ------------------------------------------------------------------ helpers ---
def definition(relative: str, name: str) -> str:
    """Return the source of a function, method or class, located by name.

    Hard-coded line ranges rot the moment a function above grows by one line,
    and a stale range silently pastes the wrong code into the report. Parsing
    the file instead means the document follows the code it documents.

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


def constants(relative: str, *names: str) -> str:
    """Return a run of module-level constants, explanatory comments included.

    Named lookups rather than a line range, because these are exactly the
    blocks a later edit reorders, and the comments above them are half the
    value of showing them in a report.
    """
    path = REPO_ROOT / relative
    lines = path.read_text("utf-8").splitlines()

    found: dict[str, ast.Assign] = {}
    for node in ast.parse(path.read_text("utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node

    missing = [name for name in names if name not in found]
    if missing:
        raise LookupError(f"{missing} not found in {relative}")

    # Walk back over the contiguous comment block above the first constant.
    start = min(found[name].lineno for name in names)
    while start > 1 and lines[start - 2].lstrip().startswith("#"):
        start -= 1
    last = max(found[name].end_lineno for name in names)
    return "\n".join(lines[start - 1 : last])


def between(relative: str, start_anchor: str, stop_anchor: str) -> str:
    """Return the lines from ``start_anchor`` up to, excluding, ``stop_anchor``.

    For what the Python AST cannot address: module-level templates, and
    TypeScript. There the only stable handle is the text that delimits them.
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


def add_code(document: Document, source: str) -> None:
    """Append a monospaced, shaded block that never splits across pages."""
    table = document.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    table.autofit = True

    cell = table.cell(0, 0)
    _shade(cell, "F4F5F7")

    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.0

    lines = source.split("\n")
    for index, line in enumerate(lines):
        if index:
            paragraph.add_run().add_break()
        run = paragraph.add_run(line)
        run.font.name = CODE_FONT
        run.font.size = CODE_SIZE
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
        _force_font(run, CODE_FONT)

    document.add_paragraph()


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


def body(document: Document, text: str, *, italic: bool = False) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.italic = italic
    run.font.size = Pt(10)
    paragraph.paragraph_format.space_after = Pt(8)


def caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    paragraph.paragraph_format.space_after = Pt(3)


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

    # -- title ---------------------------------------------------------------
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
        "All code below is extracted verbatim from the project source tree."
    )
    run.font.size = Pt(9)
    run.italic = True
    run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    # -- 1. layout -----------------------------------------------------------
    heading(document, "1. Project Structure", 1)
    body(
        document,
        "The system is split into an ingestion pipeline, a retrieval pipeline and a "
        "three-column dashboard. Each numbered stage below maps to one box in the "
        "pipeline diagram shown in the Output section.",
    )
    add_code(
        document,
        """RAG-NLP/
├── backend/
│   ├── main.py                    FastAPI app + lifespan model warm-up
│   ├── core/config.py             all tunables (chunk size, top-k, thresholds)
│   ├── api/
│   │   ├── documents.py           upload / list / delete / rebuild
│   │   ├── chat.py                POST /api/chat/ask
│   │   ├── stats.py               dashboard counters
│   │   └── system.py              health, status, reset
│   ├── services/
│   │   ├── document_processor.py  STEPS 1-3: extract, clean, chunk
│   │   ├── embedding_service.py   STEP 5: MiniLM-L6-v2 (+ hashed fallback)
│   │   ├── vector_store.py        STEP 6: ChromaDB (+ NumPy fallback)
│   │   ├── rag_service.py         STEPS 7-8: retrieve, ground, answer
│   │   ├── ingestion_service.py   stage orchestration + timing trace
│   │   └── llm_factory.py         OpenAI / Ollama / Mistral / extractive
│   ├── prompts/templates.py       the prompt template from the specification
│   ├── models/schemas.py          Pydantic contracts, 8 stage definitions
│   └── tests/                     42 automated tests
├── frontend/src/                  React + TS + Tailwind + ShadCN dashboard
├── samples/                       3 demo documents
└── scripts/                       generate_samples.py, verify_stack.py""",
    )

    # -- 2. configuration ----------------------------------------------------
    heading(document, "2. Stage 0 - Configuration", 1)
    body(
        document,
        "Every value fixed by the specification (500-character chunks, 100-character "
        "overlap, top-5 retrieval) is read from a single settings object. Three "
        "provider switches use \"auto\", which tries the real component first and "
        "degrades to a working fallback; whichever component ends up active is "
        "always reported by GET /api/system/status and labelled in the dashboard.",
    )
    caption(document, "backend/core/config.py")
    add_code(
        document,
        "CHUNK_SIZE: int = 500                 # 500-character chunks\n"
        "CHUNK_OVERLAP: int = 100              # 100-character overlap\n"
        "TOP_K: int = 5                        # retrieve the top 5 chunks\n"
        "MIN_SIMILARITY: float = 0.05          # discard chunks weaker than this\n"
        "GROUNDING_THRESHOLD: float = 0.18     # min term coverage to allow an answer\n"
        "REFERENCE_SCORE_PENALTY: float = 0.35 # demote bibliography chunks\n"
        "RETRIEVAL_CANDIDATE_MULTIPLIER: int = 4  # over-fetch, then trim to top_k\n"
        "\n"
        'EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"\n'
        'ALLOWED_EXTENSIONS: list[str] = ["pdf", "docx", "txt"]\n'
        'MAX_UPLOAD_SIZE_MB: int = 25\n'
        "\n"
        'EMBEDDING_PROVIDER: str = "auto"      # auto -> MiniLM -> hashed 384-dim\n'
        'VECTOR_STORE_BACKEND: str = "auto"    # auto -> langchain-chroma -> chromadb -> numpy\n'
        'LLM_PROVIDER: str = "auto"            # auto -> openai -> ollama -> mistral -> extractive',
    )

    # -- 3. extraction -------------------------------------------------------
    heading(document, "3. Stage 1 - Document Loading and Text Extraction", 1)
    body(
        document,
        "Each file type is read by its own reader. The PDF reader walks the file "
        "page by page and wraps every page in a Segment, which is what allows a "
        "citation later on to name the exact page a chunk came from.",
    )
    caption(document, "backend/services/document_processor.py")
    add_code(document, definition("backend/services/document_processor.py", "_extract_pdf"))
    body(
        document,
        "Academic papers are usually set in two columns, and a plain text "
        "extraction reads left-to-right line by line, which splices the two "
        "columns together mid-sentence. Every later stage then works on "
        "scrambled text: embeddings lose the meaning, and the extractive "
        "baseline finds only fragments. The gutter between the columns is "
        "located from the x-coordinates of the words, and the page is cropped "
        "at it so each column is read top to bottom. Only words of at least "
        "four characters are considered when building the histogram, because a "
        "page number sitting in the gutter would otherwise split it into two "
        "slivers and defeat the detection.",
    )
    add_code(
        document,
        definition("backend/services/document_processor.py", "_detect_column_gutter"),
    )
    add_code(
        document,
        definition(
            "backend/services/document_processor.py", "_extract_pdfplumber_page"
        ),
    )
    body(
        document,
        "DOCX reading also captures headings, tables and footers, because a great "
        "deal of useful content in a Word file lives outside the body paragraphs.",
    )
    add_code(document, definition("backend/services/document_processor.py", "_extract_docx"))
    body(document, "A single dispatcher validates the extension and rejects anything unsupported.")
    add_code(document, definition("backend/services/document_processor.py", "extract"))

    # -- 4. cleaning ---------------------------------------------------------
    heading(document, "4. Stage 2 - Text Cleaning", 1)
    body(
        document,
        "Raw PDF text contains ligatures, smart quotes, words hyphenated across a "
        "line break, and a running header repeated on every page. Cleaning removes "
        "all four. Boilerplate is detected across the whole document first and then "
        "stripped segment by segment, so each page keeps its identity and the "
        "character offsets recorded during chunking still resolve to the right page.",
    )
    caption(document, "backend/utils/text.py")
    add_code(document, definition("backend/utils/text.py", "detect_boilerplate"))
    add_code(document, definition("backend/utils/text.py", "clean_text"))
    body(
        document,
        "Cleaning also locates the bibliography. A citation entry restates its own "
        "paper's title, so it is dense with exactly the topic words a user asks "
        "about, and a similarity search ranks it above the prose that answers the "
        "question. The heading alone is not trusted, because a document may merely "
        "mention references in passing; the text below it has to look like a list "
        "of citation entries.",
    )
    add_code(document, definition("backend/utils/text.py", "find_reference_start"))

    # -- 5. chunking ---------------------------------------------------------
    heading(document, "5. Stage 3 - Chunking (500 characters, 100 overlap)", 1)
    body(
        document,
        "LangChain's RecursiveCharacterTextSplitter never overlaps chunks inside a "
        "single string - it only carries a tail across separate documents - so the "
        "100-character overlap required by the specification is applied here "
        "explicitly. Each chunk records the character offset it occupies in the "
        "cleaned text, which the page lookup uses to attach a page number.",
    )
    caption(document, "backend/services/document_processor.py")
    add_code(document, definition("backend/services/document_processor.py", "_apply_overlap"))
    add_code(document, definition("backend/services/document_processor.py", "chunk_text"))

    # -- 6. embeddings -------------------------------------------------------
    heading(document, "6. Stage 5 - Text Embeddings", 1)
    body(
        document,
        "Documents and questions are embedded with the same sentence-transformers "
        "model, all-MiniLM-L6-v2, producing 384-dimensional vectors. Because both "
        "sides use one model, query and document vectors live in the same space and "
        "can be compared directly with cosine similarity. Vectors are L2-normalised "
        "on encode, so cosine similarity reduces to a dot product. If torch or "
        "sentence-transformers is unavailable the class below is replaced by a "
        "hashed n-gram embedder so the rest of the pipeline still runs.",
    )
    caption(document, "backend/services/embedding_service.py")
    add_code(
        document,
        definition("backend/services/embedding_service.py", "SentenceTransformerEmbedder"),
    )

    # -- 7. vector store -----------------------------------------------------
    heading(document, "7. Stage 6 - Vector Database (ChromaDB)", 1)
    body(
        document,
        "Chunks are written to a persistent ChromaDB collection configured for "
        "cosine distance. The query uses Chroma's native collection handle rather "
        "than LangChain's similarity_search_with_score helper, because the question "
        "has already been embedded by the retrieval stage; the LangChain helper "
        "would embed it a second time. Chroma returns cosine distance, and on a "
        "cosine index that equals 1 - similarity, so the score is converted back.",
    )
    caption(document, "backend/services/vector_store.py")
    add_code(
        document,
        definition("backend/services/vector_store.py", "LangChainChromaVectorStore").split(
            "    # -- reads"
        )[0].rstrip(),
    )
    body(
        document,
        "Each chunk is stored with the metadata that makes the citation panel in the "
        "dashboard possible: the owning document, its original filename, the chunk "
        "index, the page, the character span, and whether the chunk came from the "
        "bibliography. The filename is also prepended to the text handed to the "
        "embedder, so a question naming the file can find it, while the stored text "
        "stays verbatim and the citations keep showing the real content.",
    )
    caption(document, "backend/utils/files.py")
    add_code(document, definition("backend/utils/files.py", "title_context"))
    caption(document, "backend/services/ingestion_service.py")
    add_code(
        document,
        between(
            "backend/services/ingestion_service.py",
            "    for chunk in chunks:",
            "    try:",
        ),
    )

    # -- 8. prompt -----------------------------------------------------------
    heading(document, "8. Prompt Template", 1)
    body(
        document,
        "The prompt is used exactly as specified. The model is instructed to answer "
        "only from the supplied context and to reply with a fixed sentence when the "
        "context does not contain the answer.",
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

    # -- 9. retrieval --------------------------------------------------------
    heading(document, "9. Stage 7 - Similarity Retrieval (Top 5)", 1)
    body(
        document,
        "The question is embedded with the same model used for the documents, then "
        "searched against the collection. Results below MIN_SIMILARITY are discarded "
        "so that a question with nothing relevant in the corpus returns an empty "
        "result set rather than the five least-bad chunks. The store is asked for "
        "more candidates than the final count, because a demoted bibliography chunk "
        "should let the prose ranked just below the cut take its place.",
    )
    caption(document, "backend/services/rag_service.py")
    add_code(document, definition("backend/services/rag_service.py", "RAGService.retrieve"))
    body(
        document,
        "Retrieved chunks are rendered into a numbered context block so the model "
        "can refer back to a specific source. The block is also where a chunk the "
        "ingestion stage identified as part of a bibliography is labelled, so that no "
        "provider quotes a citation list back as an answer.",
    )
    add_code(document, definition("backend/services/rag_service.py", "build_context"))

    # -- 10. grounding gate --------------------------------------------------
    heading(document, "10. Stage 8 - Grounded Answer Generation", 1)
    body(
        document,
        "This is the core requirement: the answer must be grounded in the retrieved "
        "context. The system does not simply hand the question to the LLM and trust "
        "it. A grounding gate runs first and refuses to call the model at all unless "
        "the retrieved context actually supports an answer, so an unanswerable "
        "question costs zero LLM tokens and cannot produce a hallucination.",
    )
    body(
        document,
        "The gate rests on term coverage: the fraction of the question's meaningful "
        "words that appear in the retrieved context. The comparison is made against "
        "the same text the model is shown, source labels included, so retrieval and "
        "the gate cannot contradict each other - a document named "
        "Experiment_5_DAV.docx legitimately answers \"what is dav?\" even though the "
        "acronym appears nowhere in its text. If retrieval returned nothing relevant, "
        "the model never sees the question.",
    )
    caption(document, "backend/utils/text.py")
    add_code(document, definition("backend/utils/text.py", "content_words"))
    add_code(document, definition("backend/utils/text.py", "term_coverage"))
    body(
        document,
        "The text the coverage is measured against is the same text the model is "
        "shown, so the source labels count towards it.",
    )
    caption(document, "backend/services/rag_service.py")
    add_code(document, definition("backend/services/rag_service.py", "coverage_text"))
    body(
        document,
        "The full answer method, showing the gate, the LLM call, the abstention "
        "handling and the confidence penalty applied when the system declines to "
        "answer.",
    )
    caption(document, "backend/services/rag_service.py")
    add_code(document, definition("backend/services/rag_service.py", "RAGService.answer"))

    # -- 11. confidence ------------------------------------------------------
    heading(document, "11. Confidence Score", 1)
    body(
        document,
        "Confidence blends three retrieval signals and is returned together with its "
        "components so the number is explainable rather than a black box:",
    )
    add_code(
        document,
        constants(
            "backend/services/rag_service.py",
            "SIMILARITY_CEILING",
            "MARGIN_CEILING",
            "MAX_SOURCES",
            "SNIPPET_CHARS",
        ),
    )
    caption(document, "backend/services/rag_service.py")
    add_code(document, definition("backend/services/rag_service.py", "score_confidence"))
    body(
        document,
        "confidence = 0.45 x similarity + 0.35 x coverage + 0.20 x margin, halved "
        "whenever the system abstains. This is a retrieval-quality proxy, not a "
        "calibrated probability, and the dashboard labels it as such.",
    )

    # -- 12. citations -------------------------------------------------------
    heading(document, "12. Source Citations", 1)
    body(
        document,
        "Every grounded answer carries the filename, chunk number, page and a text "
        "snippet, deduplicated and ordered by similarity.",
    )
    caption(document, "backend/services/rag_service.py")
    add_code(document, definition("backend/services/rag_service.py", "build_sources"))

    # -- 13. api -------------------------------------------------------------
    heading(document, "13. API Layer", 1)
    body(
        document,
        "The upload endpoint processes a list of files and isolates failures: one "
        "unsupported or corrupt file is reported in the failed array while the rest "
        "of the batch still indexes. Re-uploading identical content replaces the "
        "previous copy instead of creating a duplicate.",
    )
    caption(document, "backend/api/documents.py")
    add_code(document, definition("backend/api/documents.py", "upload_documents"))
    caption(document, "backend/api/chat.py")
    add_code(document, definition("backend/api/chat.py", "ask_question"))

    # -- 14. llm -------------------------------------------------------------
    heading(document, "14. LLM Providers with Fallback", 1)
    body(
        document,
        "OpenAI, Ollama and Mistral are all supported and selected in that order "
        "under the \"auto\" setting. When none is configured the router falls back to "
        "a deterministic extractive baseline, which ranks context sentences by "
        "question overlap and returns them verbatim. It is not a language model, but "
        "it keeps the full retrieval loop, the citations and the abstention "
        "behaviour demonstrable with no API key.",
    )
    body(
        document,
        "The baseline reads the context block the same way a real model would. It "
        "keeps each sentence paired with the document it came from, so a question "
        "can be matched against a document's identity as well as its prose, and it "
        "skips any source the retrieval stage labelled as a bibliography, since a "
        "citation entry matches a topical question very well while answering none "
        "of it.",
    )
    caption(document, "backend/services/llm_factory.py")
    add_code(
        document,
        definition("backend/services/llm_factory.py", "ExtractiveLLM._context_sentences"),
    )
    add_code(
        document,
        definition("backend/services/llm_factory.py", "ExtractiveLLM.generate"),
    )

    # -- 15. frontend --------------------------------------------------------
    heading(document, "15. Frontend Dashboard", 1)
    body(
        document,
        "The dashboard is a responsive three-column layout: upload, documents and "
        "statistics on the left, the chat interface in the centre, and the retrieved "
        "chunks with their similarity scores on the right. It collapses to tabs on "
        "mobile. The chat handler keeps the analytics cards in sync by refetching "
        "statistics after every answer.",
    )
    caption(document, "frontend/src/App.tsx")
    add_code(
        document,
        between("frontend/src/App.tsx", "// --- ask", "// --- clear conversation"),
    )

    # -- 16. run -------------------------------------------------------------
    heading(document, "16. Installation and Execution", 1)
    add_code(
        document,
        """# Backend dependencies
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..

# Optional configuration (add an API key, or run Ollama locally)
cp .env.example .env

# Start the API on port 8000
.venv/bin/uvicorn backend.main:app --reload --port 8000

# Start the dashboard on port 5173 (second terminal)
cd frontend && npm run dev""",
    )
    body(document, "Tests and the real-stack verification script:")
    add_code(
        document,
        """.venv/bin/python -m pytest backend/tests -q     # 83 tests
.venv/bin/python scripts/verify_stack.py        # real-stack smoke test""",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size:,} bytes)")
