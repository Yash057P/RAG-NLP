# 🧠 RAG-NLP Assistant

A complete **Retrieval-Augmented Generation** question-answering system for an NLP lab:
upload PDF / DOCX / TXT documents, and the backend extracts → cleans → chunks → embeds →
indexes them, then answers questions **strictly from the retrieved context** with source
citations, per-answer confidence scores and a full pipeline inspector.

* **Frontend** — React 18 + TypeScript + TailwindCSS + shadcn/ui (dark-first glassmorphism,
  responsive 3-column dashboard)
* **Backend** — FastAPI + LangChain + ChromaDB + `sentence-transformers/all-MiniLM-L6-v2`
* **LLM** — OpenAI (or any OpenAI-compatible gateway) / Ollama (Llama 3.1) / Mistral,
  selected automatically, with a deterministic offline baseline as the last resort

---

## 📸 What you get

| Panel | Contents |
| --- | --- |
| **Left** | Multi-file drag & drop uploader with progress, document registry (name, size, status, **chunk count**, pages, embedding model), delete / rebuild / reset, and the six analytics cards |
| **Centre** | Chat transcript with markdown answers, a **confidence meter** (hover for the breakdown), inline source citations, one-click **PDF export** and a "view prompt" dialog showing the exact string sent to the model |
| **Right** | The top-K **retrieved chunks** with cosine similarity bars and question-term highlighting, plus the deduplicated **source citation list** |
| **Bonus** | An 8-stage **RAG pipeline visualisation** that lights up with the real status and duration of each stage of the last ingestion or query |

---

## 🚀 Quick start

Two terminals, ~2 minutes.

### 1. Backend

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional - defaults work out of the box

uvicorn backend.main:app --reload  # http://127.0.0.1:8000/docs
```

> The first start downloads `all-MiniLM-L6-v2` (~90 MB) and warms it, so the first request is
> already fast. If the model cannot be downloaded the app transparently falls back to a
> dependency-free hashed embedder and keeps working.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                        # http://localhost:5173
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`, so there is no CORS setup to do.

### 3. Try it

1. Drag the three files from [`samples/`](samples) onto the dropzone (or click **Browse files**).
2. Watch the chunk counts appear, then ask one of the suggested questions, e.g.
   *"Which embedding model is used and why?"*
3. Inspect the retrieved chunks, the similarity scores and the citations on the right.
4. Ask something unrelated (*"Who won the 1998 FIFA World Cup final?"*) to see the model
   **abstain** instead of hallucinating.

---

## ⚙️ Configuration

Everything is read from the environment or a `.env` file at the repository root
(see [`.env.example`](.env.example)). The defaults run without any configuration.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | Chunking strategy (characters) |
| `TOP_K` | `5` | Chunks retrieved per question |
| `MIN_SIMILARITY` | `0.05` | Cosine floor below which a chunk is discarded |
| `GROUNDING_THRESHOLD` | `0.18` | Minimum question-term coverage before the LLM may answer |
| `EMBEDDING_PROVIDER` | `auto` | `huggingface` \| `hashing` \| `auto` |
| `VECTOR_STORE_BACKEND` | `auto` | `chroma` \| `numpy` \| `auto` |
| `LLM_PROVIDER` | `auto` | `openai` \| `ollama` \| `mistral` \| `extractive` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | – / `gpt-4o-mini` | OpenAI or any compatible base URL |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | – / `llama3.1` | Local Ollama |

### Graceful degradation

The system never crashes because a component is missing — it downgrades and the UI always
shows which provider is live:

```
EMBEDDING_PROVIDER=auto  →  sentence-transformers  →  hashed 384-dim embedder
VECTOR_STORE_BACKEND=auto → langchain-chroma       →  chromadb     →  NumPy flat index
LLM_PROVIDER=auto        →  openai                 →  ollama  →  mistral  →  extractive
```

The extractive baseline is not a language model: it ranks the sentences of the retrieved
context against the question, so the whole RAG loop stays demonstrable (and gradable) with no
API key. The answer card is labelled as such.

---

## 🔁 The pipeline

```
 1  ingest       receive PDF / DOCX / TXT (≤ 25 MB, 3 formats)
 2  extraction   pdfplumber → PyPDF2, python-docx (paragraphs + tables), plain text
 3  cleaning     de-hyphenate, drop repeated page headers, normalise whitespace,
                 strip boilerplate, collapse lists — offsets are tracked so page
                 provenance survives
 4  chunking     500 characters, 100 characters of word-aligned overlap
 5  embedding    all-MiniLM-L6-v2 → 384-dim L2-normalised vectors
 6  vector store ChromaDB collection `rag_documents`, cosine metric, persisted to disk
 7  retrieval    embed the query, top-5 cosine search, drop < MIN_SIMILARITY
 8  generation   ground the answer, or abstain when term coverage is too low
```

Stages 1-6 run once per document (ingestion); stages 7-8 run per question (query). Every run
returns a `PipelineTrace` with per-stage status, duration and metrics, which is exactly what
the pipeline diagram in the UI renders.

### Reading real-world documents

Three things a naive extraction gets wrong on actual documents, all fixed and all covered by
tests:

| Problem | What it did | Fix |
| --- | --- | --- |
| Two-column PDFs | `extract_text()` reads left-to-right *per line*, interleaving both columns mid-sentence and scrambling every downstream stage | Detect the empty gutter and crop into it, so each column is read top-to-bottom. Sentences in a real paper went from 77 garbled to 183 coherent |
| Acronyms in filenames | "what is dav?" scored **0%** coverage, because `DAV` appears in `Experiment_5_DAV.docx` but nowhere in the indexed text | Prepend a normalised form of the filename to the text sent to the *embedder* (stored text stays verbatim) and count the filename in the grounding gate |
| Conversation filler | "tell me about mathematics" was scored on 4 terms when it has 1 topic term, pushing real answers below the threshold | Treat request verbs ("tell", "explain", "about", "me") as stopwords |

### Bibliographies are demoted, not deleted

A citation entry restates its paper's own title, so it is lexically dense with exactly the
topic words a user asks about — a similarity search ranks it *above* the prose that answers the
question. `find_reference_start` locates the bibliography structurally (a `References` heading
followed by text that actually looks like citation entries, which measured 0.00 entries/line
for body text against 0.22 for a real bibliography), and those chunks are scaled by
`REFERENCE_SCORE_PENALTY` at retrieval. They stay visible and are labelled `bibliography` in
the UI; the same label is passed to the LLM so no provider quotes one as an answer.

### Grounding gate

Before the LLM is called, the service measures how much of the question is actually covered by
the retrieved context (`term_coverage`). Below `GROUNDING_THRESHOLD` the model is asked to
abstain and the confidence score is halved — this is what keeps the system from inventing
answers.

Coverage is measured against the same text the model is shown, source labels included, so
retrieval and the gate cannot disagree: a document named `Experiment_5_DAV.docx` legitimately
answers "what is dav?", and a question with no bearing on the corpus is still refused.

### Confidence score

```
confidence = 0.45 · similarity + 0.35 · coverage + 0.20 · margin
```

* `similarity` — top cosine score, normalised against a 0.75 ceiling
* `coverage` — fraction of question terms present in the retrieved context
* `margin` — how far the best chunk stands out from the average of the rest

Hover the meter in the answer card to see the three components. The same three inputs are
used for the `High / Moderate / Low` label.

---

## 🗂️ Project structure

```
.
├── backend/
│   ├── main.py                 FastAPI app, CORS, lifespan (warms the embedder)
│   ├── api/                    documents · chat · stats · system routers
│   ├── core/                   config (pydantic-settings) · logging · errors
│   ├── models/schemas.py       Pydantic contracts + the 8 stage definitions
│   ├── prompts/templates.py    the exact prompt template required by the lab
│   ├── services/               document_processor · embedding_service · vector_store
│   │                           llm_factory · pipeline · registry · ingestion_service
│   │                           analytics · rag_service
│   ├── utils/                  files (validation, fingerprints) · text (cleaning)
│   └── tests/                  83 tests, forced to run fully offline
├── frontend/
│   └── src/
│       ├── App.tsx             3-column shell, chat state, polling
│       ├── components/         chat/ documents/ retrieval/ stats/ pipeline/ layout/ ui/
│       ├── hooks/              useDocuments · useSystemData · useTheme · useMediaQuery
│       ├── lib/                api (typed fetch) · export (PDF/JSON/MD) · format · utils
│       └── types/api.ts        TypeScript mirror of the Pydantic schemas
├── samples/                    3 generated demo documents (PDF, DOCX, TXT)
├── scripts/                    generate_samples.py · verify_stack.py
│                               build_program_doc.py (lab report)
├── uploads/                    original uploaded files (runtime)
├── vector_store/               persisted ChromaDB collection (runtime)
└── data/                       document registry + chat history (runtime)
```

---

## 🔌 API

Interactive docs at <http://127.0.0.1:8000/docs>.

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/documents` | Upload one or more files, run ingestion, return the pipeline trace |
| `GET` | `/api/documents` | Document registry with chunk counts and per-stage timings |
| `GET` | `/api/documents/{id}/chunks` | Inspect the stored chunks of a document |
| `DELETE` | `/api/documents/{id}` | Remove a document, its vectors and its file |
| `POST` | `/api/documents/rebuild` | Rebuild the whole vector store from `uploads/` |
| `POST` | `/api/chat/ask` | Ask a question — the full RAG loop |
| `GET` | `/api/chat/history` | Question history (drives the export button) |
| `DELETE` | `/api/chat/history` | Clear the history and the analytics counters |
| `GET` | `/api/stats` | Dashboard analytics |
| `GET` | `/api/system/status` | Component health + the pipeline stage definitions |
| `POST` | `/api/system/reset` | Wipe uploads, index, registry and history |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/chat/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What vector database is used for storage?", "top_k": 5}'
```

---

## 🧪 Tests & checks

```bash
pytest backend/tests -q          # 83 tests, no network required
python scripts/verify_stack.py   # end-to-end smoke test on the REAL stack
                                  # (real MiniLM + real ChromaDB + the 3 samples)
```

`backend/tests/conftest.py` forces the offline providers (hashed embeddings, NumPy index,
extractive LLM) so the suite never needs a network connection or an API key.

---

## 🛠 Handy commands

```bash
make install     # venv + pip install + npm install
make dev         # backend (:8000) + frontend (:5173) together
make test        # pytest
make build       # typecheck + production frontend bundle
make samples     # regenerate the three demo documents
```

Or without `make`:

```bash
python scripts/generate_samples.py    # writes samples/*.pdf|docx|txt
python scripts/verify_stack.py        # real-stack verification
cd frontend && npm run typecheck && npm run build
```

---

## 📄 Requirements

| Layer | Technology |
| --- | --- |
| UI | React 18, TypeScript 5.7, Vite 6, TailwindCSS 3.4, shadcn/ui (Radix), lucide-react, sonner |
| API | Python 3.10+, FastAPI, Pydantic v2, uvicorn |
| RAG | LangChain, langchain-text-splitters, langchain-chroma |
| Retrieval | ChromaDB (persisted, cosine) — NumPy flat index as a fallback |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Documents | pdfplumber, pypdf, python-docx |
| LLM | OpenAI · Ollama · Mistral · extractive baseline |

`requirements-minimal.txt` installs everything **except** torch / sentence-transformers /
ChromaDB, for machines that cannot download the heavy dependencies — the app then runs on the
hashed embedder and the NumPy index.

---

## 🎓 Notes for the lab report

* **Why the overlap matters** — a 100-character overlap keeps a sentence that straddles a
  chunk boundary retrievable from at least one chunk.
* **Why pages are tracked** — cleaning happens per page, and the char offsets are recomputed,
  so every citation can say *page 2, chunk 7* instead of just a chunk number.
* **Why the prompt is fixed** — `backend/prompts/templates.py` contains the required template
  verbatim; the "View prompt" button in the UI shows exactly what the model received.
* **Why a grounding gate** — a language model will happily answer a question the corpus does
  not cover. Term coverage is checked *before* generation and the model is told to abstain.

## 📄 License

MIT — free to use for the lab.
