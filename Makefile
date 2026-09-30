# ---------------------------------------------------------------------------
# RAG-NLP Assistant - developer shortcuts
#   make install   first-time setup (venv + pip + npm)
#   make dev       backend on :8000 and frontend on :5173, Ctrl-C stops both
#   make test      backend test suite (offline, no API key needed)
#   make verify    end-to-end smoke test against the real stack
#   make build     typecheck + production bundle
# ---------------------------------------------------------------------------
PY      ?= python3
VENV    ?= .venv
BIN      = $(VENV)/bin
PIP      = $(BIN)/pip
UVICORN  = $(BIN)/uvicorn
PYTEST   = $(BIN)/pytest

.DEFAULT_GOAL := help
.PHONY: help install install-backend install-frontend dev dev-backend dev-frontend \
        test verify build typecheck samples reset clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: install-backend install-frontend ## Create the venv, install Python and Node deps

install-backend: ## python -m venv .venv && pip install -r requirements.txt
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-frontend: ## npm install inside frontend/
	cd frontend && npm install

$(UVICORN): ## Create the virtualenv on demand
	$(PY) -m venv $(VENV) && $(PIP) install --upgrade pip && $(PIP) install -r requirements.txt

dev: $(UVICORN) ## Run the API and the Vite dev server together
	@trap 'kill 0' INT TERM EXIT; \
	$(UVICORN) backend.main:app --reload --host 127.0.0.1 --port 8000 & \
	(cd frontend && npm run dev) & \
	wait

dev-backend: $(UVICORN) ## Run only the FastAPI server with autoreload
	$(UVICORN) backend.main:app --reload --host 127.0.0.1 --port 8000

dev-frontend: ## Run only the Vite dev server
	cd frontend && npm run dev

test: $(PYTEST) ## Run the backend test suite
	$(PYTEST) backend/tests -q

verify: $(UVICORN) ## End-to-end check on the real stack (MiniLM + ChromaDB)
	$(BIN)/python scripts/verify_stack.py

build: typecheck ## Typecheck and bundle the frontend
	cd frontend && npm run build

typecheck: ## TypeScript typecheck
	cd frontend && npm run typecheck

samples: $(UVICORN) ## Regenerate the three demo documents in samples/
	$(BIN)/python scripts/generate_samples.py

reset: ## Delete uploads, the vector store and the registry/history
	rm -rf vector_store/* uploads/* data/registry.json data/chat_history.json

clean: ## Remove build artefacts and caches
	rm -rf frontend/dist frontend/node_modules/.vite .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
