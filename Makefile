.PHONY: app-dev dev frontend-dev frontend-build frontend-check import-hf-docs ingest eval test test-backend test-frontend test-bm25

PG_PORT ?= 5433
PG_DSN ?= postgresql://rag:rag@localhost:$(PG_PORT)/rag

dev:
	PG_PORT=$(PG_PORT) docker compose up -d
	PG_DSN=$(PG_DSN) uv run uvicorn backend.api:app --reload

app-dev:
	$(MAKE) -j2 dev frontend-dev

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

frontend-check:
	cd frontend && npm run check

import-hf-docs:
	uv run python -m backend.ingestion.hf_course_import

ingest:
	PG_PORT=$(PG_PORT) docker compose up -d
	PG_DSN=$(PG_DSN) uv run python -m backend.ingestion

eval:
	PG_PORT=$(PG_PORT) docker compose up -d
	PG_DSN=$(PG_DSN) uv run python -m backend.evaluation

test: test-backend test-frontend

test-backend:
	uv run pytest -m "not integration and not model_integration"

test-frontend:
	cd frontend && npm test

test-bm25:
	PG_PORT=$(PG_PORT) docker compose up -d
	PG_DSN=$(PG_DSN) REQUIRE_POSTGRES_TESTS=1 uv run pytest -m integration
