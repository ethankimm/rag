# CLAUDE.md — git-rag-app

Internal RAG over Git-backed markdown docs. The prototype phase is complete;
the current objective is to harden and deploy the system on GCP for internal
use.

## Current architecture and deployment direction

This is the ground truth. Work against this stack and the accepted ADRs that
describe it.

- **Backend API**: FastAPI, running locally.
- **Vector store / retrieval**: Postgres + `pgvector`, run locally via
  `docker-compose.yml`. Hybrid retrieval (dense cosine + full-text search, fused
  with RRF) is served by the store itself in one SQL query — there is no
  separate BM25 index or Python fuser.
- **Embeddings**: local GGUF embedding model through `llama-cpp-python`.
- **Generation**: OpenAI-compatible chat completion through the OpenAI SDK.
- **Source of truth for RAG ingestion**: `backend/rag-docs/`, with documents as
  markdown files.
- **Frontend**: Astro, calling the FastAPI `/chat` endpoint during local
  development.
- **Deployment target**: GCP for internal use; local Docker services remain
  available for development and verification.
- **Cloud adoption status**: GCP deployment is an active workstream. Introduce
  cloud services incrementally when required by an accepted deployment design
  or ADR.
- **Explicitly out of scope**: Wiki.js, Next.js, and a required Ollama runtime.

## Working rules / engineering standards

The prototype phase is complete. Treat every change as production-bound while
keeping the implementation proportional to the current scale and deployment
requirements.

- Approach design, implementation, and review as a senior enterprise engineer.
- Before changing code, understand the existing architecture, contracts,
  dependencies, failure modes, and operational impact.
- Prefer small, reversible, independently verifiable changes over broad
  rewrites.
- Preserve clear module boundaries and provider abstractions so infrastructure
  and service implementations can be replaced without changing application
  behavior.
- Do not add dependencies, cloud services, or infrastructure without
  documenting the problem they solve, the operational cost, and why existing
  components are insufficient.
- Production-facing code must include appropriate validation, type hints,
  structured error handling, logging, tests, and configuration through
  environment variables or deployment configuration.
- Treat security, least privilege, secret management, data privacy, dependency
  safety, and network boundaries as first-class requirements.
- Consider reliability explicitly: health checks, timeouts, bounded retries,
  graceful shutdown, idempotent ingestion, observability, resource limits,
  and rollback paths.
- Do not silently change public APIs, data schemas, embedding dimensions,
  retrieval behavior, or persistence semantics. Update tests and documentation
  when contracts change.
- Validate changes with focused tests, the full test suite where practical,
  linting, and an appropriate deployment or integration smoke test.
- Keep ADRs, runbooks, configuration examples, and deployment documentation
  synchronized with the implementation.
- Avoid speculative production architecture. Build only the GCP components
  required for the current deployment milestone, without creating avoidable
  operational debt.

### Tooling and dependency management

- Use `uv` for environment creation, dependency installation, locking, and
  command execution.
- Run Python tools through `uv run`; do not bypass the project environment with
  system Python or ad hoc `pip` installs.
- Run Ruff after substantive Python changes:
  `uv run ruff check backend tests`.
- Run the formatter check before completing work:
  `uv run ruff format --check backend tests`.
- Apply formatting with `uv run ruff format backend tests` when necessary.
- Keep runtime dependencies, development dependencies, Ruff configuration, and
  project metadata in `pyproject.toml`.
- Update `pyproject.toml` whenever dependencies or tooling configuration
  change, then regenerate and commit `uv.lock`.
- Keep tests, linting, and formatting checks passing before considering a
  change complete.

## How to run

Setup (one time). Dependencies are declared in `pyproject.toml`:

    uv sync --extra dev

Start the local Postgres + pgvector store:

    docker compose up -d

Load and index the markdown corpus:

    uv run python -m backend.ingestion

Run a manual RAG query smoke test:

    uv run python -m backend.rag

Optional cross-encoder reranking (off by default). Download the reranker GGUF
into `models/`, serve it, then set `RERANK_ENABLED=1`. Reranking pools over the
whole (query, document) pair in one batch, so `-ub`/`-c` must cover the longest
pair (the 512-token default 500s on long docs):

    llama-server -m models/bge-reranker-v2-m3-Q8_0.gguf --reranking -c 8192 -b 8192 -ub 8192 -ngl 99 --port 8080

Run the backend API:

    uv run uvicorn backend.api:app --reload

Run the tests:

    uv run pytest

Lint and format-check with Ruff:

    uv run ruff check backend tests
    uv run ruff format --check backend tests

## Testing

- **pytest** — unit tests for pipeline logic, in `tests/`. Uses small temporary
  fixtures plus smoke coverage over the local RAG components. Run with
  `uv run pytest`.

Planned (do NOT install ahead of a step that explicitly needs them):

- **Playwright** — end-to-end / UI tests once the Astro frontend has a real page
  and the RAG chat flow works.
- **k6** — load / performance testing once the retrieval + generation endpoint
  behavior is stable enough to benchmark.

## Architecture decisions

`docs/adr/` records the design choices for the project. Accepted ADRs are active
instructions unless superseded. Superseded ADRs are historical context only.

How the current choices map to the ADRs:

- **FastAPI + Postgres/pgvector + hybrid retrieval + GGUF embeddings +
  OpenAI-compatible generation** — current RAG implementation. The pgvector
  consolidation and retrieval-quality direction are documented in the active
  retrieval ADRs.
- **Git backend + markdown files** — matches accepted
  `docs/adr/0004-git-only-markdown-for-initial-rollout.md`.
- **Astro frontend** — the accepted single-site direction in
  `docs/adr/0006-evaluate-single-astro-vite-frontend-vs-wikijs-split.md`, instead
  of the superseded Wiki.js + Next.js split in
  `docs/adr/0001-split-frontend-wikijs-and-rag-chat.md` and
  `docs/adr/0002-use-nextjs-and-react-for-frontend.md`.

Status of the ADRs:

- **ADRs 0001–0003 (Superseded)** describe the earlier Wiki.js, Next.js, and
  Vertex AI RAG Engine target. Do not implement against them.
- **ADR 0004 (Accepted)** keeps Git-only markdown as the source format.
- **ADR 0006 (Accepted)** chooses a single Astro frontend.
- **ADR 0009 (Accepted)** records the original local FastAPI, GGUF embeddings,
  and OpenAI-compatible RAG direction. The current Postgres/pgvector
  implementation supersedes its ChromaDB details.
- **ADRs 0005, 0007, and 0008 (Proposed)** are unresolved evaluations or future
  improvements. Do not treat them as implementation decisions until accepted.
