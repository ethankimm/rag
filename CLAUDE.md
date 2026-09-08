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
- **Local models**: embedding and reranking GGUF models served by a persistent,
  externally owned Model Server. Application processes own only their validated
  Local Model Runtime clients.
- **Generation**: OpenAI-compatible chat completion through the OpenAI SDK.
- **Source of truth for RAG ingestion**: `backend/rag-docs/`, with documents as
  markdown files.
- **Access model**: one local installation. All locally available documents are
  available to chat; there are no user or administrator roles.
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

Start the local Model Server and Postgres, then load and index the markdown
corpus:

    make ingest

Run a manual RAG query smoke test:

    make rag-smoke

Direct `uv run python -m backend.rag` invocation requires PostgreSQL and the
Model Server to be running already.

The standard `make dev`, `make ingest`, `make eval`, and `make rag-smoke`
commands start or reuse the persistent Model Server. Manage it explicitly with:

    make model-server-up
    make model-server-down

Startup reports server and embedding-model waits. Shutdown signals only the
process identity recorded by this repository. An untracked process on the
configured port fails startup and is never terminated automatically.

`docs/local-model-runtime.md` is the canonical configuration and operations
reference. Run `make model-server-down` before changing local model environment
settings, then restart application processes. Standard Make commands start the
server again with the new contract. The lifecycle decision is recorded in
`docs/adr/0015-use-an-external-persistent-local-model-server.md`.

Run the backend API:

    make dev

Run the portable backend and frontend tests:

    make test

Run the opt-in real GGUF runtime gate only on a model-capable machine:

    make test-runtime

Lint and format-check with Ruff:

    uv run ruff check backend tests
    uv run ruff format --check backend tests

## Testing

- **pytest** — unit tests for pipeline logic, in `tests/`. Normal backend tests
  use deterministic HTTP adapters and require no GGUF files or Model Server.
  Run with `make test-backend`.
- **Real runtime gate** — `make test-runtime` validates configured GGUF
  embedding and reranking through the production client contract. It leaves
  the persistent Model Server running for reuse.

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
