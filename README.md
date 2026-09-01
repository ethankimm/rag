# Git RAG App

Retrieval-augmented chat over Git-backed markdown documents. The current stack
is an Astro + React frontend, Python/FastAPI, PostgreSQL with pgvector and
pg_textsearch, local BGE-M3 embeddings and cross-encoder reranking, and OpenAI
chat completion.

Current architecture decisions are recorded in `docs/adr/`: ADR 0004 keeps the
Git-only markdown corpus, ADR 0006 chooses the single Astro frontend, and ADR
0009 chooses the local FastAPI/ChromaDB/OpenAI-compatible RAG backend. ADRs
0001-0003 are superseded historical context.

## Local Setup

Install the locked Python environment from the repo root:

```bash
uv sync --extra dev
```

Copy the example environment file and add your local API value:

```bash
cp .env.example .env
```

If your organization stores these values in Vault, retrieve the secret value and
paste the value into `.env`. Do not commit `.env`, real API keys, or personal
Vault paths.

## Hugging Face LLM Course Snapshot

The repository includes a normalized Markdown snapshot of all 104 English
lessons from chapters 0–12 of the Hugging Face LLM Course. The snapshot is
pinned to `huggingface/course@5805d51523d561a82520b301dbc8c5759b212844`
and includes its Apache-2.0 license, source metadata, and checksums under
`backend/rag-docs/huggingface-llm-course`.

Normal ingestion is network-free. To deliberately regenerate the same pinned
snapshot from its upstream archive:

```bash
make import-hf-docs
```

The importer converts course MDX into retrieval-friendly Markdown and fails on
unexpected source counts, unsafe archive paths, or unsupported course-specific
components.

## Build The Vector Store

Start PostgreSQL, then load, chunk, embed, and atomically replace the
`markdown_docs` application collection:

```bash
docker compose up -d
uv run python -m backend.ingestion
```

## Start The Application

After configuring `.env` and indexing the markdown corpus, start the backend
and frontend together:

```bash
make app-dev
```

The Astro development server sends browser requests to `/api/chat` and proxies
them to FastAPI at `http://127.0.0.1:8000/chat`. This keeps the browser on one
origin during local development. For separate processes, use the commands below.

### Start The Backend

Run the FastAPI server from the repo root:

```bash
uv run uvicorn backend.api:app --reload
```

FastAPI lifespan management starts and closes the local model client, PostgreSQL
store, and generation client. The chat API runs at
`http://127.0.0.1:8000/chat`.

Each chat request runs through a bounded LangGraph workflow. The graph retrieves
evidence, drafts an answer, and classifies its grounding and completeness. A
low-confidence answer produces up to two targeted follow-up searches before it
is rewritten and classified again. The graph makes at most three answer
attempts; if the third remains below `RAG_CONFIDENCE_THRESHOLD`, the latest
answer is returned with an `Unsure:` prefix and `found=false`.

`POST /chat` accepts a required `message` plus an optional `history` array of up
to 12 prior user/assistant messages. On follow-up turns, a low-temperature query
rewrite resolves references into a standalone retrieval query while answer
generation retains the original message and history. The response contract
remains `answer`, `found`, `confidence`, and `sources`.

`RAG_CONFIDENCE_THRESHOLD` defaults to `0.5` for the graph's grounded-answer
assessment. Generation uses `RAG_TEMPERATURE=0.1` by default to reduce output
variability. Every workflow completion uses `OPENAI_MODEL`, which defaults to
the small `gpt-4.1-nano` model. An explicit `RagWorkflow` model override takes
precedence over the environment setting.

### Start The Frontend

In a second terminal, install and run the Astro + React application:

```bash
cd frontend
npm install
npm run dev
```

The chat UI runs at `http://127.0.0.1:4321` and calls the local FastAPI service
through Astro's `/api` proxy. Set `RAG_API_URL` in `frontend/.env` when the
development backend is hosted elsewhere. For a deployment without a same-origin
reverse proxy, set `PUBLIC_RAG_API_URL` to the browser-accessible backend URL and
set the matching browser origin in `RAG_CORS_ORIGINS` on the backend.

To create the production static bundle:

```bash
make frontend-build
```

## Run Tests

Run all default backend and frontend tests, followed by static frontend checks
and the production build:

```bash
make test
make frontend-check
make frontend-build
```

Run the optional PostgreSQL integration suite, which includes native BM25
coverage and an isolated full-course ingestion/search check:

```bash
make test-bm25
```

The default suites mock network, OpenAI, local model, and database boundaries;
they do not consume API credits. Set `RUN_MODEL_INTEGRATION=1` to include the
slower local GGUF routing test.

## Full-Stack Course Smoke Test

After adding `OPENAI_API_KEY` to `.env`, run the real corpus through the local
models and open the chat application:

```bash
make ingest
make app-dev
```

Ask a course question such as “What is the difference between NLP and an LLM?”,
then a contextual follow-up such as “Which libraries does it use?”. Confirm the
UI renders a grounded answer, confidence, and course source chips. Conversation
history is held only in browser memory and is cleared by refresh or **New
conversation**.

## Run the BEIR SciFact Evaluation

Place the embedding and reranker GGUF files in `backend/models/` using these
model IDs:

```text
bge-m3-q8_0.gguf
bge-reranker-v2-m3-Q8_0.gguf
```

The GGUF weights are local runtime artifacts and are intentionally excluded
from Git because each file is roughly 600 MB. Provision both files separately
on every development or deployment machine. The tracked
`backend/models/llama-models.ini` file supplies their llama.cpp model settings.

The PostgreSQL image includes `pg_textsearch` 1.4.0 for native BM25 alongside
pgvector. The managed llama.cpp router reads `backend/models/llama-models.ini`
so the embedding model keeps CLS pooling while rank pooling is applied only to
the cross-encoder. Start Postgres, install the development dependencies, and run:

```bash
docker compose up -d
uv sync --extra dev
uv run python -m backend.evaluation
```

To evaluate only the lexical arm—without starting llama.cpp, embedding queries,
fusion, or reranking—run:

```bash
uv run python -m backend.evaluation --lexical-only --rebuild
```

If port 5432 is already occupied, start Docker with `PG_PORT=5433` and pass
`--dsn postgresql://rag:rag@localhost:5433/rag` to the evaluator.

The evaluator automatically rebuilds an existing SciFact collection when its
stored embedding-model metadata is missing or incompatible. Use `--rebuild` to
force a fresh index regardless.
