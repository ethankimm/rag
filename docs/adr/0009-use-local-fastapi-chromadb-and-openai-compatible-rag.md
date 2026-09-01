# 0009. Use Local FastAPI, ChromaDB, and OpenAI-Compatible RAG

## Status

Accepted

## Context

Project pivoted from earlier target: Wiki.js, Next.js, Vertex AI RAG Engine, GCS import, managed cloud retrieval. Current build = local prototype proving RAG loop end to end before production infrastructure.

Repository now contains:

- Astro frontend posting chat requests to local API
- FastAPI backend exposing `/chat`
- markdown documents in `backend/rag-docs/`
- local ingestion pipeline: load, chunk, embed, index markdown
- persisted ChromaDB vector store under `backend/data/vector_store`
- GGUF embedding generation through `llama-cpp-python`
- chat completion through OpenAI SDK against OpenAI-compatible provider

Still want system to evolve toward enterprise controls: observability, repeatable ingestion, citation quality, auth, deployment discipline. Add incrementally on local architecture instead of reintroducing superseded production stack by default.

## Decision

Use local Python RAG backend for current accepted build:

1. **API** — FastAPI owns local chat endpoint at `/chat`.
2. **Retrieval** — ChromaDB is local persisted vector store.
3. **Document loading and chunking** — Markdown loaded from `backend/rag-docs/`, split locally before indexing.
4. **Embeddings** — Local GGUF model through `llama-cpp-python`.
5. **Generation** — OpenAI SDK for OpenAI-compatible chat completion.
6. **Frontend integration** — Astro frontend calls FastAPI endpoint directly during local development.

Current accepted build does **not** use Wiki.js, Next.js, Vertex AI RAG Engine, GCS imports, managed cloud retrieval, or Ollama as required infrastructure.

## Reasons

### Proves The Product Loop Locally

Smallest useful system: markdown corpus, retrieval, context assembly, generation, browser chat UI. Local stack proves workflow without cloud setup, multiple deployed apps, or managed RAG services.

### Keeps Provider Boundaries Replaceable

FastAPI, ChromaDB, embedding function, chat client = clear seams for later replacement. Can evaluate better retrieval, hosted inference, managed vector stores, cloud deployment without rewriting frontend.

### Matches The Current Repo

Existing code already uses FastAPI, ChromaDB, local GGUF embeddings, OpenAI SDK. Recording as accepted architecture keeps future agents and contributors from implementing against superseded ADRs.

### Avoids Premature Production Scope

Vertex AI RAG Engine, GCS sync, Wiki.js, Next.js, enterprise auth may become relevant later. Not required to validate retrieval quality, citation UX, query preparation, or answer generation in current prototype.

## Consequences

Team owns ingestion, chunking, embedding, vector persistence, retrieval thresholds, context formatting, API behavior for now.

Initial HDTC Confluence corpus test suggests local embedding and retrieval latency acceptable for single team space, but retrieval quality and evaluation discipline now limiting factors.

Operational maturity must be added deliberately before production use: repeatable reindexing, failure handling, request logging, citation validation, configuration management, auth, deployment, monitoring.

OpenAI-compatible generation stays behind small client boundary so backend can move between OpenAI, local OpenAI-compatible server, or another provider later.

Older managed-cloud direction in ADR 0003 superseded for current build, not permanently rejected for all future production deployments.

## Alternatives Considered

### Vertex AI RAG Engine

Superseded for current build. Remains plausible production option if managed retrieval, GCS import, cloud operations become right tradeoff after local RAG behavior proven.

### Wiki.js Plus Next.js

Superseded by ADR 0006. Current architecture uses one Astro frontend for browse and chat instead of separate wiki and chat products.

### Ollama As Required Runtime

Deferred. Local model runner may still be useful, but current backend implemented against OpenAI SDK and should only require OpenAI-compatible chat completion provider.
