# 0001. Wiki.js for Write and Browse, Custom Chat App for RAG

## Status

Superseded

Superseded by
[0006. Use a Single Astro Frontend](0006-evaluate-single-astro-vite-frontend-vs-wikijs-split.md)
for frontend direction and
[0009. Use Local FastAPI, ChromaDB, and OpenAI-Compatible RAG](0009-use-local-fastapi-chromadb-and-openai-compatible-rag.md)
for current RAG architecture.

Retained as historical context for earlier split Wiki.js + custom chat direction. Current accepted build: one Astro frontend, no Wiki.js.

## Context

Internal RAG over markdown in Git. Employees read cited sources; some amend docs without editing GitHub directly.

Rollout uses **one Git repository**. All markdown in that repo; both frontends read same corpus (see ADR 0004).

Wiki.js: open-source wiki, markdown, page search, auth (Okta), Git storage. Good for **writing and browsing** for non-markdown/GitHub users.

Wiki.js poor fit for hybrid search, source selection, citation-aware RAG. Those belong in purpose-built chat interface.

## Historical Decision

**Split frontend model** over **single shared markdown corpus**:

1. **Wiki.js** — internal wiki for **writing and browsing**. Content syncs to Git via Wiki.js Git storage. Okta + other auth in Wiki.js for wiki access.

2. **Custom Next.js chat app** (see ADR 0002) — **RAG Q&A with citations**. Users ask; answers cite sources linking to Wiki.js pages (and repo paths). Does not replace wiki editor.

3. **Shared corpus** — both frontends reference same Git-backed markdown tree (e.g. `rag-docs/`). Wiki.js = editing surface; Vertex AI RAG Engine indexes content for retrieval (ADR 0003).

## Reasons

### Wiki.js Covers Write and Browse Well

Building markdown editing, navigation, version history, Okta SSO into custom app duplicates Wiki.js without improving RAG product. Wiki.js solves wiki UX for non-technical authors; don't rebuild.

### RAG Needs A Purpose-Built Chat UI

RAG needs search, source inspection, question input, cited answers. Wiki.js not designed for retrieval transparency, source selection before generation, or chunk-level citations in chat flow.

### Citations Reference The Wiki

Answers cite Wiki.js pages — surface users already use — not parallel preview UI. Ingestion maps repo paths to wiki URLs for consistent citations.

### One Repo, Two Surfaces

Not splitting content across repos. One Git repo = source of truth; Wiki.js and RAG backend consume same markdown corpus.

## Consequences

Operate **two applications** for v1: Wiki.js + RAG chat app. Same identity provider (e.g. Okta) where practical.

Wiki.js Git sync and RAG ingestion must stay aligned on same repo/branch. Path-to-wiki-URL mapping documented and applied at ingest.

We own **RAG chat app**: question UX, citation display, API integration. Wiki.js owns editing, browsing, wiki search.

Next.js framework choice in
[0002. Use Next.js and React for the Frontend](0002-use-nextjs-and-react-for-frontend.md).

## Alternatives Considered

### Wiki.js As The Only Frontend

RAG with source selection and chunk-level citations doesn't fit Wiki.js without heavy customization or poor bolt-on chat.

### Fully Custom Application

Single custom app rebuilds wiki functionality Wiki.js already provides; increases scope before RAG workflow proven.

### Wiki.js Plus Custom RAG Chat App

Chosen pattern: Wiki.js for write/browse, custom Next.js for RAG with citations to wiki pages, both backed by one Git markdown corpus.
