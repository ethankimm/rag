# 0003. Use Vertex AI RAG Engine for Retrieval and Indexing

## Status

Superseded

Superseded by
[0009. Use Local FastAPI, ChromaDB, and OpenAI-Compatible RAG](0009-use-local-fastapi-chromadb-and-openai-compatible-rag.md)
for current build.

Retained as historical context for managed Vertex AI RAG Engine production direction. Vertex AI, GCS import, managed RAG Engine retrieval not active implementation guidance for current local prototype.

## Context

Internal RAG over markdown in single Git repository (ADR 0004). Backend must ingest, chunk, embed, index, retrieve at query time, feed context to LLM for grounded answers with citations.

Google **Vertex AI RAG Engine**: managed RAG runtime — corpus creation, file import, chunking, embedding, indexing, retrieval. **Not** LLM training/fine-tuning — RAG corpus = searchable index for retrieval at inference time.

## Historical Decision

Use **Vertex AI RAG Engine** for **retrieval and indexing**:

- **Corpus** — create/manage `RagCorpus` as knowledge base index.
- **Import** — ingest markdown via Cloud Storage `import_files` (production) or `upload_file` (local dev).
- **Chunk, embed, index** — RAG Engine handles unless we override parsers or embedding config later.
- **Retrieve** — `retrieval_query` (or RAG retrieval tools on `generate_content`) against corpus at query time.

**Source sync model** for Git-backed markdown (ADR 0004):

1. Team commits markdown to Git repo (and/or publishes via Wiki.js Git sync into same repo).
2. CI syncs files to GCS bucket (e.g. `gs://<bucket>/rag-docs/`).
3. RAG Engine `import_files` imports from GCS into corpus.
4. Re-import skips unchanged files via content-hash `version_id`; changed files reindexed as whole files (not chunk-level diff).
5. Deleted Git files **not** removed from corpus automatically; must call `delete_file` or implement explicit delete reconciliation.

**LLM generation** via direct Google Vertex AI SDK in application code, not LangChain chains/wrappers. RAG Engine handles retrieve; backend handles generate + citation assembly.

**Not** operate own vector store or LangChain ingestion pipeline for production retrieval in v1.

## Reasons

### Managed Ingestion And Retrieval

RAG Engine owns chunking, embedding, indexing, retrieval. We focus on Git→GCS sync, corpus lifecycle, application-layer citations — not building/operating embedding pipelines.

### Aligns With GCP And Single-Repo Markdown

Git-backed markdown syncs cleanly to GCS + `import_files`. Matches one-repo rollout without custom parsers for Confluence or other formats.

### Production Scope

Architecture supports ongoing internal rollout over Git-backed corpus in one repository.

### LangChain Not Used For Ingestion

Avoid LangChain for ingestion/retrieval. RAG Engine + thin application code keeps corpus lifecycle, cloud sync, retrieval policy in managed Google infrastructure.

## Consequences

Depend on Vertex AI RAG Engine APIs, GCS for corpus import, CI to sync Git markdown to bucket.

Operational work: corpus refresh, delete reconciliation, monitoring import/retrieval failures.

Citation metadata must map retrieved chunks to repo paths and Wiki.js URLs (ADR 0001).

## Alternatives Considered

### LangChain Ingestion Pipeline

LangChain loaders + local vector stores add framework dependency without corpus lifecycle or enterprise retrieval policy ownership. Rejected for v1.

### Custom Chunking, Embedding, And Vector Store

Full pipeline control, but we operate indexing infrastructure ourselves. Deferred for RAG Engine in v1. RAG Engine allows chunking parameters to reduce hallucinations; own pipeline adds overhead without clear benefit now.

### LangChain End-To-End RAG

Rejected — avoid LangChain owning retrieval, generation, or storage schema.
