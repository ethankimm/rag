# 0010. Use bm25s for a Local BM25 Keyword Store

## Status

Accepted

## Context

Local stack (ADR 0009) retrieves only with ChromaDB dense vectors. Dense retrieval misses exact-term matches — identifiers, error codes, config keys, acronyms — common in "how" documents we optimize for (ADR 0011). Want lexical BM25 signal to complement dense retrieval, kept local and in-process for current prototype without adding service.

## Decision

Add BM25 as separate keyword store alongside vector store, backed by
[`bm25s`](https://github.com/xhluca/bm25s):

1. **Library** — `bm25s`: SciPy sparse matrices with scores precomputed at index time (query is sparse mat-vec), plus memory-mappable indices. In-process, no server.
2. **Boundary** — Expose behind `keyword_store` interface mirroring `vector_store`, swappable later without touching callers.
3. **Fusion and rerank** — Written in-house as separate layer over both stores. Fuse on rank or per-source normalized scores (e.g. RRF), never by adding raw BM25 scores to Chroma distances.
4. **Index build** — Built from same chunks during ingestion, persisted next to vector store.

## Reasons

- **Hybrid improves recall** — Lexical plus dense covers exact-term and semantic matches; standard retrieval baseline for RAG.
- **Fits the local prototype** — Pure-Python and in-process, consistent with local-first stack in ADR 0009 and replaceable provider seams.
- **Fast at our scale** — Precomputed sparse scoring keeps latency low for current single-space corpus.

## Consequences

- Ingestion gains BM25 index build and persist step. Freshness follows batch reindex cadence; `bm25s` has no incremental add/update/delete.
- We own tokenization and normalization (lowercasing, stopwords, stemming) shared between index and query time; retrieval quality depends on it.
- We own fusion and rerank logic and its evaluation.
- `bm25s` limits (no incremental updates, no metadata filtering, in-process memory) will force move to Lucene-class store if we later need freshness, filtered keyword search, or larger scale.

## Alternatives Considered

### rank_bm25

Rejected — simpler but pure-Python loops too slow past few thousand documents.

### Tantivy or SQLite FTS5

Deferred — both run locally, add incremental updates and metadata filtering. Revisit when freshness or filtered keyword search becomes requirement.

### OpenSearch / Elasticsearch (self-hosted)

Deferred — production-grade Lucene BM25, but JVM server premature for local prototype. Managed cloud variants add cost we don't need yet.

### Pyserini

Rejected as runtime — JVM-based and research-oriented. Keep only as possible evaluation harness for benchmark numbers.

## Relationship to Other ADRs

- **ADR 0009** — Extends local FastAPI/Chroma stack by adding lexical retrieval path; vector store unchanged.
- **ADR 0011** — Sibling decision it references; BM25 indexes same GitHub-sourced markdown corpus, not Confluence sync.
