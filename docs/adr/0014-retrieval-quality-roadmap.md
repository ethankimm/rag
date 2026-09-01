# 0014. Retrieval Quality Roadmap: Chunking, Contextual Retrieval, Reranking, and pgvector Consolidation

## Status

Proposed

## Context

ADR 0013 committed us to hybrid retrieval + reranking + agentic loop. SciFact benchmarking showed on single-hop zero-shot queries, tuned hybrid retriever already near ceiling. Further gains come from *outside* first-stage retriever: better chunks in, better ranking out, better query understanding on top.

Also want to stop maintaining custom BM25 index and linear fuser. Corpus small, grows slowly — dedicated vector DB (Weaviate, Pinecone, Qdrant) overkill. Postgres already in stack (ADR 0012), so `pgvector` = natural consolidation target.

## Decision

Staged roadmap, each step independently measured and revertible:

1. **On-corpus eval harness (prerequisite).** 50–200 labeled queries over `backend/rag-docs/` with gold chunk IDs. Report Hit@1/3/5, nDCG@10, MAP@10, latency, cost. SciFact stays external sanity check only.
2. **Semantic chunking.** Structure-aware (markdown headings, paragraph boundaries, code fences as atomic units) instead of fixed-token windows.
3. **Contextual Retrieval (Anthropic 2024).** Prepend LLM-generated per-chunk context to each chunk before embedding and lexical indexing. One-time ingest cost.
4. **Query-aware chunk expansion.** Generate 3–5 hypothetical questions per chunk at ingest, embed alongside chunk. Closes query↔doc vocabulary gap for FAQ/how-to corpus (ADR 0011).
5. **Cross-encoder reranker.** Keep `bge-reranker-v2-m3` as stage 2: retrieve top ~50 → rerank to top 5–10. (Reranker = cross-encoder; bi-encoders = first-stage embedding model.)
6. **Consolidate hybrid onto pgvector.** Dense via `pgvector` + HNSW, lexical via Postgres full-text search (`tsvector` / `ts_rank_cd`), fusion via RRF in SQL. Delete custom BM25 index and linear fuser once parity verified on step 1's harness. Keep retriever interface as adapter so future move to dedicated vector DB = swap.
7. **Agentic loop unchanged.** LangGraph loop with forced iteration + weighted RRF sits on top of reranker (ADR 0013).

Steps 2–4 are ingest-side and compose into one LLM pass per chunk.

## Deferred

- **SPLADE / SPLADE-v3** — better lexical arm, but marginal gain over BM25/tsvector once cross-encoder in place doesn't justify new model-serving dependency now.
- **ColBERTv2 / multi-vector** — only if reranker latency becomes bottleneck.
- **Listwise LLM rerankers (RankZephyr, RankGPT)** — Stage-3 rerank for high-value queries later.
- **HyDE, Step-Back prompting, Self-RAG-style critic** — cheap A/Bs inside LangGraph rewrite/critic node once ingest-side wins land.
- **GraphRAG** — deferred per ADR 0013.
- **CoRAG, Search-R1** — reference-only; training out of scope.

## Reasons

- **ROI order.** Ingest-side changes + cross-encoder rerank = highest and most reliable delta for corpora like ours, at lowest runtime cost.
- **One substrate.** pgvector removes custom BM25 + fuser code without over-committing to dedicated vector DB at current scale.
- **Attribution.** Each step = separate A/B so we know which lever moved metrics.

## Consequences

- Ingest cost rises (LLM calls per chunk for context + hypothetical questions). Must be prompt-cached and batched.
- Storage grows ~4–6× per chunk from hypothetical-question embeddings. pgvector handles this at our scale.
- RRF discards score magnitudes; acceptable because reranker sits downstream. Reranker-free fast path would need weighted score fusion reintroduced in adapter.
- pgvector cutover must hit parity on harness before custom BM25 + fuser deleted.

## Alternatives Considered

- **Agentic-loop improvements first, ingest-side later** — rejected; loop headroom on mostly-single-hop corpus is small.
- **Dedicated vector DB (Weaviate/Pinecone/Qdrant) now** — rejected for current corpus size; pgvector sufficient and reuses ADR 0012 infra.
- **Replace BM25 with SPLADE in this ADR** — rejected; marginal gain behind cross-encoder doesn't justify new dependency. Kept in Deferred.

## Relationship to Other ADRs

- **ADR 0009, 0010** — Step 6 supersedes local BM25 + linear fuser implementation; dense + lexical substrate decision preserved, only implementation moves to pgvector.
- **ADR 0011** — Motivates step 4 (hypothetical questions) for how-to corpus.
- **ADR 0012** — Reuses Postgres instance via pgvector.
- **ADR 0013** — This ADR is quality roadmap on top of 0013's control-flow decision; does not change it.
