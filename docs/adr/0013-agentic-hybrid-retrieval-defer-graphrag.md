# 0013. Agentic Hybrid Retrieval as the Default; Defer GraphRAG

## Status

Proposed

## Context

At frontier, "complete" retrieval architecture can route across several strategies behind intent classifier: hybrid (dense + BM25) for factual lookup, GraphRAG local search for entity-anchored multi-hop, GraphRAG global search for corpus-wide sensemaking, all cascaded with reranking and increasingly agentic loop controlling when and how often to retrieve.

Building all lanes up front = over-engineering for our workload. Two observations drive decision:

1. **Corpus-wide sensemaking is rare.** "Summarize major themes across whole corpus" = demo question, not typical traffic for internal docs/how-to QA (ADR 0011). Small fraction of real queries.
2. **Most multi-hop is shallow and discoverable.** Majority of questions single-hop factual or shallow multi-hop where each next hop discoverable from previously retrieved chunks. Agentic loop over hybrid retrieval handles these by decomposing query and issuing iterative, targeted retrievals.

GraphRAG's distinct value confined to two cases agentic loop can't cover cheaply: **structural multi-hop**, where bridging entity not discoverable from local context and only precomputed graph reveals it, and **corpus-wide sensemaking**, where community summaries beat point retrieval. Both high-fixed-cost to serve: graph construction expensive at index time, costly to keep fresh as knowledge base changes.

## Decision

Adopt **hybrid retrieval + reranking + agentic control loop** as default and only retrieval architecture for now. Defer GraphRAG (local and global) until evidence from real traffic shows we need it.

Committed direction:

1. **Substrate** — Keep hybrid dense + BM25 retrieval (ADR 0009, ADR 0010) as base retrieval layer.
2. **Reranking** — Rerank top fused candidates with cross-encoder before generation.
3. **Agentic control** — LLM controls retrieval: decompose complex queries into sub-questions, retrieve per sub-question, iterate until enough to answer. Covers factual and discoverable-bridge multi-hop by pushing reasoning to query time.
4. **Guardrails** — Bound loop (max iterations, retrieval budget, graceful give-up) so runtime cost and non-convergence stay controlled.
5. **GraphRAG deferred** — Don't build local or global graph lanes now. Add later, behind router, only when two triggers below appear in real traffic.

## Reasons

- **Cheaper, and it gets most of the way there.** Hybrid + reranker + loop covers vast majority of queries at fraction of GraphRAG's index-time and maintenance cost.
- **Fits continually evolving knowledge base.** Vector/BM25 updates = cheap upserts; graph needs re-extraction, entity resolution, re-linking, community recomputation — either constantly rebuilt or served stale.
- **Compute shifts to runtime, where only paid when needed.** Agentic loop spends tokens per query on multi-hop work instead of large fixed graph-build cost up front for payoff most queries never use.
- **Matches where pragmatic 2026 systems land.** Recent architectural shift = *who controls retrieval* (static pipeline → agent), not new index structures. Hybrid + reranker + agentic control + router = stable spine; extra lanes conditional.

## Consequences

- Multi-hop answers cost more at query time (retrieval + LLM turn per hop) and have higher, more variable latency than single retrieval pass.
- Agentic loop = new surface to evaluate and observe: convergence, iteration counts, cost per query, failure/guardrail behavior.
- **Structural multi-hop** (non-discoverable bridging entity) and **corpus-wide sensemaking** knowingly under-served until GraphRAG added. Accept small quality gap on these rare query types for lower cost and operational simplicity.
- Must measure own query distribution to know when deferral no longer holds.

## Alternatives Considered

### Full four-lane router (hybrid + local + global + classifier) now

Rejected for now — provisions expensive graph infrastructure for rare query types in our domain. Target design space, not size current workload needs.

### Static hybrid RAG without an agentic loop

Rejected — leaves discoverable-bridge multi-hop on table. Loop = cheap way to cover most multi-hop without graph.

### "Run both, pick better" (hybrid and GraphRAG in parallel, judge)

Rejected — doubles cost and latency, adds judging LLM call, most expensive architecture for least marginal gain.

## Follow-up Work

- Build small eval set of *actual* hard multi-hop questions; label each discoverable-bridge vs structural. That ratio decides how much graph would add.
- Instrument query traffic for two GraphRAG triggers (structural multi-hop, corpus sensemaking). Promote GraphRAG from deferred to first-class lane only when frequency justifies fixed cost.
- When adding GraphRAG, prefer cheaper incrementally updatable variant (e.g. LightRAG / HippoRAG-style) over original GraphRAG index given freshness requirement.

## Relationship to Other ADRs

- **ADR 0007** — Supplies routing/classification layer this decision would reuse to branch between agentic hybrid path and future GraphRAG path.
- **ADR 0009, ADR 0010** — Provides dense + BM25 substrate agentic loop and reranker sit on top of.
- **ADR 0011** — GitHub-sourced how-to corpus whose query distribution (mostly factual/shallow multi-hop) motivates deferring GraphRAG.
