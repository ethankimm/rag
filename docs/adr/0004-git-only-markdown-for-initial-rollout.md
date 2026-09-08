# 0004. Git-Only Markdown for Initial Rollout

## Status

Accepted

## Context

Internal RAG for team over markdown in **one Git repository**. Much company knowledge lives in Confluence, Google Docs, SharePoint, PDF archives — not Git. Not solving multi-format or multi-repo ingestion this rollout.

Earlier ADRs used Wiki.js + custom chat + Vertex AI RAG Engine as production path. Superseded by ADR 0006 and ADR 0009 for current build.

Current implementation: Astro for browse/chat + local FastAPI + ChromaDB RAG backend. RAG ingestion corpus under `backend/rag-docs/`. Astro-rendered markdown may exist under `frontend/app/src/pages/posts/` — treat as presentation content or copied build input unless single shared content pipeline replaces it.

## Decision

All ingestible documents this rollout: **markdown files in single Git repository**. Active RAG corpus: `backend/rag-docs/`.

If same source rendered by Astro under `frontend/app/src/pages/posts/`, `backend/rag-docs/` stays authoritative RAG path until follow-up defines one shared markdown location for browse + retrieval.

Will not build this rollout:

- Confluence-to-markdown export or sync
- Native Confluence or other third-party datastore connectors
- PDF, DOCX, Google Docs, or other non-markdown loaders
- Additional Git repositories or cross-repo discovery

Ingestion assumes `.md` files with stable branch, path, commit metadata in **that one repo**. v1 access control bounded by repository permissions; won't sync Confluence or other source-system ACLs.

## Reasons

### Matches The Local RAG Ingestion Path

Git-backed markdown loads, chunks, embeds, indexes locally via FastAPI/ChromaDB RAG backend (ADR 0009). One repo keeps source control, citations, future build automation simple.

### Keeps Metadata Simple

Path, branch, commit, document, section metadata straightforward when source of truth is one Git tree. Format conversion adds page IDs, export timestamps, citation ambiguity we don't need for v1.

### Browse And RAG Should Share One Corpus

Astro browse pages and RAG ingestion should converge on same repo/paths so rendered docs and retrieved citations stay aligned.

### Avoids Export Quality Risk

Confluence macros, diagrams, internal links don't convert cleanly to markdown without ongoing maintenance. Defer conversion rather than ship brittle export pipeline before core workflow proven.

## Consequences

Team treats **one Git repository** as publishing surface for all RAG-indexed documentation. Content only in Confluence or other systems out of scope this rollout.

Ingestion stays markdown-only, single-repo. No Confluence connectors, export jobs, or multi-format loaders in v1.

Post-rollout options (Confluence connectors, other formats) may reconsider later; out of scope now, not permanently rejected.

## Alternatives Considered

### Confluence Export Or Sync Pipeline

Deferred — export quality + sync state before Git + RAG workflow proven for team.

### Native Cloud Connectors (Confluence via Vertex)

Deferred — connector setup + ACL modeling add scope. Git-repo permissions sufficient for v1.

### Multi-Format Loaders (PDF, DOCX, Google Docs)

Deferred — initial rollout assumes markdown in single Git repo only.

### Multiple Git Repositories

Out of scope this rollout. Shipping one repo for team documentation and RAG, not multi-repo discovery product.
