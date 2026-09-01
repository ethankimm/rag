# 0012. Use PostgreSQL for Structured Backend Data

## Status

Accepted

## Context

Current local RAG prototype uses purpose-built local stores:

- ChromaDB for dense vector retrieval.
- `bm25s` for local keyword retrieval.
- Local chunk store for canonical chunk text and metadata during retrieval hydration.
- Git-backed markdown as source of truth for knowledge content.

As backend grows beyond prototype loop, needs durable structured data that is not retrieval index: users, teams, permissions, source records, ingestion runs, evaluation sets, feedback, audit events, operational state. Needs transactional guarantees, migrations, constraints, queryability, backup/restore workflows, well understood operational path.

SQL database choice should be explicit before structured backend state spreads across ad hoc local files, SQLite databases, JSON blobs, or retrieval-store metadata.

## Decision

Use **PostgreSQL** as backend SQL database for durable structured data.

PostgreSQL = system of record for relational backend state once introduced. Retrieval-specific systems remain separate:

1. **Vector retrieval** continues ChromaDB while ADR 0009 active.
2. **Keyword retrieval** continues `bm25s` while ADR 0010 active.
3. **Knowledge source content** continues from Git-backed markdown while ADR 0004 and ADR 0011 active.
4. **Chunk text and metadata** may stay in local chunk store during prototype. If chunk records become shared, multi-user, or operationally durable backend data, migrate into PostgreSQL through explicit schema rather than duplicating payloads inside retrieval indexes.

Don't use retrieval stores as primary database for structured application state.

## Reasons

### Strong Default for Enterprise Backend State

PostgreSQL: transactions, constraints, indexes, migrations, roles, backup/restore support, observability hooks, broad operational familiarity. Matters more for backend state than lightest possible local file format.

### Keeps Retrieval and Application State Separate

ChromaDB and `bm25s` are indexes optimized for search. Should identify chunks and support retrieval, not become source-of-truth stores for users, ingestion history, permissions, evaluations, audit records.

### Avoids SQLite Sprawl

SQLite useful for local prototype state and focused embedded stores, but multiple unrelated SQLite files make ownership, migrations, testing, deployment harder once backend becomes real service.

### Preserves a Clean Migration Path

Choosing PostgreSQL early lets future schema work start with durable application boundaries: explicit tables, migrations, foreign keys, typed repository/service layers.

## Consequences

- Future structured backend features add PostgreSQL-backed schemas and migrations instead of storing durable state in Chroma metadata, BM25 corpus payloads, loose JSON files, or unrelated SQLite databases.
- Local development needs PostgreSQL path once structured state implemented, likely Docker or documented local service.
- Tests use isolated temporary databases or repository-level fakes where appropriate; don't depend on shared developer data.
- Current retrieval stores remain valid — indexes, not backend SQL system of record.
- This ADR does not require adding PostgreSQL dependencies until first backend feature needs structured durable data.

## Alternatives Considered

### SQLite as the Main Backend Database

Deferred for application state. SQLite excellent for local tools and embedded prototype storage, but PostgreSQL better default for service needing concurrency, migrations, access control, observability, production operations.

### ChromaDB Metadata as the Structured Store

Rejected. ChromaDB metadata attached to vector index entries, not general relational system of record. Using for application state blurs retrieval concerns with backend domain state.

### JSON Files in the Repository

Rejected for durable backend state. Git-backed files appropriate for source knowledge documents and ADRs, but runtime state (ingestion runs, evaluations, feedback, audit records) needs database semantics.

### Managed NoSQL Store

Deferred. Document store may be useful for specific future workloads, but should not replace PostgreSQL as default relational backend database without separate ADR.

## Relationship to Other ADRs

- **ADR 0004 / ADR 0011** — GitHub-sourced markdown remains knowledge source; PostgreSQL for backend structured state, not authored documentation.
- **ADR 0009** — FastAPI remains backend boundary. ChromaDB remains current vector index.
- **ADR 0010** — `bm25s` remains current keyword index. PostgreSQL does not replace BM25 scoring or sparse retrieval.
