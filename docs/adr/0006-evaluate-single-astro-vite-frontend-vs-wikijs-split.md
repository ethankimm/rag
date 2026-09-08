# 0006. Use a Single Astro Frontend

## Status

Accepted

## Context

ADRs 0001 and 0002 originally used **split frontend**:

1. **Wiki.js** — write and browse markdown for non-GitHub authors; Git sync; Okta auth (ADR 0001)
2. **Next.js chat app** — RAG or agent Q&A with citations (ADR 0001, ADR 0002)

ADR 0005 added pressure to consolidate: agent-first workflows, browse-first UX with chat/agent secondary, Git as only source of truth without Wiki.js database layer.

Current implementation: **one markdown-friendly Astro site** instead of Wiki.js + Next.js. This ADR covers **frontend shape only**; retrieval and generation stack in ADR 0009.

### Historical split

```
Wiki.js (edit + browse)          Next.js (chat + citations)
         \                              /
          \____ same Git markdown ______/
```

Two apps, two deploys, two auth integrations, Wiki.js DB sync, path-to-URL mapping for citations.

### Accepted consolidation

```
Astro site (browse-first + chat panel)
         |
    Git markdown — CI clones repo, builds static site; no Wiki.js DB
```

One app, one URL space; citations link to pages on same origin.

## Decision

Use **Astro** as single frontend for current product direction.

Astro owns browse-first documentation surface and chat shell calling FastAPI backend. Not operating Wiki.js, separate Next.js chat app, or split frontend in current build.

If non-GitHub business authoring becomes hard requirement, evaluate Git-backed CMS on same Astro site before reintroducing separate wiki product.

## Reasons for a single site

### One surface, browse-first

Users read docs in primary UI; chat or agent panel secondary ("ask about this doc"). Same origin for citations — no Wiki.js URL mapping, no app switch. Auth, layout, navigation unified.

### Git-native without Wiki.js sync

Wiki.js keeps internal database, syncs to Git — second writer that can lag or conflict with direct Git edits. Single site built from CI (`Git →
build → deploy`) makes repo only source of truth (ADR 0004). Pages served from built artifact, not live GitHub API per request.

### Chat and agent UX belong in one codebase

ADR 0001 ruled RAG chat, source inspection, citation flows don't fit Wiki.js without heavy customization. Single app owns browse + chat shell hosting agent (ADR 0005) — whether retrieval is vector chunks or wiki navigation.

### Lighter ops

One framework, one deploy, one auth surface. Astro can host interactive islands later if chat UI outgrows plain browser JavaScript.

## Alternatives and residual concerns

### Non-GitHub authors

Wiki.js lets people without GitHub access edit via familiar UI with Okta. Astro + Git-only editing means more GitHub licenses, Git-backed CMS on site, or accepting PR-only authoring for business teams.

### Wiki.js is proven for wiki UX

Page history, permissions, editor UX built in. CMS-on-site repeats part of scope ADR 0001 deferred.

## Consequences

### Rejected split model

Don't continue Wiki.js + Next.js for current build. Adds second frontend, Wiki.js sync care, citation URL mapping before RAG workflow proven.

### Current accepted direction

- **ADR 0001** — Superseded; Wiki.js removed from current architecture.
- **ADR 0002** — Superseded; Next.js removed from current architecture.
- **ADR 0004** — Still accepted; Git-backed markdown remains source format.
- **ADR 0005** — Still proposed; single site compatible with agent-first or vanilla RAG retrieval patterns.
- **New work** — CMS and/or Okta can evaluate later for non-GitHub editors.
