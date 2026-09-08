# 0005. Evaluate Wiki LLM vs Vanilla RAG

## Status

Proposed

## Context

Current accepted build: **vanilla RAG** over Git-backed markdown:

- Astro for browse and chat (ADR 0006)
- FastAPI, ChromaDB, local GGUF embeddings, OpenAI-compatible generation (ADR 0009)
- Single Git repository as document source of truth (ADR 0004)

Evaluating alternative inspired by
[Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
and OpenWiki-style repo wikis.

Records comparison and possible **Wiki LLM** direction. Does **not** supersede ADRs 0004, 0006, or 0009 until explicit accept + update those records.

### Why we are considering this shift

Main driver: planned **agent-first architecture**. Vanilla RAG optimizes today's pattern: **human types prompt**, system retrieves chunks, LLM answers. Wiki LLM optimizes different pattern: **agent navigates repo**, ingests/compiles knowledge into maintained wiki, **builds own context** before acting — Q&A, code edits, doc updates.

Building vanilla RAG now risks anchoring product to human-prompts-AI use case. Wiki LLM aligns with expected team work: agents read `AGENTS.md`, browse compiled pages, ingest sources, maintain knowledge layer without assuming humans keep raw docs perfect.

Vanilla RAG still viable, **easier to build**: Git markdown → local ingestion → ChromaDB retrieval → FastAPI chat already in repo. Pattern **well documented**, clear local libraries + managed service options later. Long-term problem: **doc maintenance**. Architecture assumes humans keep markdown current, structured, cross-linked in Git. In practice docs drift, go stale, scatter across READMEs and Confluence exports, nobody wants bookkeeping.

Wiki LLM **much harder to build**. Pattern still **experimental** at org scale: ingest workflows, compiled `wiki/` layer, lint passes, agent-driven updates add scope; must govern **agent drift** (wiki pages over-synthesize, contradict `raw/` sources, go stale unnoticed). If controls work, upkeep shifts to agent (or ingest pipeline) that summarizes, links, refreshes when sources change — not authors keeping corpus retrieval-ready.

### Vanilla RAG (current plan)

Query time: embed question, retrieve top-k **chunks** from raw markdown via vector search, LLM synthesizes answer with citations. Cross-document connection/synthesis on **every question**. Knowledge base = raw docs + search index; doesn't compound from Q&A unless added separately.

```
Question → embed → retrieve chunks from corpus → LLM answer + citations
```

### Wiki LLM (proposed alternative)

System maintains **compiled wiki layer** in Git: `AGENTS.md` (schema/rules), `index.md` (catalog), topic pages (synthesized, cross-linked), optionally `log.md`. Query time **agent** navigates wiki — read `AGENTS.md`, open `index.md`, follow links, optional search tools — synthesizes answer with citations. Chat service hosts agent; employees use normal web UI, not separate agent IDE. Synthesis primarily at **ingest and maintenance**. Good answers filed back into wiki so knowledge compounds.

```
Question → agent reads index.md → opens wiki pages → LLM answer + citations
         → (optional) file answer back into wiki
```

## Decision

**Defer commitment.** Continue building against ADRs 0004, 0006, 0009 while evaluating Wiki LLM as possible pivot.

Working hypothesis:

1. **Agent-first is strategic direction.** Wiki LLM fits agents that navigate, ingest, compile repo knowledge; vanilla RAG fits humans prompting chat UI over static corpus.
2. **Build vs maintenance tradeoff.** Vanilla RAG faster to ship — well documented, mature local libraries, possible managed services later — but depends on sustained human doc hygiene. Wiki LLM costs more to build, still experimental, requires governing agent drift; reduces reliance on people managing docs if controls work.
3. **Vanilla RAG lower-risk v1** if we accept short-term human maintenance and revisit Wiki LLM when doc drift or agent-first workflows become painful.
4. **Patterns can hybridize**: human docs in Git, LLM-maintained wiki pages, optional dedicated retrieval when index-browse stops scaling (Karpathy suggests ~100 sources and hundreds of pages before adding dedicated search).

No change to accepted ADRs until deliberate review amends or supersedes ADRs 0004, 0006, 0009.

## Reasons to consider Wiki LLM

### Agent-first architecture

Expect more work through agents (coding, Q&A, ingest, maintenance), not only humans typing questions into chat. Wiki LLM gives agents durable, navigable knowledge layer (`AGENTS.md`, `index.md`, topic pages) to read, update, extend. Vanilla RAG gives agents same raw chunks, repeats synthesis every query; doesn't model agent that **builds and owns context** over time.

### Long-term maintenance burden (vanilla RAG)

Vanilla RAG easier to build, but long-term cost = **ongoing human upkeep**. ADR 0004 and vanilla RAG path treat markdown in Git as source of truth — workable target, not reliable operating assumption. Teams leave docs outdated, split knowledge, skip cross-links. Wiki LLM assigns compilation/upkeep to agent (ingest, lint, cross-reference updates) instead of hoping authors keep corpus retrieval-ready. Humans curate sources and review; LLM does bookkeeping.

### Same chat UX, agent-driven retrieval

Employees don't need Cursor or separate agent IDE. Chat app hosts agent navigating `index.md` and wiki pages with read/search tools instead of `retrieval_query` over pre-fetched chunks.

### Compounding knowledge

Ingest, lint, filing good answers back into wiki build organizational memory. Vanilla RAG typically doesn't update corpus from chat unless added explicitly.

### Aligns with repo-agent workflows

`AGENTS.md` pointing agents at compiled wiki pages before source code matches how coding agents (Cursor, Codex, Claude Code) work. OpenWiki and Cursor-generated wikis practical bootstrap for this layer.

## Reasons to keep vanilla RAG (or hybrid)

### Scale and paraphrase

Index-browse works at moderate scale. Semantic search over large corpus handles paraphrased questions and facts buried across many files. ChromaDB retrieval (ADR 0009) addresses this when browse alone fails.

### Source of truth and trust

Wiki LLM adds LLM-maintained layer that may **drift** — over-synthesize, contradict authoritative `raw/` sources, miss updates — unless we enforce `raw/` (immutable) vs `wiki/` (maintained), run lint passes, review agent edits. **Agent drift** first-class operational concern. Explicit maintenance with agent tooling vs vanilla RAG's hidden risk: **stale or messy human docs silently degrade retrieval** while we assume corpus is fine.

### Easier initial build (vanilla RAG)

Vanilla RAG has **clear, well-documented build path**. Local markdown loading, chunking, embedding, ChromaDB retrieval, OpenAI-compatible generation = premade pieces for ingest, retrieval, chat UI. Simplicity externalizes doc freshness to authors. Wiki LLM requires more custom, **experimental** infrastructure (ingest, lint, human review, drift detection), fewer off-the-shelf packages, but maintenance **explicit** rather than hoping corpus stays clean.

### Narrow factual questions

"What is the Vault path for prod?" may not need synthesized wiki page; chunk retrieval over authoritative docs can suffice.

## Consequences

### If we stay on vanilla RAG

No change to ADRs 0004, 0006, 0009. Optional: index OpenWiki-style pages in same corpus so RAG retrieves synthesized guides + human docs.

### If we pivot to Wiki LLM

Likely impacts:

- **ADR 0009** — ChromaDB/vector retrieval may be **deferred** until wiki search (e.g. qmd, BM25/vector over markdown) proves insufficient; not necessarily eliminated forever.
- **ADR 0004** — Still Git-backed markdown; may add `wiki/` vs `raw/` layout and LLM maintenance rules in `AGENTS.md`.

Individual ADRs keep original text. Superseded ones get updated **Status** pointing here; root README and project guidance become current map of what we build.

## Alternatives Considered

### Stay on vanilla RAG only

Keep ADRs 0004, 0006, 0009 accepted. Simplest path to cited Q&A over human docs: well-documented local RAG stack, mature libraries, possible managed services later, if we accept human-prompts-chat use case and ongoing human doc maintenance for v1. Revisit Wiki LLM when maintenance pain or agent-first workflows justify higher build cost and experimental governance overhead.

### Wiki LLM only (no RAG, no Wiki.js)

Single Git wiki maintained by agent; employees use chat hosting same agent pattern for Q&A. Higher build and governance cost (experimental pattern, agent drift); higher risk on freshness, trust, scale without human authoring surface.

### Wiki LLM for agents, vanilla RAG for everyone else

Repo-level `AGENTS.md` + OpenWiki for coding agents; team chat stays on local RAG over shared markdown. Two retrieval paths, one Git corpus. Lower pivot risk; doesn't unify into single Wiki LLM product experience. High long-term overhead monitoring both human-authored and AI-assisted docs.
