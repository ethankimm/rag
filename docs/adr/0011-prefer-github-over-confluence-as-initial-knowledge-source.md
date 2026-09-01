# 0011. Prefer GitHub Over Confluence as the Initial Knowledge Source

## Status

Accepted

## Context

Evaluated Confluence as primary corpus for local RAG prototype (HDTC space exports, recent-activity sampling). Confluence holds large share of existing team writing, but operational and evaluation problems showed up quickly:

- Many pages are informal **versions of each other** (copies, "v2" pages, overlapping runbooks) without single canonical lineage easy to detect or deduplicate.
- Measuring retrieval quality and **cleaning knowledge base consistently** hard when near-duplicates and stale forks coexist under different titles and spaces.
- Adoption uneven: some knowledge never lands in Confluence, so syncing Confluence alone doesn't guarantee coverage.
- Want to test RAG over material faithful to **active code repositories** — READMEs, ADRs, runbooks next to systems they describe.

Much tribal knowledge is **why** knowledge (intent, tradeoffs, history). Answering those well often pushes toward stronger reasoning models (e.g. o-series). Adds hidden intermediate reasoning that makes **faithfulness and validity harder to measure** in early prototype. Intentionally scope product to **how** questions: procedures, configuration, behavior grounded in explicit documents and code-adjacent markdown.

ADR 0004 already committed to git-backed markdown, deferred Confluence connectors. This ADR records stronger product decision after Confluence trials: **GitHub (git-backed repos) is primary knowledge source for current phase**; Confluence remains later integration candidate, not default corpus.

## Decision

For current rollout and evaluation phase:

1. **Primary source:** documentation and knowledge in **GitHub repositories** (markdown and other text versioned with code), ingested through existing git/markdown path (ADR 0004, ADR 0009).
2. **Defer Confluence** as first-class indexed source until clear strategy for canonicalization, version/dedup, and freshness.
3. **Question scope:** optimize and evaluate for **how** questions grounded in retrieved docs. Don't design v1 stack around open-ended **why** tribal-knowledge answers depending on opaque long-chain reasoning.
4. **LLM-generated wiki content** summarizing or explaining repos may experiment later, but only with explicit grounding to underlying repo docs/code. Standalone LLM-written knowledge base without that link out of scope this phase — faithfulness to code hard to trust and hard to score.

Confluence (and other wiki systems) may integrate later once GitHub-sourced retrieval quality, citation behavior, and KB hygiene proven.

## Reasons

### Versioning and Canonical Documents

Git gives commit history, paths, PRs as version model. Confluence page trees often contain parallel "versions" of same procedure without clean canonical pointer — pollutes retrieval, makes systematic cleanup expensive.

### Faithfulness to Active Systems

GitHub docs sit next to active repositories. Supports evaluating whether answers match how system works today. Confluence pages drift when teams stop updating; measuring drift at scale harder.

### Measurable "How" Before Opaque "Why"

How-questions checkable against retrieved passages and repo docs. Why-questions tend to require model-side reasoning beyond corpus — complicates faithfulness metrics for early enterprise prototype.

### Adoption and Publishing Path

Engineering teams already publish to GitHub. Relying on Confluence first adds dependency on surface many contributors underuse, plus export/macro quality risk already noted in ADR 0004.

### Smaller Integration Surface Now

Staying on git/markdown avoids Confluence sync, ACL mirroring, dedup pipelines before core RAG loop solid. Can add later without blocking current learning.

## Consequences

- Ingestion and evaluation focus on **GitHub-hosted markdown** (and closely related repo text), not live Confluence spaces.
- Confluence experiment scripts and exports may remain for research, but **not** target production corpus this phase.
- Product success metrics emphasize grounded **how** answers with citations, not open-ended tribal **why** explanations.
- Non-Git authors and Confluence-only knowledge out of scope until later ADR revisits connectors and canonicalization.
- This ADR **extends** ADR 0004: git-backed markdown remains format; preferred *organizational* source is GitHub repos over Confluence wiki spaces.

## Alternatives Considered

### Confluence as Primary Corpus

Rejected this phase — version/dedup noise, uneven adoption, weak alignment to active code make measurement and KB hygiene too costly before core RAG path proven.

### Hybrid Confluence + GitHub from Day One

Deferred — doubles connector and cleanup work. Prefer proving GitHub-sourced how-QA first, then add Confluence with explicit canonicalization rules.

### LLM-Generated Knowledge Base as Source of Truth

Deferred / rejected for v1 — generated pages can describe "how" without staying faithful to code, often omit measurable grounding. If revisited, generation must cite and track underlying repo sources.

### Optimize for Why-Questions with Strong Reasoning Models

Deferred — may be valuable later for tribal knowledge, but increases opacity of faithfulness evaluation. Current phase stays on grounded how-questions.

## Relationship to Other ADRs

- **ADR 0004** — Still accepted; this ADR narrows *source system* preference (GitHub over Confluence) and adds question-scope guidance.
- **ADR 0005** — Wiki-LLM ideas remain evaluation-only; any generated corpus must stay grounded to GitHub sources if pursued.
- **ADR 0008 / 0009** — Query prep and local FastAPI/Chroma stack continue; consume GitHub markdown corpus, not Confluence sync.
- **ADR 0010** — Sibling decision recorded separately; this ADR does not supersede it.
