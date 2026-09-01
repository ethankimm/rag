# 0008. Use Query Normalization and Retrieval Rewrite

## Status

Proposed

## Context

RAG pipeline depends on retrieval quality before generator produces good answer. Embedding models and chat completion providers do own tokenization and low-level text handling, but that preprocessing doesn't make retrieval robust alone.

Retrieval fails when same concept represented differently in source docs and user query. Misspelled words in docs or query produce different token sequences and vector representations. Relevant chunks may score lower or fail to appear even when knowledge base contains answer.

Same issue with inconsistent terminology, acronyms, aliases, domain-specific synonyms. Improve retrieval matching without blindly rewriting terms that may be meaningful: code identifiers, file paths, product names, internal names.

Records need for retrieval query preparation layer. Does not choose final rewrite or search implementation yet.

HDTC Confluence corpus test showed single-team Confluence ingestion and embedding feasible for local prototype. Indexing one team space into ChromaDB did not expose latency or embedding throughput as main bottleneck.

Larger issue: retrieval quality. User questions often vague, conversational, acronym-heavy, or phrased differently from Confluence page titles and body text. Relevant chunks may exist in vector store, but current single-query vector search + fixed similarity threshold can fail to surface them. Manual chat testing not enough to validate whether system retrieves right evidence or produces trustworthy answers.

Moves retrieval improvement and evaluation from optional polish to required prototype work before broader rollout.

## Decision

Preserve original user query as authoritative, while allowing system to derive separate retrieval-oriented query representation before embedding or search.

Committed direction intentionally small:

1. Original user query retained for answer generation, debugging, auditability.
2. Retrieval may use normalized, rewritten, expanded, or otherwise derived query when that improves recall.
3. Any derived retrieval query = optimization, not replacement for user's original request.
4. Retrieval becomes multi-stage process, not single vector lookup.
5. System retrieves broader candidate set before filtering.
6. System combines semantic search with exact-term or metadata-aware search for page titles, acronyms, labels, paths, modified dates.
7. Candidate chunks reranked before assembling final context.
8. First implementation stays local and lightweight, consistent with current prototype.

Not yet committing to simple normalization, LLM rewrite, keyword expansion, hybrid search, reranking, or combination.

## Candidate Approaches

### Light normalization

Deterministic cleanup: trim whitespace, normalize repeated spaces, handle copied text artifacts.

Low risk, easy to test, won't solve vague wording, synonyms, or multi-part questions.

### LLM query rewrite

Ask model for concise retrieval query preserving original intent while clarifying search terms.

May improve recall for messy natural-language questions, but adds latency and can distort intent if rewrite not constrained.

### Retrieve with original and rewritten queries

Run retrieval against both original and derived retrieval query, merge and deduplicate results.

Reduces risk of losing original intent, but increases retrieval work and requires clear result merging behavior.

### Hybrid vector and keyword search

Combine semantic vector retrieval with keyword or BM25-style search for exact terms, code identifiers, filenames, acronyms.

Often stronger for enterprise knowledge bases, but adds another retrieval path to tune and observe.

### Reranking

Retrieve broader candidate set first, rerank before assembling context for generation.

Can improve context quality, but likely after basic query preparation and retrieval behavior is observable.

## Output Constraint Experiment

Also explored constrained decoding for more predictable generated answers. Initial experiment used Outlines with Pydantic-style schema to force local `llama.cpp` model to return structured response such as:

```json
{ "answer": "..." }
```

Remains useful optional direction — structured output can reduce response boilerplate, simplify frontend handling, give backend clearer contract to validate. Experiment did not produce ideal answer quality alone, and `llama.cpp` constrained decoding not part of required generation path in ADR 0009.

Likely reason: constrained decoding controls output shape, not poor input context. If retrieved chunks contain raw markdown, tutorial navigation, links, unrelated examples, noisy boilerplate, or poorly bounded chunk content, model may still copy or summarize inside `answer` field. Weaker local model may continue retrieved document style instead of clean direct answer.

Output constraints = one layer of system, not replacement for retrieval and context quality work. Continue exploring Pydantic schemas and grammar-constrained generation while evaluating ways to clean and prepare context before passing to model.

## Consequences

### Expected benefits

- Better retrieval recall when user wording doesn't match corpus exactly.
- More resilience to typos, acronyms, multi-part questions.
- Clearer observability: compare original query, retrieval query, retrieved chunks, final answer.
- Cleaner place for future enterprise controls: query logging, privacy filtering, policy checks, search strategy selection.

### Costs and risks

- Rewriting can change user intent if too aggressive.
- Spell correction can damage internal terms, code symbols, filenames, customer names, product names.
- LLM-based rewrite adds latency and another model behavior to test.
- Hybrid search and reranking add implementation and tuning complexity.
- Structured output can make response contract cleaner but may hide noisy or low-quality answer content inside valid schema.
- Overbuilding this layer too early conflicts with incremental local-prototype direction.

## Relationship to ADR 0007

ADR 0007 covers query routing and generation budgeting: answer shape, prompt behavior, output token budget.

This ADR covers retrieval query preparation: prepare user's question for embedding or search before context assembled.

Two decisions work together but are separate concerns.

## Follow-up Work

Keep first implementation small. Reasonable next step: preserve both `original_query` and `retrieval_query` in retrieval flow, start with light normalization or constrained rewrite, log enough to evaluate whether retrieval quality improves before adding hybrid search or reranking.

Manual chat testing not sufficient for this prototype. Before treating retrieval quality acceptable, team should build small evaluation set with input from HDTC subject-matter experts.

Benchmark should include representative user questions, expected source pages or acceptable source paths, expected answer facts where known, required citation metadata, negative questions where system should say it does not know, latency measurements for retrieval and generation.

Track retrieval metrics: recall@k, MRR, source hit rate, citation accuracy. Track answer metrics: faithfulness to retrieved context, helpfulness, SME approval. Record p50 and p95 latency, but don't optimize latency ahead of retrieval correctness while corpus still small.

Additional follow-up: cleaning markdown before indexing or generation, removing navigation and repeated boilerplate from retrieved context, improving chunk boundaries, comparing constrained decoding with prompt-only formatting, post-processing, reranking, stricter context assembly.
