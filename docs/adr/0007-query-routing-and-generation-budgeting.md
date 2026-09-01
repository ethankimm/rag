# 0007. Use Query Routing and Generation Budgeting

## Status

Proposed

## Context

Current local QA flow retrieves relevant markdown chunks, asks OpenAI-compatible chat completion provider to answer from context. Works well for narrow factual questions — short answer, retrieved context usually enough.

Generation step uses fixed output token budget. Too rigid for expected question range:

- Simple factual questions don't need large answer budget.
- Complex explanation, comparison, troubleshooting may need larger budget to avoid truncation.
- Single hard number makes quality, latency, resource usage harder to tune together.

Records need for routing and budgeting layer. Does not choose final routing implementation yet.

## Decision

Introduce query routing and generation budgeting step before answer generation.

Committed direction intentionally small:

1. System should not use one fixed output token budget for every question.
2. System should classify or route questions into small number of answer shapes before generation.
3. Chosen route should influence generation behavior — likely output token cap, prompt instructions, possibly retrieval depth.
4. First implementation stays local and lightweight, consistent with current prototype.

Not yet committing to rule-based, classifier-based, embedding-based, or hybrid routing.

## Candidate Approaches

### Simple rule-based routing

Deterministic heuristics: query length, keywords, question type, explicit user wording to choose simple, explanatory, or deep-answer budgets.

Easiest starting point, simple to test, may miss ambiguous queries.

### Lightweight classifier routing

Small classifier prompt or model call to label question intent and complexity before retrieval or generation.

May handle ambiguous queries better than rules, but adds latency, failure modes, another prompt surface to evaluate.

### Hybrid routing

Simple rules for obvious cases; classifier only when rule-based route uncertain.

May be best long-term shape, but more implementation work than current prototype needs immediately.

## Consequences

### Expected benefits

- Factual questions stay concise and fast.
- Complex questions get enough generation budget to finish naturally.
- Prompt style and retrieval depth dynamically adjusted by question type instead of fixed.
- Cleaner place for future enterprise controls: answer policy, citation requirements, model selection, escalation.

### Costs and risks

- Routing adds behavior to test and observe.
- Misclassification can produce answers too short, too verbose, or retrieved with wrong depth.
- Classifier-based approach may increase latency.

## Follow-up Work

Keep first implementation small. Reasonable next step: define few query routes (`factual`, `explanatory`, `deep`), test whether simple rule-based router enough before adding classifier.
