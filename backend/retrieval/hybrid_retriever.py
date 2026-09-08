"""Dense + BM25 retrieval, reciprocal-rank fusion, and cross-encoder reranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from backend.retrieval.model_client import (
    DEFAULT_MODEL_CONFIG,
    ModelClient,
    validate_embedding_vector,
)
from backend.storage.pg_search_store import PgSearchStore, SearchHit

STAGE1_FETCH_LIMIT = 50
RERANK_CANDIDATE_LIMIT = 50
RRF_K = 60

# Compatibility exports for result metadata and existing callers.
EMBEDDING_MODEL = DEFAULT_MODEL_CONFIG.embedding_model
RERANK_MODEL = DEFAULT_MODEL_CONFIG.reranking_model


def rrf(
    semantic_results: Sequence[SearchHit],
    keyword_results: Sequence[SearchHit],
) -> list[SearchHit]:
    """Fuse two equal-weight ranked lists with reciprocal rank fusion."""
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    for ranked_results in (semantic_results, keyword_results):
        for rank, hit in enumerate(ranked_results, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            hits[hit.chunk_id] = hit
    return [
        hits[chunk_id]
        for chunk_id, _ in sorted(scores.items(), key=lambda pair: -pair[1])
    ]


def rerank(
    query: str,
    results: Sequence[SearchHit],
    model_client: ModelClient,
    *,
    top_k: int,
    rerank_limit: int = RERANK_CANDIDATE_LIMIT,
) -> list[SearchHit]:
    """Cross-encode the candidate window and preserve the remaining fused tail."""
    if not results or top_k <= 0:
        return []
    if rerank_limit <= 0:
        return list(results[:top_k])

    candidates = list(results[:rerank_limit])
    scores = model_client.rerank(query, [hit.content for hit in candidates])
    # Preserve the cross-encoder relevance score on the returned hit. Downstream
    # answer gating needs this score; the stage-one dense and lexical scores are
    # not comparable enough to use as a single confidence signal.
    reranked = [
        replace(
            candidates[score.document_index],
            score=score.relevance_score,
        )
        for score in scores
    ]
    return (reranked + list(results[len(candidates) :]))[:top_k]


class HybridRetriever:
    """Orchestrate typed model and PostgreSQL search boundaries."""

    def __init__(self, store: PgSearchStore, model_client: ModelClient) -> None:
        if store.embedding_dimension != model_client.config.embedding_dimension:
            raise ValueError(
                "Store and model-client embedding dimensions must match: "
                f"{store.embedding_dimension} != "
                f"{model_client.config.embedding_dimension}"
            )
        self.store = store
        self.model_client = model_client

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        stage1_limit: int = STAGE1_FETCH_LIMIT,
        rerank_limit: int = RERANK_CANDIDATE_LIMIT,
    ) -> list[SearchHit]:
        """Retrieve, fuse, and rerank documents for one query."""
        query_embedding = self.model_client.embed_query(query)
        fused = rrf(
            self.store.dense_search(query_embedding, limit=stage1_limit),
            self.store.bm25_search(query, limit=stage1_limit),
        )
        return rerank(
            query,
            fused,
            self.model_client,
            top_k=top_k,
            rerank_limit=rerank_limit,
        )


__all__ = [
    "EMBEDDING_MODEL",
    "RERANK_MODEL",
    "HybridRetriever",
    "rerank",
    "rrf",
    "validate_embedding_vector",
]
