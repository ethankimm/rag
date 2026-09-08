"""Unit tests for typed hybrid fusion and reranking."""

from __future__ import annotations

from typing import cast

from backend.retrieval.hybrid_retriever import rerank, rrf
from backend.retrieval.model_client import ModelClient, RerankScore
from backend.storage.pg_search_store import SearchHit


def hit(chunk_id: str) -> SearchHit:
    return SearchHit(chunk_id, f"content {chunk_id}", {}, 0.0)


def test_rrf_fusion_ordering() -> None:
    semantic = [hit("doc_a"), hit("doc_b"), hit("doc_c")]
    keyword = [hit("doc_b"), hit("doc_d"), hit("doc_a")]

    fused = rrf(semantic, keyword)

    assert [result.chunk_id for result in fused[:2]] == ["doc_b", "doc_a"]
    assert {result.chunk_id for result in fused} == {
        "doc_a",
        "doc_b",
        "doc_c",
        "doc_d",
    }


def test_rerank_uses_every_cross_encoder_score() -> None:
    candidates = [hit("doc_1"), hit("doc_2"), hit("doc_3")]

    class FakeModelClient:
        received_documents: list[str] = []

        def rerank(self, query: str, documents: list[str]) -> list[RerankScore]:
            assert query == "query"
            self.received_documents = documents
            return [RerankScore(2, 4.0), RerankScore(1, 1.0), RerankScore(0, -2.0)]

    model_client = FakeModelClient()
    ranked = rerank(
        "query",
        candidates,
        cast(ModelClient, model_client),
        top_k=3,
    )

    assert model_client.received_documents == [result.content for result in candidates]
    assert [result.chunk_id for result in ranked] == ["doc_3", "doc_2", "doc_1"]
    assert [result.score for result in ranked] == [4.0, 1.0, -2.0]


def test_rerank_preserves_unscored_fused_tail() -> None:
    candidates = [hit("doc_1"), hit("doc_2"), hit("doc_3")]

    class FakeModelClient:
        @staticmethod
        def rerank(query: str, documents: list[str]) -> list[RerankScore]:
            del query, documents
            return [RerankScore(1, 1.0), RerankScore(0, 0.0)]

    ranked = rerank(
        "query",
        candidates,
        cast(ModelClient, FakeModelClient()),
        top_k=3,
        rerank_limit=2,
    )

    assert [result.chunk_id for result in ranked] == ["doc_2", "doc_1", "doc_3"]
