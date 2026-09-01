"""Contract tests for the typed llama.cpp model client."""

from __future__ import annotations

import json

import httpx
import pytest

from backend.retrieval.model_client import ModelClient, ModelConfig


def test_model_client_uses_bge_models_and_rerank_fallback() -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path == "/v1/embeddings":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": index, "embedding": [1.0, float(index), 0.5]}
                        for index, _ in enumerate(payload["input"])
                    ]
                },
            )
        if request.url.path == "/rerank":
            return httpx.Response(404)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": -1.0},
                    {"index": 1, "relevance_score": 2.0},
                ]
            },
        )

    config = ModelConfig(embedding_dimension=3)
    with ModelClient(config, transport=httpx.MockTransport(handler)) as client:
        vectors = client.embed_documents(["first", "second"])
        scores = client.rerank("query", ["first", "second"])

    assert len(vectors) == 2
    assert [score.document_index for score in scores] == [1, 0]
    assert requests[0][1]["model"] == "bge-m3-q8_0"
    assert requests[1][0] == "/rerank"
    assert requests[2][0] == "/v1/rerank"
    assert requests[2][1]["model"] == "bge-reranker-v2-m3-Q8_0"
    assert requests[2][1]["top_n"] == 2


def test_model_client_rejects_partial_rerank_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"results": [{"index": 1, "relevance_score": 2.0}]},
        )

    with (
        ModelClient(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(RuntimeError, match="1/2 candidates"),
    ):
        client.rerank("query", ["first", "second"])


def test_model_client_rejects_zero_embedding() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.0, 0.0, 0.0]}]},
        )

    config = ModelConfig(embedding_dimension=3)
    with (
        ModelClient(config, transport=httpx.MockTransport(handler)) as client,
        pytest.raises(RuntimeError, match="all zeros"),
    ):
        client.embed_query("query")
