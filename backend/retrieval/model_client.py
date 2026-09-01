"""Typed, reusable client for local embedding and reranking models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ModelConfig:
    """Complete model contract shared by ingestion, retrieval, and evaluation."""

    base_url: str = "http://localhost:8080"
    embedding_model: str = "bge-m3-q8_0"
    reranking_model: str = "bge-reranker-v2-m3-Q8_0"
    embedding_dimension: int = 1024
    query_prefix: str = ""
    document_prefix: str = ""
    max_rerank_characters: int = 4096


@dataclass(frozen=True)
class RerankScore:
    """A cross-encoder score associated with one input document."""

    document_index: int
    relevance_score: float


DEFAULT_MODEL_CONFIG = ModelConfig()


def validate_embedding_vector(
    raw_embedding: Any,
    *,
    expected_dimension: int,
    context: str,
) -> list[float]:
    """Validate a model response before using it as a dense vector."""
    if not isinstance(raw_embedding, list):
        raise RuntimeError(f"{context} response is not a vector")

    try:
        embedding = [float(value) for value in raw_embedding]
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"{context} response contains a non-numeric value"
        ) from error

    if len(embedding) != expected_dimension:
        raise RuntimeError(
            f"{context} response has dimension {len(embedding)}; "
            f"expected {expected_dimension}"
        )
    if not all(math.isfinite(value) for value in embedding):
        raise RuntimeError(f"{context} response contains a non-finite value")
    if math.sqrt(sum(value * value for value in embedding)) <= 1e-12:
        raise RuntimeError(
            f"{context} response is all zeros; check llama.cpp pooling settings"
        )
    return embedding


class ModelClient:
    """Own llama.cpp HTTP transport, response validation, and model selection."""

    def __init__(
        self,
        config: ModelConfig = DEFAULT_MODEL_CONFIG,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self._http = httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(120.0),
            transport=transport,
        )

    def __enter__(self) -> ModelClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def embed_query(self, query: str) -> list[float]:
        """Embed one query using the configured query transformation."""
        vectors = self._embed([f"{self.config.query_prefix}{query}"], "query embedding")
        return vectors[0]

    def embed_documents(
        self,
        documents: Sequence[str],
        *,
        batch_size: int = 16,
    ) -> list[list[float]]:
        """Embed documents in bounded batches through one reusable connection pool."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        vectors: list[list[float]] = []
        for start in range(0, len(documents), batch_size):
            batch = [
                f"{self.config.document_prefix}{document}"
                for document in documents[start : start + batch_size]
            ]
            vectors.extend(self._embed(batch, "document embedding"))
        return vectors

    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        """Cross-encode every document and return scores in descending order."""
        if not documents:
            return []

        candidates = [
            document[: self.config.max_rerank_characters] for document in documents
        ]
        payload = {
            "model": self.config.reranking_model,
            "query": query,
            "top_n": len(candidates),
            "documents": candidates,
        }
        response = self._post_with_fallback(("/rerank", "/v1/rerank"), payload)

        try:
            raw_results = response.json()["results"]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Reranking response has an invalid schema") from error
        if not isinstance(raw_results, list):
            raise RuntimeError("Reranking response 'results' must be a list")

        scores: list[RerankScore] = []
        seen_indexes: set[int] = set()
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                raise RuntimeError("Reranking response contains a non-object result")

            index = raw_result.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise RuntimeError("Reranking result has an invalid document index")
            if index < 0 or index >= len(candidates) or index in seen_indexes:
                raise RuntimeError(
                    "Reranking result has an out-of-range or duplicate index"
                )

            try:
                relevance_score = float(raw_result["relevance_score"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "Reranking result has an invalid relevance score"
                ) from error
            if not math.isfinite(relevance_score):
                raise RuntimeError("Reranking result has a non-finite relevance score")

            seen_indexes.add(index)
            scores.append(RerankScore(index, relevance_score))

        if len(scores) != len(candidates):
            raise RuntimeError(
                "Reranker returned scores for "
                f"{len(scores)}/{len(candidates)} candidates"
            )
        return sorted(scores, key=lambda score: score.relevance_score, reverse=True)

    def _embed(self, inputs: list[str], context: str) -> list[list[float]]:
        if not inputs:
            return []

        response = self._post(
            "/v1/embeddings",
            {"model": self.config.embedding_model, "input": inputs},
        )
        try:
            raw_items = response.json()["data"]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Embedding response has an invalid schema") from error
        if not isinstance(raw_items, list):
            raise RuntimeError("Embedding response 'data' must be a list")

        try:
            items = sorted(raw_items, key=lambda item: item["index"])
            indexes = [item["index"] for item in items]
        except (KeyError, TypeError) as error:
            raise RuntimeError("Embedding response has an invalid schema") from error
        if indexes != list(range(len(inputs))):
            raise RuntimeError(
                f"Embedding server returned invalid indexes {indexes!r} "
                f"for {len(inputs)} inputs"
            )

        return [
            validate_embedding_vector(
                item.get("embedding"),
                expected_dimension=self.config.embedding_dimension,
                context=context,
            )
            for item in items
        ]

    def _post_with_fallback(
        self,
        paths: tuple[str, ...],
        payload: dict[str, object],
    ) -> httpx.Response:
        for path in paths:
            response = self._request(path, payload)
            if response.status_code != 404:
                return self._require_success(response, path)
        return self._require_success(response, paths[-1])

    def _post(self, path: str, payload: dict[str, object]) -> httpx.Response:
        return self._require_success(self._request(path, payload), path)

    def _request(self, path: str, payload: dict[str, object]) -> httpx.Response:
        try:
            return self._http.post(path, json=payload)
        except httpx.HTTPError as error:
            raise RuntimeError(f"Model request to {path} failed: {error}") from error

    @staticmethod
    def _require_success(response: httpx.Response, path: str) -> httpx.Response:
        if response.status_code != 200:
            raise RuntimeError(
                f"Model endpoint {path} rejected request "
                f"({response.status_code}): {response.text}"
            )
        return response
