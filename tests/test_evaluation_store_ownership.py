"""Evaluation builders operate only on caller-owned search stores."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.documents import Document

from backend.evaluation.beir_scifact import build_pg_index, build_pg_lexical_index
from backend.retrieval.model_client import ModelClient
from backend.storage.pg_search_store import PgSearchStore


class _CallerOwnedStore:
    def __init__(self, *, embedding_dimension: int | None) -> None:
        self.collection_name = "caller_owned"
        self.embedding_dimension = embedding_dimension
        self.closed = False

    @property
    def has_embeddings(self) -> bool:
        return self.embedding_dimension is not None

    def __enter__(self) -> _CallerOwnedStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def can_reuse(self, _manifest: object) -> bool:
        return False

    def replace_documents(self, *_args: object) -> None:
        raise RuntimeError("replacement failed")


class _FailingModelClient:
    config = SimpleNamespace(embedding_dimension=3, embedding_model="test-bge")

    def embed_documents(self, *_args: object, **_kwargs: object) -> list[list[float]]:
        raise RuntimeError("embedding failed")


def test_hybrid_builder_failure_is_closed_by_caller() -> None:
    store = _CallerOwnedStore(embedding_dimension=3)
    documents = [Document(page_content="document", metadata={"chunk_id": "1"})]

    with pytest.raises(RuntimeError, match="embedding failed"), store:
        build_pg_index(
            documents,
            cast(ModelClient, _FailingModelClient()),
            cast(PgSearchStore, store),
        )

    assert store.closed


def test_lexical_builder_failure_is_closed_by_caller() -> None:
    store = _CallerOwnedStore(embedding_dimension=None)
    documents = [Document(page_content="document", metadata={"chunk_id": "1"})]

    with pytest.raises(RuntimeError, match="replacement failed"), store:
        build_pg_lexical_index(documents, cast(PgSearchStore, store))

    assert store.closed
