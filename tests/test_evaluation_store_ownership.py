"""Evaluation builders operate only on caller-owned search stores."""

from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.documents import Document

from backend.evaluation import __main__ as evaluation
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


@pytest.mark.parametrize("batch_failure", [None, RuntimeError("evaluation failed")])
def test_hybrid_main_uses_validated_runtime_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
    batch_failure: RuntimeError | None,
) -> None:
    lifecycle: list[str] = []
    model_client = SimpleNamespace(
        config=SimpleNamespace(
            embedding_model="runtime-embedding",
            reranking_model="runtime-reranker",
            embedding_dimension=384,
        )
    )
    store = SimpleNamespace(collection_name="runtime_collection")
    documents = [Document(page_content="document", metadata={"chunk_id": "1"})]
    args = Namespace(
        collection_name=None,
        lexical_only=False,
        data_root=Path("unused"),
        dsn="postgresql://unused",
        rebuild=False,
        rerank_depth=50,
        top_k=10,
    )

    @contextmanager
    def fake_local_model_runtime():
        lifecycle.append("models started")
        try:
            yield model_client
        finally:
            lifecycle.append("models closed")

    @contextmanager
    def fake_search_store(
        *,
        collection_name: str,
        embedding_dimension: int,
        dsn: str,
    ):
        assert collection_name == evaluation.DEFAULT_COLLECTION_NAME
        assert embedding_dimension == 384
        assert dsn == "postgresql://unused"
        lifecycle.append("store started")
        try:
            yield store
        finally:
            lifecycle.append("store closed")

    def fake_build_pg_index(
        received_documents: list[Document],
        received_client: object,
        received_store: object,
        *,
        rebuild: bool,
    ) -> None:
        assert received_documents is documents
        assert received_client is model_client
        assert received_store is store
        assert not rebuild
        lifecycle.append("batch started")
        if batch_failure is not None:
            raise batch_failure

    def fake_load_scifact(
        _path: Path,
    ) -> tuple[
        dict[str, dict[str, str]],
        dict[str, str],
        dict[str, dict[str, int]],
    ]:
        lifecycle.append("source loaded")
        return {"1": {}}, {"q1": "query"}, {"q1": {"1": 1}}

    monkeypatch.setattr(evaluation, "_parse_args", lambda: args)
    monkeypatch.setattr(
        evaluation,
        "load_scifact",
        fake_load_scifact,
    )
    monkeypatch.setattr(evaluation, "corpus_to_documents", lambda _corpus: documents)
    monkeypatch.setattr(evaluation, "local_model_runtime", fake_local_model_runtime)
    monkeypatch.setattr(evaluation, "PgSearchStore", fake_search_store)
    monkeypatch.setattr(evaluation, "build_pg_index", fake_build_pg_index)
    monkeypatch.setattr(
        evaluation,
        "HybridRetriever",
        lambda received_store, received_client: SimpleNamespace(
            store=received_store,
            model_client=received_client,
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "run_hybrid_pipeline",
        lambda *_args, **_kwargs: {"q1": {"1": 1.0}},
    )
    monkeypatch.setattr(evaluation, "evaluate", lambda *_args: {"Accuracy@10": 1.0})
    monkeypatch.setattr(evaluation, "_report_and_save", lambda *_args, **_kwargs: None)

    if batch_failure is None:
        evaluation.main()
    else:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            evaluation.main()

    assert lifecycle == [
        "models started",
        "store started",
        "source loaded",
        "batch started",
        "store closed",
        "models closed",
    ]
