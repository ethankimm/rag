"""Markdown loading, chunking, embedding, and ingestion orchestration tests."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from langchain_core.documents import Document

from backend.ingestion import __main__ as ingestion
from backend.ingestion.__main__ import (
    IngestionSummary,
    assign_chunk_ids,
    build_search_documents,
    embed_chunks_via_http,
    find_markdown_files,
    index_chunks,
    ingest_documents,
    load_documents,
    parse_frontmatter,
    prepare_chunks,
    split_documents,
)
from backend.retrieval.model_client import ModelClient
from backend.storage.pg_search_store import CollectionManifest, PgSearchStore


class FakeModelClient:
    def __init__(self, dimension: int = 3) -> None:
        self.config = SimpleNamespace(
            embedding_model="test-embedding",
            embedding_dimension=dimension,
        )
        self.calls: list[tuple[list[str], int]] = []

    def embed_documents(
        self,
        documents: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        self.calls.append((documents, batch_size))
        return [
            [float(index + 1)] * self.config.embedding_dimension
            for index in range(len(documents))
        ]


class FakeStore:
    collection_name = "test_markdown_docs"

    def __init__(self) -> None:
        self.replacements: list[tuple[list[Any], CollectionManifest]] = []

    def replace_documents(
        self,
        documents: list[Any],
        manifest: CollectionManifest,
    ) -> None:
        self.replacements.append((documents, manifest))


def write_document(path: Path, content: str = "# Lesson\n\nUseful content.") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_find_markdown_files_is_recursive_sorted_and_markdown_only(
    tmp_path: Path,
) -> None:
    write_document(tmp_path / "b/2.md")
    write_document(tmp_path / "a/1.md")
    write_document(tmp_path / "ignored.mdx")

    discovered = [
        path.relative_to(tmp_path).as_posix() for path in find_markdown_files(tmp_path)
    ]
    assert discovered == [
        "a/1.md",
        "b/2.md",
    ]


def test_parse_frontmatter_reads_scalars_and_preserves_body() -> None:
    metadata, body = parse_frontmatter(
        '---\ntitle: "Lesson"\nchapter: 2\npublished: true\n---\n\n# Body\n'
    )

    assert metadata == {"title": "Lesson", "chapter": 2, "published": True}
    assert body == "# Body"
    assert parse_frontmatter("# No frontmatter") == ({}, "# No frontmatter")


def test_load_documents_preserves_course_metadata_and_source_path(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path / "chapter1/1.md",
        '---\ntitle: "Intro"\nsource_revision: "abc"\n---\n# Intro',
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].page_content == "# Intro"
    assert documents[0].metadata["title"] == "Intro"
    assert documents[0].metadata["source_revision"] == "abc"
    assert documents[0].metadata["source_file"] == "chapter1/1.md"
    assert documents[0].metadata["file_type"] == "md"


def test_load_documents_detects_legacy_text_encoding(tmp_path: Path) -> None:
    path = tmp_path / "legacy.md"
    path.write_bytes("# Café\n\nRésumé".encode("windows-1252"))

    documents = load_documents(tmp_path)

    assert documents[0].page_content == "# Café\n\nRésumé"


def test_split_and_assign_chunk_ids_are_stable_per_source() -> None:
    document = Document(
        page_content=("first paragraph\n\n" * 100),
        metadata={"source_file": "chapter1/1.md"},
    )

    first = assign_chunk_ids(split_documents([document]))
    second = assign_chunk_ids(split_documents([document]))

    assert len(first) > 1
    assert [chunk.metadata["chunk_id"] for chunk in first] == [
        chunk.metadata["chunk_id"] for chunk in second
    ]
    assert [chunk.metadata["chunk_index"] for chunk in first] == list(range(len(first)))


def test_prepare_chunks_rejects_missing_and_empty_corpora(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        prepare_chunks(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="contains no .md"):
        prepare_chunks(empty)


def test_embed_chunks_delegates_one_bounded_batch() -> None:
    chunks = [Document(page_content="one"), Document(page_content="two")]
    client = FakeModelClient()

    vectors = embed_chunks_via_http(chunks, cast(ModelClient, client))

    assert vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    assert client.calls == [(["one", "two"], 16)]


def test_build_search_documents_pairs_vectors_and_rejects_mismatch() -> None:
    chunk = Document(
        page_content="content",
        metadata={"chunk_id": "chunk_1", "source_file": "lesson.md"},
    )

    documents = build_search_documents([chunk], [[1.0, 2.0, 3.0]])

    assert documents[0].chunk_id == "chunk_1"
    assert documents[0].content == "content"
    assert isinstance(documents[0].embedding, np.ndarray)
    assert documents[0].embedding.dtype == np.float32
    with pytest.raises(RuntimeError, match="Embedding count"):
        build_search_documents([chunk], [])


def test_index_chunks_atomically_replaces_store_and_returns_summary() -> None:
    chunk = Document(page_content="content", metadata={"chunk_id": "chunk_1"})
    client = FakeModelClient()
    store = FakeStore()

    summary = index_chunks(
        1,
        [chunk],
        cast(ModelClient, client),
        cast(PgSearchStore, store),
    )

    assert summary == IngestionSummary(
        source_document_count=1,
        chunk_count=1,
        collection_name="test_markdown_docs",
        embedding_model="test-embedding",
        embedding_dimension=3,
    )
    documents, manifest = store.replacements[0]
    assert len(documents) == 1
    assert manifest.document_count == 1
    assert manifest.embedding_dimension == 3


def test_ingest_documents_runs_complete_pipeline(tmp_path: Path) -> None:
    write_document(tmp_path / "chapter1/1.md")
    client = FakeModelClient()
    store = FakeStore()

    summary = ingest_documents(
        tmp_path,
        cast(ModelClient, client),
        cast(PgSearchStore, store),
    )

    assert summary.source_document_count == 1
    assert summary.chunk_count >= 1
    assert len(store.replacements) == 1


@pytest.mark.parametrize("failure", [None, RuntimeError("index failed")])
def test_main_uses_validated_runtime_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
    failure: RuntimeError | None,
) -> None:
    lifecycle: list[str] = []
    chunks = [Document(page_content="content", metadata={"chunk_id": "chunk_1"})]
    model_client = SimpleNamespace(
        config=SimpleNamespace(
            embedding_model="runtime-embedding",
            embedding_dimension=384,
        )
    )
    store = SimpleNamespace(collection_name="runtime_collection")

    @contextmanager
    def fake_local_model_runtime():
        lifecycle.append("models started")
        try:
            yield model_client
        finally:
            lifecycle.append("models closed")

    @contextmanager
    def fake_search_store(*, embedding_dimension: int):
        assert embedding_dimension == 384
        lifecycle.append("store started")
        try:
            yield store
        finally:
            lifecycle.append("store closed")

    def fake_index_chunks(
        source_document_count: int,
        received_chunks: list[Document],
        received_client: object,
        received_store: object,
    ) -> IngestionSummary:
        assert source_document_count == 1
        assert received_chunks is chunks
        assert received_client is model_client
        assert received_store is store
        lifecycle.append("batch started")
        if failure is not None:
            raise failure
        return IngestionSummary(
            source_document_count=1,
            chunk_count=1,
            collection_name="runtime_collection",
            embedding_model="runtime-embedding",
            embedding_dimension=384,
        )

    monkeypatch.setattr(ingestion, "prepare_chunks", lambda _path: (1, chunks))
    monkeypatch.setattr(ingestion, "local_model_runtime", fake_local_model_runtime)
    monkeypatch.setattr(ingestion, "PgSearchStore", fake_search_store)
    monkeypatch.setattr(ingestion, "index_chunks", fake_index_chunks)

    if failure is None:
        ingestion.main()
    else:
        with pytest.raises(RuntimeError, match="index failed"):
            ingestion.main()

    assert lifecycle == [
        "models started",
        "store started",
        "batch started",
        "store closed",
        "models closed",
    ]
