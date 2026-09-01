"""PostgreSQL integration coverage for the vendored course corpus."""

from __future__ import annotations

import os

import pytest
from psycopg import sql

from backend.ingestion.__main__ import DOCS_DIR, build_search_documents, prepare_chunks
from backend.storage.pg_search_store import (
    MANIFEST_TABLE,
    CollectionManifest,
    PgSearchStore,
    resolve_dsn,
)

COLLECTION = "test_hf_course_ingestion"


@pytest.mark.integration
def test_course_snapshot_replaces_isolated_collection_and_is_searchable() -> None:
    try:
        store = PgSearchStore(
            collection_name=COLLECTION,
            embedding_dimension=3,
            dsn=resolve_dsn(),
            verbose=False,
        )
    except Exception as error:
        if os.environ.get("REQUIRE_POSTGRES_TESTS") == "1":
            pytest.fail(f"Required PostgreSQL search service is unavailable: {error}")
        pytest.skip(f"PostgreSQL search service is unavailable: {error}")

    try:
        source_count, chunks = prepare_chunks(DOCS_DIR)
        vectors = [[1.0, 0.0, 0.0] for _ in chunks]
        documents = build_search_documents(chunks, vectors)
        manifest = CollectionManifest(len(documents), "integration-fixture", 3)

        store.replace_documents(documents, manifest)

        assert source_count == 104
        assert store.count() == len(chunks) > source_count
        assert store.manifest() == manifest
        with store.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "SELECT COUNT(DISTINCT metadata ->> 'source_file') FROM {}"
                ).format(sql.Identifier(COLLECTION))
            )
            assert cursor.fetchone()[0] == 104
        results = store.bm25_search(
            "natural language processing large language models",
            limit=10,
        )
        assert results
        assert any(
            result.metadata.get("source_file") == "chapter1/1.md" for result in results
        )
    finally:
        with store.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(COLLECTION)
                )
            )
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE collection_name = %s").format(
                    sql.Identifier(MANIFEST_TABLE)
                ),
                (COLLECTION,),
            )
        store.close()
