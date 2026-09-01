"""Integration tests for the combined dense and lexical search store."""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pytest
from psycopg import sql

from backend.storage.pg_search_store import (
    MANIFEST_TABLE,
    CollectionManifest,
    PgSearchStore,
    SchemaContractError,
    SearchDocument,
    resolve_dsn,
)

COLLECTION = "test_combined_search_store"


@pytest.fixture(scope="module")
def search_store() -> Iterator[PgSearchStore]:
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

    documents = [
        SearchDocument(
            "dense_match",
            "A semantic document about nutrition.",
            {},
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        ),
        SearchDocument(
            "dense_other",
            "A document about planetary motion.",
            {},
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        ),
    ]
    manifest = CollectionManifest(len(documents), "test-model", 3)
    store.replace_documents(documents, manifest)
    yield store

    with store.connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(COLLECTION))
        )
        cursor.execute(
            sql.SQL("DELETE FROM {} WHERE collection_name = %s").format(
                sql.Identifier(MANIFEST_TABLE)
            ),
            (COLLECTION,),
        )
    store.close()


@pytest.mark.integration
def test_combined_store_owns_dense_bm25_and_manifest(
    search_store: PgSearchStore,
) -> None:
    dense_results = search_store.dense_search([1.0, 0.0, 0.0], limit=2)
    lexical_results = search_store.bm25_search("planetary motion", limit=2)

    assert dense_results[0].chunk_id == "dense_match"
    assert lexical_results[0].chunk_id == "dense_other"
    assert search_store.can_reuse(CollectionManifest(2, "test-model", 3))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("collection", "initial_dimension", "requested_dimension"),
    [
        ("test_contract_lexical_hybrid", None, 3),
        ("test_contract_hybrid_lexical", 3, None),
        ("test_contract_vector_dimension", 3, 4),
    ],
)
def test_existing_collection_schema_mismatch_is_rejected_without_data_loss(
    search_store: PgSearchStore,
    collection: str,
    initial_dimension: int | None,
    requested_dimension: int | None,
) -> None:
    with search_store.connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                sql.Identifier(collection)
            )
        )
        cursor.execute(
            sql.SQL("DELETE FROM {} WHERE collection_name = %s").format(
                sql.Identifier(MANIFEST_TABLE)
            ),
            (collection,),
        )

    try:
        with PgSearchStore(
            collection_name=collection,
            embedding_dimension=initial_dimension,
            dsn=resolve_dsn(),
            verbose=False,
        ) as initial_store:
            embedding = (
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                if initial_dimension is not None
                else None
            )
            initial_store.replace_documents(
                [SearchDocument("preserved", "preserved content", {}, embedding)],
                CollectionManifest(
                    1,
                    "test-model" if initial_dimension is not None else None,
                    initial_dimension,
                ),
            )

        with pytest.raises(SchemaContractError) as captured:
            PgSearchStore(
                collection_name=collection,
                embedding_dimension=requested_dimension,
                dsn=resolve_dsn(),
                verbose=False,
            )

        assert captured.value.collection_name == collection
        with search_store.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT chunk_id, content FROM {}").format(
                    sql.Identifier(collection)
                )
            )
            assert cursor.fetchall() == [("preserved", "preserved content")]
    finally:
        with search_store.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(collection)
                )
            )
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE collection_name = %s").format(
                    sql.Identifier(MANIFEST_TABLE)
                ),
                (collection,),
            )
