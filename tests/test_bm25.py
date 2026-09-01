"""PostgreSQL pg_textsearch BM25 integration suite."""

from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql

from backend.storage.pg_search_store import (
    BM25_B,
    BM25_K1,
    MANIFEST_TABLE,
    Bm25ConfigurationError,
    CollectionManifest,
    PgSearchStore,
    SearchDocument,
    bm25_index_name,
    resolve_dsn,
)

COLLECTION = "test_bm25_retrieval"


@pytest.fixture(scope="module")
def bm25_store() -> Iterator[PgSearchStore]:
    try:
        store = PgSearchStore(
            collection_name=COLLECTION,
            embedding_dimension=None,
            dsn=resolve_dsn(),
            verbose=False,
        )
    except Exception as error:
        if os.environ.get("REQUIRE_POSTGRES_TESTS") == "1":
            pytest.fail(f"Required PostgreSQL BM25 service is unavailable: {error}")
        pytest.skip(f"PostgreSQL BM25 service is unavailable: {error}")

    documents = [
        SearchDocument(
            "doc_cancer_full",
            "Cancer survivors benefit from nutrition and physical exercise guidelines.",
            {"source": "doc1"},
        ),
        SearchDocument(
            "doc_cancer_partial",
            "General physical activity is recommended for health maintenance.",
            {"source": "doc2"},
        ),
        SearchDocument(
            "doc_materials",
            "0-dimensional biomaterials show inductive properties on mammalian cells.",
            {"source": "doc3"},
        ),
        SearchDocument(
            "doc_irrelevant",
            "Planetary orbits follow elliptical paths around the sun.",
            {"source": "doc4"},
        ),
    ]
    store.replace_documents(documents, CollectionManifest(len(documents), None, None))
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
def test_bm25_ranks_multi_term_match_first(bm25_store: PgSearchStore) -> None:
    results = bm25_store.bm25_search(
        "What nutrition guidelines help cancer patients during recovery?",
        limit=5,
    )
    assert results[0].chunk_id == "doc_cancer_full"


@pytest.mark.integration
def test_bm25_uses_english_stemming(bm25_store: PgSearchStore) -> None:
    results = bm25_store.bm25_search("induction in mammalian cell cultures", limit=5)
    assert "doc_materials" in {result.chunk_id for result in results}


@pytest.mark.integration
def test_bm25_returns_no_stopword_only_matches(bm25_store: PgSearchStore) -> None:
    assert bm25_store.bm25_search("the and or of in", limit=5) == []


@pytest.mark.integration
def test_bm25_handles_punctuation_safely(bm25_store: PgSearchStore) -> None:
    results = bm25_store.bm25_search(
        '0-dimensional "biomaterials" (mammalian & cells) [test]!',
        limit=5,
    )
    assert results[0].chunk_id == "doc_materials"


@pytest.mark.integration
def test_bm25_query_uses_native_index(bm25_store: PgSearchStore) -> None:
    index_name = bm25_index_name(COLLECTION)
    with bm25_store.connection.cursor() as cursor:
        cursor.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = %s",
            (index_name,),
        )
        index_definition = cursor.fetchone()[0]
        cursor.execute(
            sql.SQL(
                "EXPLAIN SELECT chunk_id FROM {} "
                "ORDER BY content <@> to_bm25query(%s, %s) LIMIT 5"
            ).format(sql.Identifier(COLLECTION)),
            ("nutrition cancer recovery", index_name),
        )
        query_plan = "\n".join(row[0] for row in cursor.fetchall())

    assert "USING bm25" in index_definition
    assert f"Index Scan using {index_name}" in query_plan


@pytest.mark.integration
def test_corpus_replacement_is_atomic(bm25_store: PgSearchStore) -> None:
    original_count = bm25_store.count()
    duplicate_documents = [
        SearchDocument("duplicate", "first", {}),
        SearchDocument("duplicate", "second", {}),
    ]

    with pytest.raises(psycopg.errors.UniqueViolation):
        bm25_store.replace_documents(
            duplicate_documents,
            CollectionManifest(len(duplicate_documents), None, None),
        )

    assert bm25_store.count() == original_count
    assert bm25_store.bm25_search("nutrition cancer", limit=1)[0].chunk_id == (
        "doc_cancer_full"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("collection", "wrong_index_definition"),
    [
        (
            "test_bm25_wrong_k1",
            "CREATE INDEX {} ON {} USING bm25 (content) "
            "WITH (text_config='english', k1='2.0', b='0.75')",
        ),
        (
            "test_bm25_wrong_b",
            "CREATE INDEX {} ON {} USING bm25 (content) "
            "WITH (text_config='english', k1='1.2', b='0.25')",
        ),
        (
            "test_bm25_wrong_text_config",
            "CREATE INDEX {} ON {} USING bm25 (content) "
            "WITH (text_config='simple', k1='1.2', b='0.75')",
        ),
        (
            "test_bm25_wrong_method",
            "CREATE INDEX {} ON {} USING btree (content)",
        ),
        (
            "test_bm25_wrong_column",
            "CREATE INDEX {} ON {} USING bm25 (chunk_id) "
            "WITH (text_config='english', k1='1.2', b='0.75')",
        ),
    ],
)
def test_incompatible_bm25_index_is_rebuilt_and_verified(
    bm25_store: PgSearchStore,
    collection: str,
    wrong_index_definition: str,
) -> None:
    index_name = bm25_index_name(collection)
    with bm25_store.connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                sql.Identifier(collection)
            )
        )

    try:
        with PgSearchStore(
            collection_name=collection,
            embedding_dimension=None,
            dsn=resolve_dsn(),
            verbose=False,
        ):
            pass

        with bm25_store.connection.cursor() as cursor:
            cursor.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(index_name)))
            cursor.execute(
                sql.SQL(wrong_index_definition).format(
                    sql.Identifier(index_name),
                    sql.Identifier(collection),
                )
            )

        with PgSearchStore(
            collection_name=collection,
            embedding_dimension=None,
            dsn=resolve_dsn(),
            verbose=False,
        ):
            pass

        with bm25_store.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT access_method.amname,
                       index_relation.reloptions,
                       pg_get_indexdef(index_relation.oid)
                FROM pg_catalog.pg_class AS index_relation
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = index_relation.relnamespace
                JOIN pg_catalog.pg_am AS access_method
                  ON access_method.oid = index_relation.relam
                WHERE namespace.nspname = current_schema()
                  AND index_relation.relname = %s
                """,
                (index_name,),
            )
            method, raw_options, definition = cursor.fetchone()

        assert method == "bm25"
        assert set(raw_options) == {
            "text_config=english",
            f"k1={BM25_K1}",
            f"b={BM25_B}",
        }
        assert "(content)" in definition
    finally:
        with bm25_store.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(collection)
                )
            )


@pytest.mark.integration
def test_failed_bm25_rebuild_rolls_back_existing_index(
    bm25_store: PgSearchStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = "test_bm25_rebuild_rollback"
    index_name = bm25_index_name(collection)
    with bm25_store.connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                sql.Identifier(collection)
            )
        )

    try:
        with PgSearchStore(
            collection_name=collection,
            embedding_dimension=None,
            dsn=resolve_dsn(),
            verbose=False,
        ):
            pass
        with bm25_store.connection.cursor() as cursor:
            cursor.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(index_name)))
            cursor.execute(
                sql.SQL("CREATE INDEX {} ON {} USING btree (content)").format(
                    sql.Identifier(index_name),
                    sql.Identifier(collection),
                )
            )

        def fail_creation(
            _store: PgSearchStore,
            _cursor: psycopg.Cursor[object],
        ) -> None:
            raise psycopg.OperationalError("forced rebuild failure")

        monkeypatch.setattr(PgSearchStore, "_create_bm25_index", fail_creation)

        with pytest.raises(Bm25ConfigurationError) as captured:
            PgSearchStore(
                collection_name=collection,
                embedding_dimension=None,
                dsn=resolve_dsn(),
                verbose=False,
            )

        assert isinstance(captured.value.__cause__, psycopg.OperationalError)
        with bm25_store.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT access_method.amname
                FROM pg_catalog.pg_class AS index_relation
                JOIN pg_catalog.pg_am AS access_method
                  ON access_method.oid = index_relation.relam
                WHERE index_relation.relname = %s
                """,
                (index_name,),
            )
            assert cursor.fetchone()[0] == "btree"
    finally:
        with bm25_store.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(collection)
                )
            )


@pytest.mark.integration
def test_bm25_index_name_collision_is_not_dropped(
    bm25_store: PgSearchStore,
) -> None:
    collection = "test_bm25_name_collision"
    owner_table = "test_bm25_collision_owner"
    index_name = bm25_index_name(collection)
    with bm25_store.connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                sql.Identifier(collection)
            )
        )
        cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                sql.Identifier(owner_table)
            )
        )
        cursor.execute(
            sql.SQL("CREATE TABLE {} (content TEXT NOT NULL)").format(
                sql.Identifier(owner_table)
            )
        )
        cursor.execute(
            sql.SQL("CREATE INDEX {} ON {} USING btree (content)").format(
                sql.Identifier(index_name),
                sql.Identifier(owner_table),
            )
        )

    try:
        with pytest.raises(Bm25ConfigurationError):
            PgSearchStore(
                collection_name=collection,
                embedding_dimension=None,
                dsn=resolve_dsn(),
                verbose=False,
            )

        with bm25_store.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_relation.relname
                FROM pg_catalog.pg_class AS index_relation
                JOIN pg_catalog.pg_index AS index_metadata
                  ON index_metadata.indexrelid = index_relation.oid
                JOIN pg_catalog.pg_class AS table_relation
                  ON table_relation.oid = index_metadata.indrelid
                WHERE index_relation.relname = %s
                """,
                (index_name,),
            )
            assert cursor.fetchone()[0] == owner_table
            cursor.execute("SELECT pg_catalog.to_regclass(%s)", (collection,))
            assert cursor.fetchone()[0] is None
    finally:
        with bm25_store.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(collection)
                )
            )
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(owner_table)
                )
            )
