"""Canonical PostgreSQL storage and search boundary for the RAG system."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb

DEFAULT_DSN = "postgresql://rag:rag@localhost:5432/rag"
DEFAULT_EMBEDDING_DIM = 1024
FTS_LANGUAGE = "english"
BM25_K1 = 1.2
BM25_B = 0.75
MANIFEST_TABLE = "rag_collection_manifests"
_POSTGRES_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PgSearchStoreError(RuntimeError):
    """Base error for an unusable PostgreSQL search store."""


class SchemaContractError(PgSearchStoreError):
    """Raised when an existing collection has an incompatible table shape."""

    def __init__(
        self,
        collection_name: str,
        *,
        expected: object,
        observed: object,
    ) -> None:
        self.collection_name = collection_name
        self.expected = expected
        self.observed = observed
        super().__init__(
            f"Collection {collection_name!r} does not match its physical schema "
            f"contract; expected {expected!r}, observed {observed!r}"
        )


class Bm25ConfigurationError(PgSearchStoreError):
    """Raised when the required BM25 index configuration cannot be enforced."""

    def __init__(
        self,
        collection_name: str,
        *,
        expected: object,
        observed: object,
    ) -> None:
        self.collection_name = collection_name
        self.expected = expected
        self.observed = observed
        super().__init__(
            f"Collection {collection_name!r} does not have the required BM25 "
            f"index; expected {expected!r}, observed {observed!r}"
        )


@dataclass(frozen=True)
class _ColumnContract:
    name: str
    data_type: str
    not_null: bool


@dataclass(frozen=True)
class _TableContract:
    columns: tuple[_ColumnContract, ...]
    primary_key: tuple[str, ...]


@dataclass(frozen=True)
class _Bm25IndexContract:
    relation_kind: str
    table_name: str | None
    access_method: str | None
    columns: tuple[str, ...]
    text_config: str | None
    k1: float | None
    b: float | None


@dataclass(frozen=True)
class SearchDocument:
    """One document ready to be stored in a search collection."""

    chunk_id: str
    content: str
    metadata: Mapping[str, Any]
    embedding: Sequence[float] | np.ndarray[Any, Any] | None = None


@dataclass(frozen=True)
class SearchHit:
    """One typed result returned by dense or lexical search."""

    chunk_id: str
    content: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class CollectionManifest:
    """Configuration associated with the currently stored corpus."""

    document_count: int
    embedding_model: str | None
    embedding_dimension: int | None
    bm25_k1: float = BM25_K1
    bm25_b: float = BM25_B


def resolve_dsn(dsn: str | None = None) -> str:
    """Resolve an explicit DSN before falling back to environment and local defaults."""
    return dsn or os.environ.get("PG_DSN", DEFAULT_DSN)


def validate_identifier(name: str) -> str:
    """Validate an unquoted PostgreSQL identifier used for a collection."""
    if not _POSTGRES_IDENTIFIER.fullmatch(name):
        raise ValueError(f"Invalid collection name: {name!r}")
    if len(name.encode("utf-8")) > 48:
        raise ValueError("Collection names must be at most 48 bytes")
    return name


def bm25_index_name(collection_name: str) -> str:
    return f"{validate_identifier(collection_name)}_bm25_idx"


class PgSearchStore:
    """Own PostgreSQL schema, corpus replacement, manifests, and search queries."""

    def __init__(
        self,
        *,
        collection_name: str = "markdown_docs",
        embedding_dimension: int | None = DEFAULT_EMBEDDING_DIM,
        dsn: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.collection_name = validate_identifier(collection_name)
        if embedding_dimension is not None and (
            isinstance(embedding_dimension, bool) or embedding_dimension <= 0
        ):
            raise ValueError("Embedding dimension must be a positive integer or None")
        self.embedding_dimension = embedding_dimension
        self.verbose = verbose
        self.connection = psycopg.connect(resolve_dsn(dsn), autocommit=True)
        try:
            self._initialize_schema()
        except BaseException:
            with suppress(Exception):
                self.connection.close()
            raise

    def __enter__(self) -> PgSearchStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def has_embeddings(self) -> bool:
        return self.embedding_dimension is not None

    def _initialize_schema(self) -> None:
        table = sql.Identifier(self.collection_name)
        embedding_index = sql.Identifier(f"{self.collection_name}_embedding_idx")

        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")
            register_vector(self.connection)
            embedding_column = sql.SQL("")
            if self.has_embeddings:
                embedding_column = sql.SQL(", embedding vector({}) NOT NULL").format(
                    sql.Literal(self.embedding_dimension)
                )
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        chunk_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL{}
                    )
                    """
                ).format(table, embedding_column)
            )
            observed_schema = self._read_table_contract(cursor)
            expected_schema = self._expected_table_contract()
            if observed_schema != expected_schema:
                raise SchemaContractError(
                    self.collection_name,
                    expected=expected_schema,
                    observed=observed_schema,
                )
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        collection_name TEXT PRIMARY KEY,
                        document_count INTEGER NOT NULL,
                        embedding_model TEXT,
                        embedding_dimension INTEGER,
                        bm25_k1 DOUBLE PRECISION NOT NULL,
                        bm25_b DOUBLE PRECISION NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                ).format(sql.Identifier(MANIFEST_TABLE))
            )
            self._ensure_bm25_index(cursor)
            if self.has_embeddings:
                cursor.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {} "
                        "USING hnsw (embedding vector_cosine_ops)"
                    ).format(embedding_index, table)
                )

        if self.verbose:
            print(
                f"PostgreSQL search store ready: {self.collection_name} "
                f"({self.count()} documents)"
            )

    def _expected_table_contract(self) -> _TableContract:
        columns = [
            _ColumnContract("chunk_id", "text", True),
            _ColumnContract("content", "text", True),
            _ColumnContract("metadata", "jsonb", True),
        ]
        if self.embedding_dimension is not None:
            columns.append(
                _ColumnContract(
                    "embedding",
                    f"vector({self.embedding_dimension})",
                    True,
                )
            )
        return _TableContract(tuple(columns), ("chunk_id",))

    def _read_table_contract(self, cursor: psycopg.Cursor[Any]) -> _TableContract:
        cursor.execute(
            """
            SELECT attribute.attname,
                   pg_catalog.format_type(attribute.atttypid, attribute.atttypmod),
                   attribute.attnotnull
            FROM pg_catalog.pg_attribute AS attribute
            WHERE attribute.attrelid = pg_catalog.to_regclass(%s)
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
            ORDER BY attribute.attnum
            """,
            (self.collection_name,),
        )
        columns = tuple(_ColumnContract(*row) for row in cursor.fetchall())
        cursor.execute(
            """
            SELECT array_agg(attribute.attname ORDER BY key.position)
            FROM pg_catalog.pg_index AS index_metadata
            CROSS JOIN LATERAL unnest(index_metadata.indkey)
                WITH ORDINALITY AS key(attnum, position)
            JOIN pg_catalog.pg_attribute AS attribute
              ON attribute.attrelid = index_metadata.indrelid
             AND attribute.attnum = key.attnum
            WHERE index_metadata.indrelid = pg_catalog.to_regclass(%s)
              AND index_metadata.indisprimary
            GROUP BY index_metadata.indexrelid
            """,
            (self.collection_name,),
        )
        primary_key_row = cursor.fetchone()
        primary_key = tuple(primary_key_row[0]) if primary_key_row else ()
        return _TableContract(columns, primary_key)

    def _expected_bm25_index_contract(self) -> _Bm25IndexContract:
        return _Bm25IndexContract(
            relation_kind="i",
            table_name=self.collection_name,
            access_method="bm25",
            columns=("content",),
            text_config=FTS_LANGUAGE,
            k1=BM25_K1,
            b=BM25_B,
        )

    def _read_bm25_index_contract(
        self,
        cursor: psycopg.Cursor[Any],
    ) -> _Bm25IndexContract | None:
        cursor.execute(
            """
            SELECT index_relation.oid,
                   index_relation.relkind,
                   table_relation.relname,
                   access_method.amname,
                   index_relation.reloptions
            FROM pg_catalog.pg_class AS index_relation
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = index_relation.relnamespace
            LEFT JOIN pg_catalog.pg_index AS index_metadata
              ON index_metadata.indexrelid = index_relation.oid
            LEFT JOIN pg_catalog.pg_class AS table_relation
              ON table_relation.oid = index_metadata.indrelid
            LEFT JOIN pg_catalog.pg_am AS access_method
              ON access_method.oid = index_relation.relam
            WHERE namespace.nspname = current_schema()
              AND index_relation.relname = %s
            """,
            (bm25_index_name(self.collection_name),),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        index_oid, relation_kind, table_name, access_method, raw_options = row
        cursor.execute(
            """
            SELECT attribute.attname
            FROM pg_catalog.pg_index AS index_metadata
            CROSS JOIN LATERAL unnest(index_metadata.indkey)
                WITH ORDINALITY AS key(attnum, position)
            JOIN pg_catalog.pg_attribute AS attribute
              ON attribute.attrelid = index_metadata.indrelid
             AND attribute.attnum = key.attnum
            WHERE index_metadata.indexrelid = %s
            ORDER BY key.position
            """,
            (index_oid,),
        )
        columns = tuple(column_row[0] for column_row in cursor.fetchall())
        options = dict(
            option.split("=", 1) for option in (raw_options or ()) if "=" in option
        )
        return _Bm25IndexContract(
            relation_kind=relation_kind,
            table_name=table_name,
            access_method=access_method,
            columns=columns,
            text_config=options.get("text_config"),
            k1=self._parse_float_option(options.get("k1")),
            b=self._parse_float_option(options.get("b")),
        )

    @staticmethod
    def _parse_float_option(value: str | None) -> float | None:
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    def _create_bm25_index(self, cursor: psycopg.Cursor[Any]) -> None:
        cursor.execute(
            sql.SQL(
                "CREATE INDEX {} ON {} "
                "USING bm25 (content) WITH (text_config={}, k1={}, b={})"
            ).format(
                sql.Identifier(bm25_index_name(self.collection_name)),
                sql.Identifier(self.collection_name),
                sql.Literal(FTS_LANGUAGE),
                sql.Literal(BM25_K1),
                sql.Literal(BM25_B),
            )
        )

    def _ensure_bm25_index(self, cursor: psycopg.Cursor[Any]) -> None:
        expected = self._expected_bm25_index_contract()
        observed = self._read_bm25_index_contract(cursor)

        if observed is not None and observed.table_name != self.collection_name:
            raise Bm25ConfigurationError(
                self.collection_name,
                expected=expected,
                observed=observed,
            )

        try:
            if observed is None:
                self._create_bm25_index(cursor)
            elif observed != expected:
                cursor.execute(
                    sql.SQL("DROP INDEX {}").format(
                        sql.Identifier(bm25_index_name(self.collection_name))
                    )
                )
                self._create_bm25_index(cursor)
        except psycopg.Error as error:
            raise Bm25ConfigurationError(
                self.collection_name,
                expected=expected,
                observed=observed,
            ) from error

        verified = self._read_bm25_index_contract(cursor)
        if verified != expected:
            raise Bm25ConfigurationError(
                self.collection_name,
                expected=expected,
                observed=verified,
            )

    def count(self) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT COUNT(*) FROM {}").format(
                    sql.Identifier(self.collection_name)
                )
            )
            return int(cursor.fetchone()[0])

    def manifest(self) -> CollectionManifest | None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "SELECT document_count, embedding_model, embedding_dimension, "
                    "bm25_k1, bm25_b FROM {} WHERE collection_name = %s"
                ).format(sql.Identifier(MANIFEST_TABLE)),
                (self.collection_name,),
            )
            row = cursor.fetchone()
        return CollectionManifest(*row) if row else None

    def can_reuse(self, expected: CollectionManifest) -> bool:
        """Check configuration; content freshness is intentionally deferred."""
        return self.count() == expected.document_count and self.manifest() == expected

    def replace_documents(
        self,
        documents: Sequence[SearchDocument],
        manifest: CollectionManifest,
    ) -> None:
        """Atomically replace a corpus and its manifest."""
        if len(documents) != manifest.document_count:
            raise ValueError(
                "Manifest document count does not match supplied documents"
            )
        if manifest.embedding_dimension != self.embedding_dimension:
            raise ValueError(
                "Manifest embedding dimension does not match the collection contract"
            )
        if manifest.bm25_k1 != BM25_K1 or manifest.bm25_b != BM25_B:
            raise ValueError("Manifest BM25 settings do not match the verified index")

        rows = []
        for document in documents:
            if self.has_embeddings != (document.embedding is not None):
                raise ValueError(
                    "Every document must match the collection's embedding configuration"
                )
            base = (document.chunk_id, document.content, Jsonb(dict(document.metadata)))
            rows.append((*base, document.embedding) if self.has_embeddings else base)

        table = sql.Identifier(self.collection_name)
        columns = sql.SQL("chunk_id, content, metadata, embedding")
        placeholders = sql.SQL("%s, %s, %s, %s")
        if not self.has_embeddings:
            columns = sql.SQL("chunk_id, content, metadata")
            placeholders = sql.SQL("%s, %s, %s")

        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.execute(sql.SQL("TRUNCATE TABLE {}").format(table))
            if rows:
                cursor.executemany(
                    sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                        table, columns, placeholders
                    ),
                    rows,
                )
            cursor.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (
                        collection_name, document_count, embedding_model,
                        embedding_dimension, bm25_k1, bm25_b
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (collection_name) DO UPDATE SET
                        document_count = EXCLUDED.document_count,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dimension = EXCLUDED.embedding_dimension,
                        bm25_k1 = EXCLUDED.bm25_k1,
                        bm25_b = EXCLUDED.bm25_b,
                        updated_at = NOW()
                    """
                ).format(sql.Identifier(MANIFEST_TABLE)),
                (
                    self.collection_name,
                    manifest.document_count,
                    manifest.embedding_model,
                    manifest.embedding_dimension,
                    manifest.bm25_k1,
                    manifest.bm25_b,
                ),
            )

    def dense_search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[SearchHit]:
        if not self.has_embeddings:
            raise RuntimeError("Dense search requires an embedding-enabled collection")
        vector = np.asarray(query_embedding, dtype=np.float32)
        with self.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "SELECT chunk_id, content, metadata, "
                    "1 - (embedding <=> %s) AS score "
                    "FROM {} ORDER BY embedding <=> %s LIMIT %s"
                ).format(sql.Identifier(self.collection_name)),
                (vector, vector, limit),
            )
            return [SearchHit(*row) for row in cursor.fetchall()]

    def bm25_search(self, query: str, *, limit: int) -> list[SearchHit]:
        if not query.strip():
            return []
        with self.connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    SELECT chunk_id, content, metadata, -bm25_score AS score
                    FROM (
                        SELECT chunk_id, content, metadata,
                               content <@> to_bm25query(%s, %s) AS bm25_score
                        FROM {}
                        ORDER BY bm25_score
                        LIMIT %s
                    ) AS ranked
                    WHERE bm25_score < 0
                    ORDER BY bm25_score
                    """
                ).format(sql.Identifier(self.collection_name)),
                (query, bm25_index_name(self.collection_name), limit),
            )
            return [SearchHit(*row) for row in cursor.fetchall()]

    def close(self) -> None:
        if not self.connection.closed:
            self.connection.close()
