"""Failure-path tests for PostgreSQL search-store ownership and contracts."""

from __future__ import annotations

from typing import Any, cast

import psycopg
import pytest

import backend.storage.pg_search_store as store_module
from backend.storage.pg_search_store import (
    Bm25ConfigurationError,
    PgSearchStore,
)


class _FakeConnection:
    def __init__(self) -> None:
        self.close_calls = 0
        self.closed = False

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


def test_constructor_closes_connection_when_schema_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _FakeConnection()
    failure = RuntimeError("schema initialization failed")

    monkeypatch.setattr(
        store_module.psycopg,
        "connect",
        lambda *_args, **_kwargs: connection,
    )

    def fail_initialization(_store: PgSearchStore) -> None:
        raise failure

    monkeypatch.setattr(PgSearchStore, "_initialize_schema", fail_initialization)

    with pytest.raises(RuntimeError) as captured:
        PgSearchStore(verbose=False)

    assert captured.value is failure
    assert connection.closed
    assert connection.close_calls == 1


def test_bm25_verification_failure_returns_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = cast(PgSearchStore, object.__new__(PgSearchStore))
    store.collection_name = "verification_failure"
    monkeypatch.setattr(store, "_read_bm25_index_contract", lambda _cursor: None)
    monkeypatch.setattr(store, "_create_bm25_index", lambda _cursor: None)

    with pytest.raises(Bm25ConfigurationError) as captured:
        store._ensure_bm25_index(cast(Any, object()))

    assert captured.value.observed is None


def test_bm25_database_failure_is_preserved_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = cast(PgSearchStore, object.__new__(PgSearchStore))
    store.collection_name = "creation_failure"
    failure = psycopg.OperationalError("forced creation failure")
    monkeypatch.setattr(store, "_read_bm25_index_contract", lambda _cursor: None)

    def fail_creation(_cursor: object) -> None:
        raise failure

    monkeypatch.setattr(store, "_create_bm25_index", fail_creation)

    with pytest.raises(Bm25ConfigurationError) as captured:
        store._ensure_bm25_index(cast(Any, object()))

    assert captured.value.__cause__ is failure
