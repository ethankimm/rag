"""Typed PostgreSQL storage and search contracts."""

from backend.storage.pg_search_store import (
    CollectionManifest,
    PgSearchStore,
    SearchDocument,
    SearchHit,
)

__all__ = [
    "CollectionManifest",
    "PgSearchStore",
    "SearchDocument",
    "SearchHit",
]
