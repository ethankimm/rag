"""Lazy ingestion exports that keep ``python -m backend.ingestion`` predictable."""

__all__ = [
    "load_documents",
    "split_documents",
    "assign_chunk_ids",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from backend.ingestion import __main__ as pipeline

    return getattr(pipeline, name)
