"""Validated sessions with an external, independently owned Model Server."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import cache
from time import monotonic, sleep

import httpx

from backend.retrieval.model_client import (
    DEFAULT_MODEL_CONFIG,
    ModelClient,
    ModelConfig,
)


class LocalModelStartupError(RuntimeError):
    """The external server or configured model contract could not be validated."""

    def __init__(
        self, message: str, *, missing_model_ids: frozenset[str] = frozenset()
    ) -> None:
        super().__init__(message)
        self.missing_model_ids = missing_model_ids


@cache
def _load_config() -> ModelConfig:
    """Freeze environment settings at first use until the process restarts."""
    raw_dimension = os.environ.get(
        "LOCAL_EMBEDDING_DIMENSION", str(DEFAULT_MODEL_CONFIG.embedding_dimension)
    )
    try:
        dimension = int(raw_dimension)
        if dimension <= 0:
            raise ValueError("dimension is not positive")
    except ValueError as error:
        raise ValueError(
            "LOCAL_EMBEDDING_DIMENSION must be a positive integer; "
            f"received {raw_dimension!r}"
        ) from error
    config = ModelConfig(
        base_url=os.environ.get("LOCAL_MODEL_BASE_URL", "http://127.0.0.1:8080"),
        embedding_model=os.environ.get(
            "LOCAL_EMBEDDING_MODEL", DEFAULT_MODEL_CONFIG.embedding_model
        ),
        reranking_model=os.environ.get(
            "LOCAL_RERANKING_MODEL", DEFAULT_MODEL_CONFIG.reranking_model
        ),
        embedding_dimension=dimension,
    )
    url = httpx.URL(config.base_url)
    if url.scheme not in {"http", "https"} or not url.host:
        raise ValueError("LOCAL_MODEL_BASE_URL must be an absolute HTTP(S) URL")
    for name, value in (
        ("LOCAL_EMBEDDING_MODEL", config.embedding_model),
        ("LOCAL_RERANKING_MODEL", config.reranking_model),
    ):
        if not value.strip():
            raise ValueError(f"{name} must be a nonempty model ID")
    return config


def _create_client(config: ModelConfig) -> ModelClient:
    return ModelClient(config)


def _wait_for_server(client: ModelClient) -> None:
    deadline = monotonic() + 5.0
    failure: Exception = TimeoutError("Model Server validation deadline expired")
    missing: frozenset[str] = frozenset()
    while (remaining := deadline - monotonic()) > 0:
        missing = frozenset()
        try:
            client.check_health(timeout=remaining)
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise httpx.TimeoutException("Model Server health deadline expired")
            models = client.list_models(timeout=remaining)
            missing = frozenset(
                {client.config.embedding_model, client.config.reranking_model} - models
            )
            if not missing and monotonic() < deadline:
                return
            failure = (
                RuntimeError(f"Missing required models: {sorted(missing)}")
                if missing
                else TimeoutError("Model catalog validation deadline expired")
            )
        except httpx.HTTPError as error:
            failure = error
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        sleep(min(0.5, remaining))
    raise LocalModelStartupError(
        f"Model Server at {client.config.base_url} was not ready within 5 seconds: "
        f"{failure}",
        missing_model_ids=missing,
    ) from failure


@contextmanager
def local_model_runtime() -> Iterator[ModelClient]:
    """Yield one ready client, closing only its connection when the session ends."""
    with ExitStack() as stack:
        try:
            config = _load_config()
            client = stack.enter_context(_create_client(config))
            print(
                "Local Model Runtime: waiting for Model Server health at "
                f"{config.base_url}...",
                file=sys.stderr,
                flush=True,
            )
            _wait_for_server(client)
            print(
                "Local Model Runtime: loading and validating embedding model "
                f"{config.embedding_model} (up to 120 seconds)...",
                file=sys.stderr,
                flush=True,
            )
            client.embed_query("Local Model Runtime startup validation")
        except LocalModelStartupError:
            raise
        except Exception as error:
            raise LocalModelStartupError(
                f"Local Model Runtime startup failed: {error}"
            ) from error
        print("Local Model Runtime: ready", file=sys.stderr, flush=True)
        yield client
