"""Local Model Runtime contracts observed through its public context manager."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import httpx
import pytest

from backend import local_model_runtime as runtime
from backend.retrieval.model_client import ModelClient, ModelConfig


class ModelServer(httpx.MockTransport):
    """Deterministic external server with observable HTTP requests and cleanup."""

    def __init__(self) -> None:
        super().__init__(self.respond)
        self.requests: list[httpx.Request] = []
        self.closed = False
        self.routes: dict[str, Callable[[httpx.Request], httpx.Response]] = {}

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path in self.routes:
            return self.routes[request.url.path](request)
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": name}
                        for name in (
                            "bge-m3-q8_0",
                            "bge-reranker-v2-m3-Q8_0",
                            "extra-model",
                        )
                    ]
                },
            )
        if request.url.path == "/v1/embeddings":
            return httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.5]}]}
            )
        raise AssertionError(f"Unexpected endpoint: {request.url.path}")

    def close(self) -> None:
        self.closed = True
        super().close()


@pytest.fixture
def routes():
    return {}


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, routes):
    for name in (
        "LOCAL_MODEL_BASE_URL",
        "LOCAL_EMBEDDING_MODEL",
        "LOCAL_RERANKING_MODEL",
        "LOCAL_EMBEDDING_DIMENSION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCAL_EMBEDDING_DIMENSION", "3")
    runtime._load_config.cache_clear()
    servers: list[ModelServer] = []

    def create_client(config: ModelConfig) -> ModelClient:
        transport = ModelServer()
        transport.routes.update(routes)
        servers.append(transport)
        return ModelClient(config, transport=transport)

    monkeypatch.setattr(runtime, "_create_client", create_client)
    yield servers
    runtime._load_config.cache_clear()


def test_ready_client_reuses_one_connection_and_closes_on_exit(server) -> None:
    with runtime.local_model_runtime() as client:
        assert isinstance(client, ModelClient)
        assert client.config.base_url == "http://127.0.0.1:8080"
        assert len(server) == 1
        assert [request.url.path for request in server[0].requests] == [
            "/health",
            "/v1/models",
            "/v1/embeddings",
        ]
        assert not server[0].closed
        assert client.embed_query("another query") == [1.0, 0.0, 0.5]
        assert json.loads(server[0].requests[-1].content)["model"] == "bge-m3-q8_0"
        assert server[0].requests[-1].extensions["timeout"]["read"] == 120.0
    assert server[0].closed


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    elapsed = [0.0]
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        elapsed[0] += seconds

    monkeypatch.setattr(runtime, "monotonic", lambda: elapsed[0], raising=False)
    monkeypatch.setattr(runtime, "sleep", sleep, raising=False)
    return elapsed, waits


@pytest.mark.parametrize(
    "path,failure",
    [
        ("/health", "network"),
        ("/health", "status"),
        ("/v1/models", "network"),
        ("/v1/models", "status"),
        ("/v1/models", "missing"),
    ],
)
def test_transient_server_failure_recovers(server, routes, clock, path, failure):
    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if failure == "network":
                raise httpx.ConnectError("router is starting", request=request)
            if failure == "missing":
                return httpx.Response(200, json={"data": []})
            return httpx.Response(503)
        return ModelServer().respond(request)

    routes[path] = respond
    with runtime.local_model_runtime() as client:
        assert client.embed_query("ready") == [1.0, 0.0, 0.5]
    assert clock[1] == [0.5]
    assert server[0].closed


@pytest.mark.parametrize("path", ["/health", "/v1/models"])
@pytest.mark.parametrize("failure", ["network", "status"])
def test_server_deadline_chains_failure_and_closes(
    server, routes, clock, path, failure
):
    transport_error = httpx.ConnectError("connection refused")

    def respond(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            raise transport_error
        return httpx.Response(503)

    routes[path] = respond
    with (
        pytest.raises(runtime.LocalModelStartupError) as caught,
        runtime.local_model_runtime(),
    ):
        pytest.fail("An unavailable server must not yield a client")
    assert clock[0][0] == 5.0
    assert clock[1] == [0.5] * 10
    assert len([r for r in server[0].requests if r.url.path == path]) == 10
    assert caught.value.missing_model_ids == frozenset()
    if failure == "network":
        assert caught.value.__cause__ is transport_error
    else:
        assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
    assert server[0].closed


def test_missing_models_are_structured(server, routes, clock):
    routes["/v1/models"] = lambda _: httpx.Response(
        200, json={"data": [{"id": "bge-m3-q8_0"}]}
    )
    with (
        pytest.raises(runtime.LocalModelStartupError) as caught,
        runtime.local_model_runtime(),
    ):
        pytest.fail("A missing reranker must prevent startup")
    assert caught.value.missing_model_ids == frozenset({"bge-reranker-v2-m3-Q8_0"})
    assert caught.value.__cause__ is not None
    assert clock[0][0] == 5.0
    assert server[0].closed


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "null",
        "[]",
        "{}",
        '{"data":{}}',
        '{"data":[{}]}',
        '{"data":[null]}',
        '{"data":[{"id":12}]}',
    ],
)
def test_invalid_catalog_fails_immediately(server, routes, clock, body):
    routes["/v1/models"] = lambda _: httpx.Response(200, text=body)
    with (
        pytest.raises(runtime.LocalModelStartupError) as caught,
        runtime.local_model_runtime(),
    ):
        pytest.fail("An invalid catalog must prevent startup")
    assert isinstance(caught.value.__cause__, ValueError)
    assert caught.value.missing_model_ids == frozenset()
    assert clock[1] == []
    assert server[0].closed


@pytest.mark.parametrize(
    "embedding",
    [
        [1.0, 2.0],
        [],
        None,
        "not a vector",
        [0, 0, 0],
        [1, "bad", 0],
        [1, "2", 0],
        [1, True, 0],
        [1, None, 0],
        [1, {}, 0],
        [1, [2], 0],
        [1, float("nan"), 0],
        [1, float("inf"), 0],
        [1, float("-inf"), 0],
        [10**400, 0, 1],
    ],
)
def test_invalid_embedding_never_yields_and_closes(server, routes, embedding):
    routes["/v1/embeddings"] = lambda _: httpx.Response(
        200, text=json.dumps({"data": [{"index": 0, "embedding": embedding}]})
    )
    with (
        pytest.raises(runtime.LocalModelStartupError) as caught,
        runtime.local_model_runtime(),
    ):
        pytest.fail("An invalid embedding must prevent startup")
    assert caught.value.__cause__ is not None
    assert caught.value.missing_model_ids == frozenset()
    assert server[0].closed


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "null",
        "[]",
        "{}",
        '{"data":{}}',
        '{"data":[]}',
        '{"data":[null]}',
        '{"data":[{}]}',
        '{"data":[{"index":1,"embedding":[1,0,0]}]}',
        '{"data":[{"index":false,"embedding":[1,0,0]}]}',
        '{"data":[{"index":0.0,"embedding":[1,0,0]}]}',
    ],
)
def test_malformed_embedding_response_closes(server, routes, body):
    routes["/v1/embeddings"] = lambda _: httpx.Response(200, text=body)
    with pytest.raises(runtime.LocalModelStartupError), runtime.local_model_runtime():
        pytest.fail("Malformed embedding response must prevent startup")
    assert server[0].closed


def test_embedding_transport_failure_preserves_cause(server, routes):
    failure = httpx.ReadTimeout("cold model load timed out")

    def respond(_: httpx.Request) -> httpx.Response:
        raise failure

    routes["/v1/embeddings"] = respond
    with (
        pytest.raises(runtime.LocalModelStartupError) as caught,
        runtime.local_model_runtime(),
    ):
        pytest.fail("Failed embedding probe must prevent startup")
    assert caught.value.__cause__ is not None
    assert caught.value.__cause__.__cause__ is failure
    assert server[0].closed


def test_finite_small_nonzero_embedding_is_valid(server, routes):
    routes["/v1/embeddings"] = lambda _: httpx.Response(
        200, json={"data": [{"index": 0, "embedding": [1e-14, 0, 0]}]}
    )
    with runtime.local_model_runtime() as client:
        assert client.embed_query("small vector") == [1e-14, 0, 0]


def test_startup_status_is_visible_before_slow_work(server, routes, capsys):
    def health(request: httpx.Request) -> httpx.Response:
        assert "waiting for Model Server health" in capsys.readouterr().err
        return ModelServer().respond(request)

    def embedding(request: httpx.Request) -> httpx.Response:
        assert "loading and validating embedding model" in capsys.readouterr().err
        return ModelServer().respond(request)

    routes.update({"/health": health, "/v1/embeddings": embedding})
    with runtime.local_model_runtime():
        assert "Local Model Runtime: ready" in capsys.readouterr().err


def test_caller_failure_is_not_wrapped_and_client_is_closed(server):
    failure = ValueError("application failed after startup")
    with pytest.raises(ValueError) as caught, runtime.local_model_runtime():
        raise failure
    assert caught.value is failure
    assert server[0].closed


def test_config_is_immutable_and_shared_across_contexts(server, monkeypatch):
    with runtime.local_model_runtime() as first, pytest.raises(FrozenInstanceError):
        first.config.embedding_model = "changed"  # ty: ignore[invalid-assignment]
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://changed.invalid:9999")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "changed")
    monkeypatch.setenv("LOCAL_RERANKING_MODEL", "changed")
    monkeypatch.setenv("LOCAL_EMBEDDING_DIMENSION", "99")
    with runtime.local_model_runtime() as second:
        assert second is not first
        assert second.config is first.config
        assert second.embed_query("same model contract") == [1.0, 0.0, 0.5]
    assert len(server) == 2
    assert all(transport.closed for transport in server)


def test_default_model_contract(server, routes, monkeypatch):
    monkeypatch.delenv("LOCAL_EMBEDDING_DIMENSION")
    routes["/v1/embeddings"] = lambda _: httpx.Response(
        200, json={"data": [{"index": 0, "embedding": [1.0] * 1024}]}
    )
    with runtime.local_model_runtime() as client:
        assert client.config == ModelConfig(base_url="http://127.0.0.1:8080")


def test_environment_selects_endpoint_and_both_models(server, routes, monkeypatch):
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://models.local:9090")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "custom-embedding")
    monkeypatch.setenv("LOCAL_RERANKING_MODEL", "custom-reranking")
    routes["/v1/models"] = lambda _: httpx.Response(
        200, json={"data": [{"id": "custom-embedding"}, {"id": "custom-reranking"}]}
    )
    with runtime.local_model_runtime() as client:
        assert client.config.reranking_model == "custom-reranking"
        assert json.loads(server[0].requests[-1].content)["model"] == "custom-embedding"
        assert all(request.url.host == "models.local" for request in server[0].requests)


@pytest.mark.parametrize(
    "name,value",
    [
        ("LOCAL_EMBEDDING_DIMENSION", "0"),
        ("LOCAL_EMBEDDING_DIMENSION", "-1"),
        ("LOCAL_EMBEDDING_DIMENSION", "three"),
        ("LOCAL_EMBEDDING_MODEL", " "),
        ("LOCAL_RERANKING_MODEL", ""),
        ("LOCAL_MODEL_BASE_URL", ""),
        ("LOCAL_MODEL_BASE_URL", "ftp://localhost"),
        ("LOCAL_MODEL_BASE_URL", "http://"),
    ],
)
def test_invalid_environment_fails_before_opening_client(
    server, monkeypatch, name, value
):
    monkeypatch.setenv(name, value)
    with (
        pytest.raises(runtime.LocalModelStartupError) as caught,
        runtime.local_model_runtime(),
    ):
        pytest.fail("Invalid configuration must prevent startup")
    assert isinstance(caught.value.__cause__, ValueError)
    assert not server
