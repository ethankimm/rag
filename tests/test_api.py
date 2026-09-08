"""FastAPI lifecycle, validation, and multi-turn endpoint tests."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

import backend.api as api
from backend.agents.graph import ConversationMessage, RagAnswer, RagWorkflow
from backend.local_model_runtime import LocalModelStartupError


class FakeWorkflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[ConversationMessage, ...]]] = []

    def invoke(
        self,
        question: str,
        *,
        history: Sequence[ConversationMessage] = (),
    ) -> RagAnswer:
        self.calls.append((question, tuple(history)))
        return RagAnswer("answer", True, 0.91, ("chapter1/1.md",))


def test_chat_is_synchronous() -> None:
    assert not inspect.iscoroutinefunction(api.chat)


def test_lifespan_owns_resources_and_chat_forwards_history(monkeypatch) -> None:
    lifecycle: list[str] = []
    workflow = FakeWorkflow()

    @contextmanager
    def fake_resources():
        lifecycle.append("started")
        yield api.AppResources(cast(RagWorkflow, workflow))
        lifecycle.append("closed")

    monkeypatch.setattr(api, "application_resources", fake_resources)

    with TestClient(api.app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "  How does it work?  ",
                "history": [
                    {"role": "user", "content": "Tell me about Transformers"},
                    {"role": "assistant", "content": "They use attention."},
                ],
            },
        )
        assert response.json() == {
            "answer": "answer",
            "found": True,
            "confidence": 0.91,
            "sources": ["chapter1/1.md"],
        }
        assert lifecycle == ["started"]

    assert lifecycle == ["started", "closed"]
    assert workflow.calls == [
        (
            "How does it work?",
            (
                ConversationMessage("user", "Tell me about Transformers"),
                ConversationMessage("assistant", "They use attention."),
            ),
        )
    ]


def test_application_resources_uses_validated_local_model_runtime(monkeypatch) -> None:
    lifecycle: list[str] = []
    model_client = SimpleNamespace(config=SimpleNamespace(embedding_dimension=384))
    generation_client = object()

    @contextmanager
    def fake_local_model_runtime():
        lifecycle.append("models started")
        yield model_client
        lifecycle.append("models closed")

    @contextmanager
    def fake_search_store(*, embedding_dimension: int, verbose: bool):
        assert embedding_dimension == 384
        assert not verbose
        lifecycle.append("store started")
        yield SimpleNamespace(embedding_dimension=embedding_dimension)
        lifecycle.append("store closed")

    @contextmanager
    def fake_openai(*, api_key: str):
        assert api_key == "test-key"
        lifecycle.append("generation started")
        yield generation_client
        lifecycle.append("generation closed")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(api, "local_model_runtime", fake_local_model_runtime)
    monkeypatch.setattr(api, "PgSearchStore", fake_search_store)
    monkeypatch.setattr(api, "OpenAI", fake_openai)

    with api.application_resources() as resources:
        assert resources.workflow.retriever.model_client is model_client
        assert resources.workflow.generation_client is generation_client
        assert lifecycle == [
            "models started",
            "store started",
            "generation started",
        ]

    assert lifecycle == [
        "models started",
        "store started",
        "generation started",
        "generation closed",
        "store closed",
        "models closed",
    ]


def test_runtime_startup_failure_prevents_api_resource_construction(
    monkeypatch,
) -> None:
    cause = ConnectionError("model server unavailable")
    startup_error = LocalModelStartupError("startup failed")

    @contextmanager
    def failed_runtime():
        raise startup_error from cause
        yield

    def unexpected_construction(*args, **kwargs):
        pytest.fail("Downstream resources must not be constructed")

    monkeypatch.setattr(api, "local_model_runtime", failed_runtime)
    monkeypatch.setattr(api, "PgSearchStore", unexpected_construction)
    monkeypatch.setattr(api, "OpenAI", unexpected_construction)

    with (
        pytest.raises(LocalModelStartupError) as caught,
        api.application_resources(),
    ):
        pytest.fail("Failed runtime must not yield application resources")

    assert caught.value is startup_error
    assert caught.value.__cause__ is cause


def test_chat_remains_backward_compatible_without_history(monkeypatch) -> None:
    workflow = FakeWorkflow()

    @contextmanager
    def fake_resources():
        yield api.AppResources(cast(RagWorkflow, workflow))

    monkeypatch.setattr(api, "application_resources", fake_resources)
    with TestClient(api.app) as client:
        response = client.post("/chat", json={"message": "question"})

    assert response.status_code == 200
    assert workflow.calls == [("question", ())]


def test_chat_validates_message_roles_lengths_and_history_limit(monkeypatch) -> None:
    workflow = FakeWorkflow()

    @contextmanager
    def fake_resources():
        yield api.AppResources(cast(RagWorkflow, workflow))

    monkeypatch.setattr(api, "application_resources", fake_resources)
    with TestClient(api.app) as client:
        assert client.post("/chat", json={"message": "   "}).status_code == 422
        assert (
            client.post(
                "/chat",
                json={
                    "message": "question",
                    "history": [{"role": "system", "content": "instruction"}],
                },
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/chat",
                json={
                    "message": "question",
                    "history": [
                        {"role": "user", "content": str(index)} for index in range(13)
                    ],
                },
            ).status_code
            == 422
        )

    assert workflow.calls == []
