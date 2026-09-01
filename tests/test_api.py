"""FastAPI lifecycle, validation, and multi-turn endpoint tests."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from contextlib import contextmanager
from typing import cast

from fastapi.testclient import TestClient

import backend.api as api
from backend.agents.graph import ConversationMessage, RagAnswer, RagWorkflow


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
