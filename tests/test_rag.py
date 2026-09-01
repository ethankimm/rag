"""Conversation rewriting and bounded LangGraph RAG workflow tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import OpenAI

from backend.agents.graph import (
    DEFAULT_GENERATION_MODEL,
    NO_DATA_ANSWER,
    UNSURE_PREFIX,
    ConversationMessage,
    GraphResponseError,
    RagWorkflow,
    bounded_history,
    format_context,
    format_history,
    get_float_env,
    get_generation_model,
    rewrite_query,
)
from backend.rag import answer_question, llm_call
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.storage.pg_search_store import SearchHit


class FakeRetriever:
    def __init__(self, responses: list[list[SearchHit]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, top_k: int) -> list[SearchHit]:
        self.calls.append((query, top_k))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index] if self.responses else []


class FakeCompletions:
    def __init__(self, responses: list[str | None]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected completion call")
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def fake_generation_client(
    responses: list[str | None],
) -> tuple[OpenAI, FakeCompletions]:
    completions = FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return cast(OpenAI, client), completions


def hit(
    chunk_id: str = "chunk-1",
    source: str = "chapter1/1.md",
    content: str = "Course evidence",
) -> SearchHit:
    return SearchHit(chunk_id, content, {"source_file": source}, 1.0)


def test_bounded_history_validates_and_keeps_the_latest_twelve_turns() -> None:
    history = [ConversationMessage("user", f"turn {index}") for index in range(14)]

    bounded = bounded_history(history)

    assert len(bounded) == 12
    assert bounded[0].content == "turn 2"
    with pytest.raises(ValueError, match="blank"):
        bounded_history([ConversationMessage("assistant", "  ")])


def test_format_history_handles_empty_and_populated_conversations() -> None:
    assert format_history([]) == "No prior conversation."
    assert (
        format_history([ConversationMessage("user", "What is an LLM?")])
        == "user: What is an LLM?"
    )


def test_rewrite_query_skips_first_turn_without_calling_openai() -> None:
    client, completions = fake_generation_client([])

    assert rewrite_query("  transformers  ", [], client) == "transformers"
    assert completions.calls == []


def test_rewrite_query_uses_history_and_falls_back_on_blank_output() -> None:
    history = [ConversationMessage("user", "Tell me about tokenizers")]
    client, completions = fake_generation_client(
        ["How do Hugging Face tokenizers work?", "  "]
    )

    assert (
        rewrite_query("How do they work?", history, client)
        == "How do Hugging Face tokenizers work?"
    )
    assert rewrite_query("And training?", history, client) == "And training?"
    assert completions.calls[0]["temperature"] == 0.0
    assert completions.calls[0]["max_tokens"] == 128
    assert completions.calls[0]["model"] == DEFAULT_GENERATION_MODEL
    rewrite_prompt = completions.calls[0]["messages"][1]["content"]
    assert "Tell me about tokenizers" in rewrite_prompt


def test_format_context_bounds_evidence_and_includes_sources() -> None:
    assert format_context([]) == "No relevant evidence was retrieved."
    context = format_context([hit(content="x" * 4000)])
    assert "source: chapter1/1.md" in context
    assert len(context.splitlines()[1]) == 3000


def test_get_float_env_validates_numeric_range(monkeypatch) -> None:
    monkeypatch.setenv("TEST_SETTING", "0.25")
    assert get_float_env("TEST_SETTING", 0.5, minimum=0.0, maximum=1.0) == 0.25
    monkeypatch.setenv("TEST_SETTING", "invalid")
    with pytest.raises(RuntimeError, match="must be a number"):
        get_float_env("TEST_SETTING", 0.5, minimum=0.0, maximum=1.0)


def test_generation_model_defaults_and_reads_environment(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert get_generation_model() == "gpt-4.1-nano"

    monkeypatch.setenv("OPENAI_MODEL", "  env-small-model  ")
    assert get_generation_model() == "env-small-model"


def test_generation_model_override_precedes_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    assert get_generation_model(" explicit-model ") == "explicit-model"


def test_generation_model_rejects_blank_configuration(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "  ")
    with pytest.raises(RuntimeError, match="OPENAI_MODEL must not be blank"):
        get_generation_model()
    with pytest.raises(ValueError, match="generation_model must not be blank"):
        get_generation_model("  ")


def test_workflow_returns_confident_grounded_answer_and_sources(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    retriever = FakeRetriever([[hit()]])
    client, completions = fake_generation_client(
        [
            "Transformers use attention.",
            '{"confidence": 0.92, "missing_information": []}',
        ]
    )
    workflow = RagWorkflow(
        cast(HybridRetriever, retriever),
        client,
        confidence_threshold=0.5,
        max_attempts=1,
    )

    result = workflow.invoke("How do Transformers work?")

    assert result.answer == "Transformers use attention."
    assert result.found
    assert result.confidence == 0.92
    assert result.sources == ("chapter1/1.md",)
    assert retriever.calls == [("How do Transformers work?", 5)]
    assert len(completions.calls) == 2
    assert {call["model"] for call in completions.calls} == {DEFAULT_GENERATION_MODEL}


def test_workflow_uses_generation_model_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "env-small-model")
    retriever = FakeRetriever([[hit()]])
    client, completions = fake_generation_client(
        ["Grounded answer", '{"confidence": 0.9, "missing_information": []}']
    )

    result = RagWorkflow(
        cast(HybridRetriever, retriever),
        client,
        confidence_threshold=0.5,
        max_attempts=1,
    ).invoke("Question")

    assert result.found
    assert {call["model"] for call in completions.calls} == {"env-small-model"}


def test_workflow_rewrites_contextual_follow_up_for_initial_retrieval() -> None:
    retriever = FakeRetriever([[hit()]])
    client, completions = fake_generation_client(
        [
            "How are Hugging Face tokenizers trained?",
            "They are trained from an iterator.",
            '{"confidence": 0.9, "missing_information": []}',
        ]
    )
    workflow = RagWorkflow(
        cast(HybridRetriever, retriever),
        client,
        confidence_threshold=0.5,
        max_attempts=1,
    )

    result = workflow.invoke(
        "How are they trained?",
        history=[
            ConversationMessage("user", "Tell me about Hugging Face tokenizers"),
            ConversationMessage("assistant", "They convert text into tokens."),
        ],
    )

    assert result.found
    assert retriever.calls[0][0] == "How are Hugging Face tokenizers trained?"
    draft_prompt = completions.calls[1]["messages"][1]["content"]
    assert "They convert text into tokens." in draft_prompt
    assert "How are they trained?" in draft_prompt


def test_workflow_runs_bounded_follow_up_and_revision_path() -> None:
    retriever = FakeRetriever(
        [[hit()], [hit("chunk-2", "chapter2/2.md", "Additional evidence")]]
    )
    client, completions = fake_generation_client(
        [
            "Partial answer",
            '{"confidence": 0.2, "missing_information": ["training details"]}',
            '{"questions": ["How are tokenizers trained?"]}',
            "Complete answer",
            '{"confidence": 0.88, "missing_information": []}',
        ]
    )
    workflow = RagWorkflow(
        cast(HybridRetriever, retriever),
        client,
        generation_model="test-small-model",
        confidence_threshold=0.5,
        max_attempts=2,
    )

    result = workflow.invoke("Explain tokenizers")

    assert result.answer == "Complete answer"
    assert result.found
    assert result.sources == ("chapter1/1.md", "chapter2/2.md")
    assert retriever.calls[1][0] == "How are tokenizers trained?"
    assert {call["model"] for call in completions.calls} == {"test-small-model"}


def test_workflow_marks_low_confidence_final_answer_unsure() -> None:
    retriever = FakeRetriever([[hit()]])
    client, _ = fake_generation_client(
        ["Weak answer", '{"confidence": 0.1, "missing_information": ["facts"]}']
    )
    workflow = RagWorkflow(
        cast(HybridRetriever, retriever),
        client,
        confidence_threshold=0.5,
        max_attempts=1,
    )

    result = workflow.invoke("Question")

    assert result.answer == f"{UNSURE_PREFIX}Weak answer"
    assert not result.found
    assert result.sources == ("chapter1/1.md",)


def test_workflow_rejects_invalid_classifier_contract() -> None:
    retriever = FakeRetriever([[hit()]])
    client, _ = fake_generation_client(["Answer", "not json"])
    workflow = RagWorkflow(
        cast(HybridRetriever, retriever),
        client,
        max_attempts=1,
    )

    with pytest.raises(GraphResponseError, match="invalid JSON"):
        workflow.invoke("Question")


def test_compatibility_wrappers_forward_history() -> None:
    retriever = FakeRetriever([[hit()], [hit()]])
    history = [ConversationMessage("user", "Transformers")]
    client, _ = fake_generation_client(
        [
            "Transformer architecture",
            "Answer one",
            '{"confidence": 0.9, "missing_information": []}',
            "Transformer architecture",
            "Answer two",
            '{"confidence": 0.9, "missing_information": []}',
        ]
    )

    result = answer_question(
        "How does it work?",
        cast(HybridRetriever, retriever),
        client,
        history=history,
    )
    text = llm_call(
        "How does it work?",
        cast(HybridRetriever, retriever),
        client,
        history=history,
    )

    assert result.answer == "Answer one"
    assert text == "Answer two"
    assert result.answer != NO_DATA_ANSWER
