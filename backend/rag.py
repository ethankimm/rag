"""RAG orchestration for retrieving context and calling the LLM.

Run from the repo root:
    .venv/bin/python -m backend.rag
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from backend.agents.graph import (
    DEFAULT_GENERATION_MODEL,
    NO_DATA_ANSWER,
    ConversationMessage,
    RagAnswer,
    RagWorkflow,
    format_context,
    get_float_env,
    get_generation_model,
)
from backend.local_model_runtime import local_model_runtime
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.storage.pg_search_store import PgSearchStore

BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent
SMOKE_QUESTION = "What changed in the 2024 Q2 weight package count enhancements?"

load_dotenv(REPO_ROOT / ".env")


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Create .env from .env.example and set a local value."
        )
    return value


def answer_question(
    query_text: str,
    retriever: HybridRetriever,
    generation_client: OpenAI,
    *,
    history: Sequence[ConversationMessage] = (),
    generation_model: str | None = None,
    confidence_threshold: float | None = None,
    temperature: float | None = None,
) -> RagAnswer:
    """Compatibility entry point that executes the bounded LangGraph workflow."""
    return RagWorkflow(
        retriever,
        generation_client,
        generation_model=generation_model,
        confidence_threshold=confidence_threshold,
        temperature=temperature,
    ).invoke(query_text, history=history)


def llm_call(
    query_text: str,
    retriever: HybridRetriever,
    generation_client: OpenAI,
    *,
    history: Sequence[ConversationMessage] = (),
    generation_model: str | None = None,
) -> str:
    """Compatibility wrapper returning only the generated answer text."""
    return answer_question(
        query_text,
        retriever,
        generation_client,
        history=history,
        generation_model=generation_model,
    ).answer


def run_smoke() -> None:
    """Run one interactive query through production resource assembly."""
    with (
        local_model_runtime() as model_client,
        PgSearchStore(
            embedding_dimension=model_client.config.embedding_dimension,
            verbose=False,
        ) as search_store,
        OpenAI(api_key=get_required_env("OPENAI_API_KEY")) as generation_client,
    ):
        hybrid_retriever = HybridRetriever(search_store, model_client)
        print(f"Question: {SMOKE_QUESTION}\n")
        print(llm_call(SMOKE_QUESTION, hybrid_retriever, generation_client))


__all__ = [
    "ConversationMessage",
    "DEFAULT_GENERATION_MODEL",
    "NO_DATA_ANSWER",
    "RagAnswer",
    "RagWorkflow",
    "answer_question",
    "format_context",
    "get_float_env",
    "get_generation_model",
    "get_required_env",
    "llm_call",
    "run_smoke",
]


if __name__ == "__main__":
    run_smoke()
