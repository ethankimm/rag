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
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.model_client import DEFAULT_MODEL_CONFIG, ModelClient
from backend.storage.pg_search_store import PgSearchStore
from backend.utils.llama_server import ManagedLlamaServer

BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent

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
]


if __name__ == "__main__":
    MODELS_DIR = Path(__file__).resolve().parent / "models"

    with (
        ManagedLlamaServer(models_dir=MODELS_DIR, port=8080),
        ModelClient(DEFAULT_MODEL_CONFIG) as local_models,
        PgSearchStore(
            embedding_dimension=DEFAULT_MODEL_CONFIG.embedding_dimension,
            verbose=False,
        ) as search_store,
        OpenAI(api_key=get_required_env("OPENAI_API_KEY")) as generation_client,
    ):
        hybrid_retriever = HybridRetriever(search_store, local_models)
        question = "What changed in the 2024 Q2 weight package count enhancements?"
        print(f"Question: {question}\n")
        print(llm_call(question, hybrid_retriever, generation_client))
