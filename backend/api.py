"""FastAPI entry point with explicit application resource ownership."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field, StringConstraints

from backend.agents.graph import ConversationMessage, RagWorkflow
from backend.rag import get_required_env
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.model_client import DEFAULT_MODEL_CONFIG, ModelClient
from backend.storage.pg_search_store import PgSearchStore
from backend.utils.llama_server import ManagedLlamaServer


@dataclass(frozen=True)
class AppResources:
    """Long-lived dependencies shared by synchronous request handlers."""

    workflow: RagWorkflow


@contextmanager
def application_resources() -> Iterator[AppResources]:
    """Start and cleanly release the model, database, and HTTP clients."""
    models_dir = Path(__file__).resolve().parent / "models"
    with (
        ManagedLlamaServer(models_dir=models_dir, port=8080),
        ModelClient(DEFAULT_MODEL_CONFIG) as model_client,
        PgSearchStore(
            embedding_dimension=DEFAULT_MODEL_CONFIG.embedding_dimension,
            verbose=False,
        ) as search_store,
        OpenAI(api_key=get_required_env("OPENAI_API_KEY")) as generation_client,
    ):
        yield AppResources(
            workflow=RagWorkflow(
                HybridRetriever(search_store, model_client),
                generation_client,
            )
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Bind resource lifetime to FastAPI startup and shutdown."""
    with application_resources() as resources:
        app.state.resources = resources
        yield


app = FastAPI(lifespan=lifespan)
default_cors_origins = "http://localhost:4321,http://127.0.0.1:4321"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("RAG_CORS_ORIGINS", default_cors_origins).split(
            ","
        )
        if origin.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


MessageText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
]


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: MessageText


class ChatRequest(BaseModel):
    message: MessageText
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    answer: str
    found: bool
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Run blocking RAG work in FastAPI's synchronous worker pool."""
    resources: AppResources = request.app.state.resources
    result = resources.workflow.invoke(
        payload.message,
        history=[
            ConversationMessage(message.role, message.content)
            for message in payload.history
        ],
    )
    return ChatResponse(
        answer=result.answer,
        found=result.found,
        confidence=result.confidence,
        sources=list(result.sources),
    )
