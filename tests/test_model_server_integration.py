"""Optional integration test for llama-server model routing."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from backend.retrieval.model_client import DEFAULT_MODEL_CONFIG
from backend.utils.llama_server import ManagedLlamaServer

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "backend" / "models"


@pytest.mark.model_integration
@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_INTEGRATION") != "1",
    reason="Set RUN_MODEL_INTEGRATION=1 to exercise local GGUF models",
)
def test_llama_server_routes_both_bge_models() -> None:
    with (
        ManagedLlamaServer(models_dir=MODELS_DIR, port=8080),
        httpx.Client(base_url="http://127.0.0.1:8080", timeout=60.0) as client,
    ):
        models_response = client.get("/v1/models")
        assert models_response.status_code == 200
        available_models = {
            model["id"] for model in models_response.json().get("data", [])
        }
        assert DEFAULT_MODEL_CONFIG.embedding_model in available_models
        assert DEFAULT_MODEL_CONFIG.reranking_model in available_models

        embedding_response = client.post(
            "/v1/embeddings",
            json={
                "model": DEFAULT_MODEL_CONFIG.embedding_model,
                "input": "Testing model load",
            },
        )
        assert embedding_response.status_code == 200

        rerank_response = client.post(
            "/rerank",
            json={
                "model": DEFAULT_MODEL_CONFIG.reranking_model,
                "query": "What is super-resolution?",
                "top_n": 1,
                "documents": ["Super-resolution enhances image detail."],
            },
        )
        if rerank_response.status_code == 404:
            rerank_response = client.post(
                "/v1/rerank",
                json={
                    "model": DEFAULT_MODEL_CONFIG.reranking_model,
                    "query": "What is super-resolution?",
                    "top_n": 1,
                    "documents": ["Super-resolution enhances image detail."],
                },
            )
        assert rerank_response.status_code == 200
