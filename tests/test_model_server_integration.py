"""Real-model gate for the persistent local Model Server."""

from __future__ import annotations

import math
import os

import pytest

from backend.local_model_runtime import local_model_runtime


@pytest.mark.model_integration
@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_INTEGRATION") != "1",
    reason="Set RUN_MODEL_INTEGRATION=1 to exercise local GGUF models",
)
def test_persistent_server_satisfies_runtime_contract() -> None:
    documents = [
        "Super-resolution enhances image detail.",
        "Tokenization splits text into model inputs.",
    ]

    with local_model_runtime() as client:
        client.check_health(timeout=5.0)
        available_models = client.list_models(timeout=5.0)
        required_models = {
            client.config.embedding_model,
            client.config.reranking_model,
        }
        embedding = client.embed_query("How does tokenization work?")
        scores = client.rerank("What is super-resolution?", documents)

    assert required_models <= available_models
    assert len(embedding) == client.config.embedding_dimension
    assert all(math.isfinite(value) for value in embedding)
    assert any(embedding)
    assert len(scores) == len(documents)
    assert {score.document_index for score in scores} == set(range(len(documents)))
    assert all(math.isfinite(score.relevance_score) for score in scores)
