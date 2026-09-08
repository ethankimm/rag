"""Query-time retrieval for the local RAG prototype.

Dense + lexical search over Postgres, then cross-encoder rerank.
"""

from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.model_client import (
    DEFAULT_MODEL_CONFIG,
    ModelClient,
    ModelConfig,
    RerankScore,
)

__all__ = [
    "DEFAULT_MODEL_CONFIG",
    "HybridRetriever",
    "ModelClient",
    "ModelConfig",
    "RerankScore",
]
