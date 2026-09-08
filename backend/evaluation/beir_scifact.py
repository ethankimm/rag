"""Evaluate the pgvector hybrid retriever against the BEIR SciFact benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.model_client import ModelClient
from backend.storage.pg_search_store import (
    CollectionManifest,
    PgSearchStore,
    SearchDocument,
)

EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = EVALUATION_ROOT / "data" / "beir"
DEFAULT_COLLECTION_NAME = "beir_scifact"
DEFAULT_LEXICAL_COLLECTION_NAME = "beir_scifact_lexical"

_REPORT_METRICS: list[tuple[str, str]] = [
    ("Accuracy@1", "Hit Rate@1"),
    ("Accuracy@5", "Hit Rate@5"),
    ("Accuracy@10", "Hit Rate@10"),
    ("Recall@10", "Recall@10"),
    ("NDCG@10", "NDCG@10"),
    ("MAP@10", "MAP@10"),
    ("Recall@100", "Recall@100"),
]


def load_beir_dataset(
    dataset_name: str = "scifact",
    data_root: Path | str = DEFAULT_DATA_ROOT,
    split: str = "test",
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    """Download BEIR dataset if missing and load the specified split."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_dir = Path(data_root)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = data_dir / dataset_name
    if not dataset_dir.exists():
        dataset_url = (
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
            f"{dataset_name}.zip"
        )
        util.download_and_unzip(dataset_url, str(data_dir))

    return GenericDataLoader(data_folder=str(dataset_dir)).load(split=split)


def load_scifact(
    data_root: Path | str,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    """Download BEIR SciFact if missing and load its test split."""
    return load_beir_dataset(
        dataset_name="scifact",
        data_root=data_root,
        split="test",
    )


def corpus_to_documents(corpus: dict[str, dict[str, str]]) -> list[Document]:
    """Convert a BEIR corpus into chunk documents keyed by document id."""
    documents: list[Document] = []
    for doc_id, record in corpus.items():
        title = record.get("title", "")
        text = record.get("text", "")
        page_content = f"{title}\n{text}".strip()
        documents.append(
            Document(page_content=page_content, metadata={"chunk_id": str(doc_id)})
        )
    return documents


def embed_chunks_via_http(
    chunks: list[Document],
    model_client: ModelClient,
    batch_size: int = 4,
) -> list[list[float]]:
    """Embed benchmark documents through the canonical model client."""
    return model_client.embed_documents(
        [chunk.page_content for chunk in chunks],
        batch_size=batch_size,
    )


def build_pg_index(
    documents: list[Document],
    model_client: ModelClient,
    store: PgSearchStore,
    *,
    rebuild: bool = False,
) -> None:
    """Embed and index documents in a caller-owned PostgreSQL store."""
    embedding_dim = store.embedding_dimension
    if embedding_dim is None:
        raise ValueError("Hybrid evaluation requires an embedding-enabled store")
    if embedding_dim != model_client.config.embedding_dimension:
        raise ValueError("Evaluation dimension must match the configured model client")

    manifest = CollectionManifest(
        document_count=len(documents),
        embedding_model=model_client.config.embedding_model,
        embedding_dimension=embedding_dim,
    )
    if not rebuild and store.can_reuse(manifest):
        print(
            f"Reusing {len(documents)} indexed documents from "
            f"'{store.collection_name}'."
        )
        return

    print(
        f"Embedding {len(documents)} documents for collection "
        f"'{store.collection_name}'..."
    )
    vectors = embed_chunks_via_http(documents, model_client)
    store.replace_documents(
        [
            SearchDocument(
                chunk_id=str(document.metadata["chunk_id"]),
                content=document.page_content,
                metadata=document.metadata,
                embedding=np.asarray(vector, dtype=np.float32),
            )
            for document, vector in zip(documents, vectors, strict=True)
        ],
        manifest,
    )


def build_pg_lexical_index(
    documents: list[Document],
    store: PgSearchStore,
    *,
    rebuild: bool = False,
) -> None:
    """Load text into a caller-owned native pg_textsearch BM25 store."""
    if store.has_embeddings:
        raise ValueError("Lexical evaluation requires a text-only store")
    manifest = CollectionManifest(len(documents), None, None)
    if not rebuild and store.can_reuse(manifest):
        print(
            f"Reusing {len(documents)} lexical documents from "
            f"'{store.collection_name}'."
        )
        return

    print(
        f"Loading {len(documents)} lexical documents into '{store.collection_name}'..."
    )
    store.replace_documents(
        [
            SearchDocument(
                chunk_id=str(document.metadata["chunk_id"]),
                content=document.page_content,
                metadata=document.metadata,
            )
            for document in documents
        ],
        manifest,
    )


def run_lexical_pipeline(
    queries: dict[str, str],
    store: PgSearchStore,
    *,
    collection_name: str = DEFAULT_LEXICAL_COLLECTION_NAME,
    retrieval_depth: int = 100,
) -> dict[str, dict[str, float]]:
    """Run only native BM25 retrieval and build a BEIR run dictionary."""
    run: dict[str, dict[str, float]] = {}
    for query_id, query_text in queries.items():
        if store.collection_name != collection_name:
            raise ValueError("Lexical store does not match requested collection")
        candidates = store.bm25_search(query_text, limit=retrieval_depth)
        total = len(candidates)
        run[query_id] = {
            candidate.chunk_id: float(total - position)
            for position, candidate in enumerate(candidates)
        }
    return run


def run_hybrid_pipeline(
    queries: dict[str, str],
    retriever: HybridRetriever,
    *,
    retrieval_depth: int = 100,
    rerank_depth: int = 50,
) -> dict[str, dict[str, float]]:
    """Retrieve hybrid candidates and build a BEIR run dictionary."""
    run: dict[str, dict[str, float]] = {}
    for query_id, query_text in queries.items():
        candidates = retriever.retrieve(
            query_text,
            top_k=retrieval_depth,
            rerank_limit=rerank_depth,
        )
        total = len(candidates)

        run[query_id] = {
            candidate.chunk_id: float(total - position)
            for position, candidate in enumerate(candidates)
        }
    return run


def evaluate(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, float]:
    """Compute NDCG, MAP, Recall, and Hit Rate (top-k accuracy) metrics."""
    from beir.retrieval.evaluation import EvaluateRetrieval

    ndcg, mean_average_precision, recall, precision = EvaluateRetrieval.evaluate(
        qrels, run, k_values
    )
    hit_rate = EvaluateRetrieval.evaluate_custom(
        qrels, run, k_values, metric="top_k_accuracy"
    )

    metrics: dict[str, float] = {}
    for metric_group in (ndcg, mean_average_precision, recall, precision, hit_rate):
        metrics.update(metric_group)
    return metrics


def format_percent(score: float) -> str:
    """Render a 0..1 metric score as a one-decimal percentage."""
    return f"{score * 100:.1f}%"


def format_report(metrics: dict[str, float], *, top_k: int) -> str:
    """Build the CLI report: headline percent correct plus a metric breakdown."""
    headline_score = metrics.get(f"Accuracy@{top_k}", 0.0)
    lines = [
        f"Total percent correct (Hit Rate@{top_k}): {format_percent(headline_score)}",
        "",
        "Breakdown:",
    ]
    for metric_key, label in _REPORT_METRICS:
        if metric_key in metrics:
            lines.append(f"  {label:<14} {format_percent(metrics[metric_key])}")
    return "\n".join(lines)


def resolve_k_values(top_k: int) -> list[int]:
    """Metric cutoffs to request, capping at 10 to match retriever limits."""
    return sorted({1, 5, 10, top_k})
