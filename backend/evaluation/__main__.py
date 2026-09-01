"""CLI entry point: ``python -m backend.evaluation``.

Download BEIR SciFact, index its corpus into Postgres + pgvector, run the
built-in hybrid retrieval over test queries, and output benchmark metrics to JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # noqa: E402

from backend.evaluation.beir_scifact import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_LEXICAL_COLLECTION_NAME,
    build_pg_index,
    build_pg_lexical_index,
    corpus_to_documents,
    evaluate,
    format_report,
    load_scifact,
    resolve_k_values,
    run_hybrid_pipeline,
    run_lexical_pipeline,
)
from backend.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from backend.retrieval.model_client import (  # noqa: E402
    DEFAULT_MODEL_CONFIG,
    ModelClient,
)
from backend.storage.pg_search_store import (  # noqa: E402
    BM25_B,
    BM25_K1,
    DEFAULT_EMBEDDING_DIM,
    PgSearchStore,
)
from backend.utils.llama_server import ManagedLlamaServer  # noqa: E402


def _get_git_commit() -> str:
    """Attempt to get the short hash of the current git commit."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _report_and_save(
    metrics: dict[str, float],
    *,
    top_k: int,
    run_metadata: dict[str, object],
    file_stem: str,
) -> None:
    """Print an evaluation report and persist its JSON payload."""
    print("\n" + format_report(metrics, top_k=top_k))

    json_metrics = {
        key.replace("Accuracy@", "HitRate@"): round(value, 3)
        for key, value in metrics.items()
    }
    timestamp = datetime.now(UTC)
    payload = {
        "dataset": "beir/scifact",
        "split": "test",
        "run": run_metadata,
        "metrics": json_metrics,
        "recorded_at": timestamp.isoformat(timespec="seconds"),
        "commit": _get_git_commit(),
    }

    results_dir = REPO_ROOT / "backend" / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{file_stem}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    out_path = results_dir / file_name
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"\nSaved detailed JSON results to: {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    """Run the BEIR SciFact evaluation and output results to JSON."""
    args = _parse_args()
    models_dir = REPO_ROOT / "backend" / "models"
    models_preset = models_dir / "llama-models.ini"
    collection_name = args.collection_name or (
        DEFAULT_LEXICAL_COLLECTION_NAME
        if args.lexical_only
        else DEFAULT_COLLECTION_NAME
    )

    print("Loading BEIR SciFact test split...")
    corpus, queries, qrels = load_scifact(args.data_root)
    print(f"Loaded {len(corpus)} corpus documents and {len(queries)} test queries")
    documents = corpus_to_documents(corpus)

    if args.lexical_only:
        print("Building native pg_textsearch BM25 index (lexical only)...")
        with PgSearchStore(
            collection_name=collection_name,
            embedding_dimension=None,
            dsn=args.dsn,
        ) as lexical_store:
            build_pg_lexical_index(
                documents,
                lexical_store,
                rebuild=args.rebuild,
            )
            print(f"Running BM25-only retrieval over {len(queries)} queries...")
            run = run_lexical_pipeline(
                queries,
                lexical_store,
                collection_name=collection_name,
                retrieval_depth=100,
            )
        k_values = sorted({*resolve_k_values(args.top_k), 100})
        metrics = evaluate(qrels, run, k_values)
        _report_and_save(
            metrics,
            top_k=args.top_k,
            run_metadata={
                "retrieval": "bm25",
                "extension": "pg_textsearch",
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
                "retrieval_depth": 100,
                "reranking_model": None,
                "top_k": args.top_k,
            },
            file_stem="scifact_lexical_results",
        )
        return

    # llama-server router context supporting both embedding and reranking
    with (
        ManagedLlamaServer(
            models_dir=models_dir,
            models_preset=models_preset,
            port=8080,
        ),
        ModelClient(DEFAULT_MODEL_CONFIG) as model_client,
        PgSearchStore(
            collection_name=collection_name,
            embedding_dimension=DEFAULT_EMBEDDING_DIM,
            dsn=args.dsn,
        ) as vector_store,
    ):
        print("Building pgvector + pg_textsearch hybrid index...")
        build_pg_index(
            documents,
            model_client,
            vector_store,
            rebuild=args.rebuild,
        )

        retriever = HybridRetriever(vector_store, model_client)

        print(f"Running hybrid retrieval over {len(queries)} queries...")
        run = run_hybrid_pipeline(
            queries,
            retriever,
            retrieval_depth=100,
            rerank_depth=args.rerank_depth,
        )

        metrics = evaluate(qrels, run, resolve_k_values(args.top_k))

        _report_and_save(
            metrics,
            top_k=args.top_k,
            run_metadata={
                "embedding_model": model_client.config.embedding_model,
                "reranking_model": model_client.config.reranking_model,
                "semantic_weight": 0.5,
                "lexical_weight": 0.5,
                "retrieval_depth": 100,
                "rerank_depth": args.rerank_depth,
                "top_k": args.top_k,
            },
            file_stem="scifact_results",
        )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation run."""
    parser = argparse.ArgumentParser(
        description="Evaluate the pgvector hybrid retriever on BEIR SciFact.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory to download and cache the BEIR SciFact dataset.",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Postgres DSN. Defaults to $PG_DSN or the local docker-compose DSN.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Postgres table name. Defaults depend on the selected retrieval mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Cutoff for the headline Hit Rate and evaluation depth (max 10).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a fresh embed-and-index pass even if the collection exists.",
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=50,
        help="Number of fused candidates to cross-encode; 0 disables reranking.",
    )
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Evaluate native pg_textsearch BM25 without models or dense retrieval.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
