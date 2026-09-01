"""Markdown ingestion pipeline: load -> chunk -> assign IDs -> embed -> index.

Run via CLI: uv run python -m backend.ingestion
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chardet
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.retrieval.model_client import DEFAULT_MODEL_CONFIG, ModelClient
from backend.storage.pg_search_store import (
    CollectionManifest,
    PgSearchStore,
    SearchDocument,
)
from backend.utils.llama_server import ManagedLlamaServer

# --- Configuration & Paths ---
BACKEND_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = BACKEND_ROOT / "rag-docs" / "huggingface-llm-course"

CHUNKING_VERSION = "recursive-character-v1"
FRONTMATTER_BOUNDARY = "---"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionSummary:
    """Counts and model settings for one completed atomic ingestion."""

    source_document_count: int
    chunk_count: int
    collection_name: str
    embedding_model: str
    embedding_dimension: int


# --- Stage 1: Document Loading ---
def find_markdown_files(docs_dir: Path) -> list[Path]:
    """Return deterministic Markdown paths below an existing corpus directory."""
    return sorted(docs_dir.glob("**/*.md"))


def parse_frontmatter(content: str) -> tuple[dict, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_BOUNDARY:
        return {}, content
    try:
        end_index = lines[1:].index(FRONTMATTER_BOUNDARY) + 1
    except ValueError:
        return {}, content

    metadata: dict = {}
    for line in lines[1:end_index]:
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        if key.strip():
            metadata[key.strip()] = _parse_scalar(raw_value.strip())
    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    return metadata, body


def _parse_scalar(value: str) -> str | int | float | bool:
    if not value:
        return ""
    try:
        parsed = json.loads(value)
        if isinstance(parsed, str | int | float | bool):
            return parsed
    except json.JSONDecodeError:
        pass
    return value


def _read_markdown(path: Path) -> str:
    """Read UTF-8 Markdown, falling back to maintained encoding detection."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as utf8_error:
        raw_content = path.read_bytes()
        for candidate in chardet.detect_all(raw_content):
            encoding = candidate.get("encoding")
            if not isinstance(encoding, str):
                continue
            try:
                return path.read_text(encoding=encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        raise RuntimeError(f"Could not detect encoding for {path}") from utf8_error


def load_documents(docs_dir: Path, skip_errors: bool = False) -> list[Document]:
    """Load Markdown and merge its scalar frontmatter into document metadata."""
    all_docs: list[Document] = []
    for path in find_markdown_files(docs_dir):
        try:
            # Keep the common UTF-8 path simple while accepting legacy encodings.
            content = _read_markdown(path)
            frontmatter, body = parse_frontmatter(content)
            source_file = str(path.relative_to(docs_dir))
            metadata = {"source": str(path)}
            metadata.update(frontmatter)
            metadata.update({"source_file": source_file, "file_type": "md"})
            all_docs.append(Document(page_content=body, metadata=metadata))
        except Exception as error:
            if not skip_errors:
                raise RuntimeError(f"Error loading {path}") from error
    return all_docs


# --- Stage 2: Chunking & ID Assignment ---
def split_documents(documents: list[Document]) -> list[Document]:
    """Split source documents using the versioned application chunking contract."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def assign_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Assign stable content-derived IDs and chunk metadata."""
    chunk_indexes: defaultdict[str, int] = defaultdict(int)
    assigned: list[Document] = []
    for doc_idx, chunk in enumerate(chunks):
        source_file = str(chunk.metadata.get("source_file", "unknown"))
        chunk_idx = chunk_indexes[source_file]
        chunk_indexes[source_file] += 1

        content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        id_material = f"{CHUNKING_VERSION}|{source_file}|{chunk_idx}|{content_hash}"
        digest = hashlib.sha256(id_material.encode()).hexdigest()[:24]
        chunk_id = f"chunk_{digest}"

        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "doc_index": doc_idx,
                "content_hash": content_hash,
                "content_length": len(chunk.page_content),
                "chunking_version": CHUNKING_VERSION,
            }
        )
        assigned.append(Document(page_content=chunk.page_content, metadata=metadata))
    return assigned


# --- Stage 3: Embedding via HTTP ---
def embed_chunks_via_http(
    chunks: list[Document], model_client: ModelClient
) -> list[list[float]]:
    """Embed chunks through the shared BGE model client."""
    return model_client.embed_documents(
        [chunk.page_content for chunk in chunks],
        batch_size=16,
    )


def prepare_chunks(docs_dir: Path) -> tuple[int, list[Document]]:
    """Validate a corpus and return its source count plus identified chunks."""
    if not docs_dir.is_dir():
        raise RuntimeError(f"Markdown corpus directory does not exist: {docs_dir}")
    documents = load_documents(docs_dir)
    if not documents:
        raise RuntimeError(f"Markdown corpus contains no .md documents: {docs_dir}")
    chunks = assign_chunk_ids(split_documents(documents))
    if not chunks:
        raise RuntimeError(f"Markdown corpus produced no chunks: {docs_dir}")
    return len(documents), chunks


def build_search_documents(
    chunks: Sequence[Document],
    vectors: Sequence[Sequence[float]],
) -> list[SearchDocument]:
    """Pair identified chunks with vectors for the PostgreSQL storage boundary."""
    if len(chunks) != len(vectors):
        raise RuntimeError(
            f"Embedding count {len(vectors)} does not match chunk count {len(chunks)}"
        )
    return [
        SearchDocument(
            chunk_id=str(chunk.metadata["chunk_id"]),
            content=chunk.page_content,
            metadata=chunk.metadata,
            embedding=np.asarray(vector, dtype=np.float32),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def index_chunks(
    source_document_count: int,
    chunks: list[Document],
    model_client: ModelClient,
    store: PgSearchStore,
) -> IngestionSummary:
    """Embed prepared chunks and atomically replace the application collection."""
    vectors = embed_chunks_via_http(chunks, model_client)
    documents_to_store = build_search_documents(chunks, vectors)
    manifest = CollectionManifest(
        document_count=len(documents_to_store),
        embedding_model=model_client.config.embedding_model,
        embedding_dimension=model_client.config.embedding_dimension,
    )
    store.replace_documents(documents_to_store, manifest)
    return IngestionSummary(
        source_document_count=source_document_count,
        chunk_count=len(documents_to_store),
        collection_name=store.collection_name,
        embedding_model=model_client.config.embedding_model,
        embedding_dimension=model_client.config.embedding_dimension,
    )


def ingest_documents(
    docs_dir: Path,
    model_client: ModelClient,
    store: PgSearchStore,
) -> IngestionSummary:
    """Run the complete testable ingestion workflow for one Markdown corpus."""
    source_document_count, chunks = prepare_chunks(docs_dir)
    return index_chunks(source_document_count, chunks, model_client, store)


# --- Orchestration ---
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info("Loading documents from %s", DOCS_DIR)
    source_document_count, chunks = prepare_chunks(DOCS_DIR)

    with (
        ManagedLlamaServer(models_dir=BACKEND_ROOT / "models", port=8080),
        ModelClient(DEFAULT_MODEL_CONFIG) as model_client,
        PgSearchStore(
            embedding_dimension=model_client.config.embedding_dimension
        ) as store,
    ):
        summary = index_chunks(
            source_document_count,
            chunks,
            model_client,
            store,
        )
    LOGGER.info(
        "Ingested %d source documents as %d chunks into %s with %s (%d dimensions)",
        summary.source_document_count,
        summary.chunk_count,
        summary.collection_name,
        summary.embedding_model,
        summary.embedding_dimension,
    )


if __name__ == "__main__":
    main()
