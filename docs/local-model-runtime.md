# Local Model Runtime

`backend.local_model_runtime.local_model_runtime()` provides a validated session
with an externally running Model Server. It takes no arguments and yields a
`ModelClient` directly:

```python
from backend.local_model_runtime import local_model_runtime

with local_model_runtime() as models:
    vector = models.embed_query("How does retrieval work?")
```

The runtime owns one reusable HTTP client per context and closes it on normal
exit, caller failure, or startup failure. It never starts, stops, adopts, or
otherwise manages a Model Server. PostgreSQL and generation clients remain
owned by their callers.

This implements [issue #3](https://github.com/ethankimm/rag/issues/3).
Production caller migration and persistent server launcher commands are later
parts of [issue #2](https://github.com/ethankimm/rag/issues/2); existing entry
points do not yet use this session.

## Configuration

Export environment settings before the first runtime context is entered. The
runtime caches one frozen `ModelConfig` on first use; changing the environment
afterward requires a process restart. It does not load `.env` itself.

| Setting | Default |
| --- | --- |
| `LOCAL_MODEL_BASE_URL` | `http://127.0.0.1:8080` |
| `LOCAL_EMBEDDING_MODEL` | `bge-m3-q8_0` |
| `LOCAL_RERANKING_MODEL` | `bge-reranker-v2-m3-Q8_0` |
| `LOCAL_EMBEDDING_DIMENSION` | `1024` |

The URL must be absolute HTTP(S), model IDs must be nonempty, and the dimension
must be a positive integer. Callers use the yielded client's frozen `config`
when they need the validated dimension.

## Startup and errors

Health (`/health`) and catalog (`/v1/models`) validation share a five-second
retry window, with 500 ms waits and request timeouts limited to the remaining
window. Transport failures, non-success HTTP responses, and missing model IDs
are retryable. Invalid JSON or catalog schemas fail immediately. Both configured
IDs must be present; additional models are allowed.

After validation, one embedding request uses the client's existing 120-second
inference timeout to accommodate cold model loading. The response must have the
configured dimension, numeric finite values, and at least one nonzero value.
The shared embedding validator now rejects booleans and numeric strings, and
response indexes must be integers. Zero-vector detection tests actual zero
values, so small finite nonzero vectors remain valid. Startup never runs a
reranking inference request.

Waiting, embedding loading, and ready status lines are flushed to stderr, so
startup progress is visible without application logging configuration.

All startup failures raise `LocalModelStartupError`. Its `missing_model_ids`
attribute is a `frozenset[str]`, empty for failures other than missing models.
Messages preserve details such as the expected and returned embedding dimensions;
`__cause__` preserves the underlying failure, including nested transport and
parsing causes. Errors raised by the caller after startup propagate unchanged.

## Deterministic verification

```sh
uv run pytest tests/test_local_model_runtime.py
uv run ruff check backend tests
uv run ruff format --check backend tests
```

Tests enter the public context manager, with a private client factory supplying
real `ModelClient` instances backed by `httpx.MockTransport`. They replace only
the external HTTP transport and clock; no GGUF files or running server are needed.
Real-model verification and `make test-runtime` belong to the later launcher and
integration work.
