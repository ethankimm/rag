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
Persistent server launcher commands are provided by
[issue #4](https://github.com/ethankimm/rag/issues/4). FastAPI and the RAG smoke
command use this session through
[issue #5](https://github.com/ethankimm/rag/issues/5). Ingestion and evaluation
migration remains in [issue #6](https://github.com/ethankimm/rag/issues/6).

## Persistent server commands

Start and stop the repository-owned Model Server independently from application
processes:

```sh
make model-server-up
make model-server-down
```

`make dev` and `make rag-smoke` run `model-server-up` before application work.
Direct module or server commands require the Model Server to be running already.

The launcher stores verified process identity in `.runtime/model-server.json`
and writes current-run server output to `.runtime/model-server.log`, replacing
the previous run's log when it starts a new process. The prior log moves to
`.runtime/model-server.previous.log`, so failed-start diagnostics survive one
retry without unbounded growth across restarts. Startup reuses a live, matching
process, clears stale state, and waits up to 30 seconds for health. An untracked
process on the configured port causes a failure that reports its PID and
executable when the operating system exposes them. The launcher never adopts or
terminates that process.

Shutdown sends `SIGTERM` only when the current PID, executable, and process start
time match the recorded identity. It uses `SIGKILL` after a five-second grace
period. Missing, malformed, stale, or PID-reused state is removed without
signaling a process.

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

`local_model_runtime()` accepts HTTP or HTTPS endpoints. The local
`model-server-up` launcher requires an explicit-port HTTP URL because it starts
`llama-server` without TLS configuration. The launcher accepts only `localhost`,
`127.0.0.1`, or `::1` to prevent accidental network exposure.

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

## Real GGUF runtime gate

Run the opt-in gate on a model-capable development machine:

```sh
make test-runtime
```

Prerequisites are `llama-server` on `PATH` and both configured GGUF files in
`backend/models/`. The tracked `llama-models.ini` must expose the configured
embedding and reranking model IDs. Environment overrides use the same settings
listed above.

The command starts or reuses the persistent repository-owned Model Server, then
enters the same `local_model_runtime()` context used by production callers. It
checks health and the model catalog, validates a real finite, nonzero embedding
with the configured dimension, and validates one finite reranking score for each
supplied document with complete, valid document indexes. The server remains
running after the gate; use `make model-server-down` when finished.

Missing executables or files fail during persistent server startup. Catalog,
embedding, and reranking incompatibilities fail in the runtime test with the
configured model contract in the error. Normal `make test-backend` runs exclude
the `model_integration` marker and require no GGUF files or Model Server.
