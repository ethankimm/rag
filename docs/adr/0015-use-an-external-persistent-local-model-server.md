# 0015. Use an External Persistent Local Model Server

## Status

Accepted

## Context

FastAPI, the RAG smoke command, ingestion, and evaluation previously assembled
and managed `llama-server` independently. Each caller duplicated process
lifecycle, endpoint, model selection, startup ordering, and cleanup knowledge.
The client endpoint could drift from the process endpoint, and application exit
discarded loaded models needed by later commands.

Cold server and embedding-model startup can take long enough to look stalled.
Process reuse also needs a clear ownership rule: a port listener is not proof
that this repository owns or may terminate that process.

## Decision

Run the local Model Server as a persistent process outside application
processes. Make targets own startup and shutdown:

- `make model-server-up` starts or reuses only the process identity recorded by
  this repository.
- `make model-server-down` signals only that verified recorded process.
- `make dev`, `make ingest`, `make eval`, `make rag-smoke`, and
  `make test-runtime` ensure the Model Server is running before their work.

Application contexts call the no-argument
`backend.local_model_runtime.local_model_runtime()` context manager. The
context manager validates health, required model IDs, and embedding behavior;
yields one configured `ModelClient`; and closes only that client. It never
starts, adopts, or stops the server process.

The shared endpoint, embedding model, reranking model, and embedding dimension
come from immutable environment configuration loaded once per application
process. Stop the Model Server before changing this configuration, then restart
application processes. Standard Make commands start the server again with the
new contract. Startup prints server and embedding-model wait states.

An untracked listener on the configured port is a startup error. The launcher
reports available PID and executable details but never adopts or terminates the
listener. Safe shutdown verifies PID, executable, and process start time before
signaling.

## Consequences

- API, smoke, ingestion, and evaluation processes reuse loaded local models
  across runs and no longer duplicate process-management policy.
- Stopping an application closes its HTTP clients but leaves the Model Server
  running until explicit shutdown.
- Local startup needs writable `.runtime/` state and operating-system process
  inspection. Stale or invalid state is discarded without signaling a process.
- Normal backend tests use deterministic HTTP adapters and need no GGUF files or
  Model Server. `make test-runtime` is an opt-in, model-capable gate for real
  embedding and reranking behavior.
- A changed environment does not reconfigure a running Model Server or
  application process. Stop the server before the change, then restart callers
  to adopt the new contract.

## Alternatives Considered

### Let each application process own a Model Server

Rejected. It duplicates lifecycle policy, makes endpoint drift possible, and
unloads models whenever a short-lived caller exits.

### Adopt any healthy process on the configured port

Rejected. Health does not establish ownership. Automatic adoption makes later
shutdown unsafe and can bind callers to an incompatible model contract.

### Require every person to start the server manually

Rejected as the only workflow. Explicit start remains available, but standard
Make commands ensure startup so callers do not need to remember a separate
step.

### Stop processes by executable name or port

Rejected. Broad process-name or port-based termination can kill unrelated local
work and cannot defend against PID reuse.

## Boundaries

PostgreSQL stores remain owned by their current API, ingestion, evaluation, and
smoke callers. OpenAI-compatible generation clients also retain their existing
owners. This decision does not change retrieval, embedding semantics,
persistence schemas, providers, authentication, GCP deployment, or knowledge
management.

The application remains a single local installation. All locally available
documents remain accessible without user or administrator roles.

## Relationship to Other ADRs

ADR 0009 established the local FastAPI and GGUF model direction. This decision
supersedes only ADR 0009's in-process `llama-cpp-python` embedding ownership;
the local model choice and generation-client boundary remain unchanged.
