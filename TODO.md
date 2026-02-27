# TODO — Missing Features & Improvements

## High Priority

### Tests
- No unit tests exist (`tests/__init__.py` is empty)
- Testable pure functions: `_classify`, `_enrich_metadata`, `_split_documents`, `_get_scope`, `_generate_symbol_documents`
- Integration tests for the full ingest and query pipeline (mock external APIs)

### Error handling
- `_clone_repo` crashes with raw `CalledProcessError` on bad token/repo/branch — needs user-friendly message
- `ask()` propagates all Claude/Pinecone exceptions, crashing the interactive REPL
- Pinecone index creation is async; code proceeds to upsert before index is ready — needs a readiness check

### Incremental ingestion
- Every `ingest` re-clones and re-embeds the entire repo from scratch
- LlamaIndex auto-generates vector IDs, so repeated ingestion creates duplicates in Pinecone
- Need: delta detection (e.g. git diff), stale vector deletion, or deterministic IDs

## Medium Priority

### Multi-repo namespace isolation
- All repos land in a single flat Pinecone index with no namespace separation
- Queries intermingle results from different repos
- Need: Pinecone namespaces per `owner/repo`, and a `--repo` filter on the `ask` subcommand

### Metadata filtering in queries
- Metadata is enriched during ingestion (`file_type`, `directory`, `is_doc`) but never used at query time
- Add `--filter` or `--path` flags to `ask` so users can scope queries (e.g. "only look in /docs")

### Embedding cache
- Every ingest re-embeds every chunk, which is slow and expensive for large repos
- LlamaIndex supports local `SimpleDocumentStore` as a cache layer

### Logging
- All output is `print()` — no structured logging, no log levels, no way to silence or redirect

### Dependency pinning
- `requirements.txt` has no pinned versions — fragile given LlamaIndex's rapid API churn
- Add `pyproject.toml` or at least pin major versions

## Low Priority

### Rename `--ctags` flag
- CLI help says "via universal-ctags" but implementation uses Python `ast` — no ctags binary needed
- Rename to `--symbols` for clarity

### `_get_scope` is O(n^2)
- Walks the entire AST for every symbol; only handles one level of nesting
- Use a recursive visitor with a scope stack instead

### Symbol extraction for other languages
- Currently Python-only (`ast` module)
- JS/TS could use tree-sitter queries (already a dependency)

### Streaming output
- `engine.query()` waits for the full response before printing
- Claude supports streaming — would improve UX for long answers

### CLI `--top-k` flag
- `similarity_top_k` is hardcoded in config (default 5), not exposed on `ask` subcommand

### CohereRerank model not configurable
- `rerank-english-v3.0` is hardcoded in `query.py` — should be a config field for multilingual or newer models
