# Architecture — personal-memory

System documentation for contributors and maintainers.

---

## Overview

`personal-memory` is a local semantic memory system. It ingests arbitrary text, splits it into token-bounded chunks, embeds each chunk via the OpenAI API, and stores the embeddings in a SQLite database extended with sqlite-vec for approximate nearest-neighbour search. Retrieval blends vector similarity with a temporal decay score.

**Stack:**
- Python ≥ 3.12
- SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) (vec0 virtual table)
- OpenAI `text-embedding-3-small` (1536 dimensions)
- Click (CLI)
- pytest (tests)

---

## Repository Layout

```
personal-memory/
├── mem/
│   ├── __init__.py
│   ├── chunker.py       # Text → overlapping sentence-aligned chunks
│   ├── embedder.py      # Chunks → OpenAI embedding vectors
│   ├── store.py         # SQLite+sqlite-vec read/write layer
│   ├── retriever.py     # Semantic + recency re-ranking
│   └── cli.py           # Click CLI (mem store/query/list/stats/dedup)
├── tests/
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_store.py
│   ├── test_retriever.py
│   ├── test_cli.py
│   ├── test_dedup.py
│   └── test_eval.py
├── eval/
│   ├── run.py           # Precision@5 harness across 5 topics
│   ├── items.json       # 50 synthetic knowledge items (10/topic)
│   └── queries.json     # 10 queries (2/topic) with expected topics
├── pyproject.toml
└── README.md
```

---

## Data Flow

```
Input text (CLI / file / stdin)
        │
        ▼
  [chunker.chunk()]
  Split into sentence-aligned, token-bounded,
  overlapping chunks: list[str]
        │
        ▼
  [SHA256 fingerprint per chunk]
  Filter: drop any chunk whose fingerprint
  already exists in the DB
        │
        ▼
  [embedder.embed_batch()]
  Call OpenAI embeddings API in batches ≤ 100
  Returns: list[list[float]] (1536-dim)
        │
        ▼
  [store.insert()]
  Write to SQLite:
    memories   → metadata (text, source, tags, fingerprint, created_at)
    memory_vss → vec0 virtual table (embedding)
        │
        ▼
  Stored ✓

Query path:
        │
  [embedder.embed(query_text)]
        │
        ▼
  [store.query(embedding, top_k=2k)]
  KNN via vec0 MATCH operator (L2 distance)
        │
        ▼
  [retriever.retrieve()]
  Score each candidate:
    semantic  = max(0, 1 - distance)
    recency   = 1 / (1 + days_since_created)
    final     = 0.8·semantic + 0.2·recency
  Re-rank by final score descending
  Return top k
        │
        ▼
  Results: list[RetrievedResult]
```

---

## Module Reference

### `mem/chunker.py`

Splits raw text into chunks suitable for embedding.

**Public API:**
```python
chunk(text: str, max_tokens: int = 256, overlap: int = 32) -> list[str]
```

**Algorithm:**
1. Split text into sentences on `.`, `!`, `?` followed by whitespace.
2. Greedily accumulate sentences into a buffer until adding the next sentence would exceed `max_tokens`.
3. When the buffer is full, emit it as a chunk.
4. Seed the next buffer with the trailing `overlap` tokens from the previous chunk (at sentence boundary where possible, else word boundary).
5. If a single sentence exceeds `max_tokens`, fall back to word-level splitting with the same overlap logic.
6. Emit the final buffer as the last chunk.

**Token counting:**
- Uses `tiktoken` (CL100k_base) if installed.
- Falls back to `len(text) // 4` otherwise.
- Always returns ≥ 1.

**Key internal helpers:**
| Function | Purpose |
|----------|---------|
| `_count_tokens(text)` | Token count with tiktoken fallback |
| `_split_sentences(text)` | Regex sentence splitter |
| `_trim_to_tokens(words, max_tok)` | Trailing words within token budget |
| `_trailing_sentences(sentences, max_tok)` | Trailing sentences within token budget |

**Design rationale:** Sentence alignment prevents splits mid-clause, which degrades retrieval quality. Overlap ensures chunk boundaries don't lose context.

---

### `mem/embedder.py`

Converts text strings to embedding vectors via OpenAI.

**Public API:**
```python
embed(text: str) -> list[float]                    # single text
embed_batch(texts: list[str]) -> list[list[float]] # batch
```

**Constants:**
```python
_MODEL      = "text-embedding-3-small"
_DIM        = 1536
_BATCH_SIZE = 100
```

**Behaviour:**
- `embed(text)` delegates to `embed_batch([text])[0]`.
- `embed_batch` splits input into batches of ≤ 100, calls the API, and reassembles results in input order.
- Returns `[]` for empty input without making an API call.
- Raises `EnvironmentError` if `OPENAI_API_KEY` is not set (checked lazily on first call).

**OpenAI call:**
```python
client.embeddings.create(model=_MODEL, input=batch)
```

---

### `mem/store.py`

Persistent read/write layer over SQLite + sqlite-vec.

**Default path:** `~/.mem/memory.db`

**Schema:**

```sql
CREATE TABLE memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_text  TEXT    NOT NULL,
    source      TEXT,
    tags        TEXT    DEFAULT '[]',     -- JSON array
    created_at  TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    fingerprint TEXT                      -- SHA256 hex, 64 chars
);

CREATE VIRTUAL TABLE memory_vss USING vec0 (
    embedding FLOAT[1536]                 -- little-endian IEEE-754
);
```

**Key class:** `MemoryStore`

| Method | Description |
|--------|-------------|
| `__init__(db_path)` | Creates dirs, loads sqlite_vec extension, initialises schema |
| `add(text, *, source, tags, max_tokens, overlap)` | Full pipeline: chunk → fingerprint → dedup → embed → insert. Returns `(stored, skipped)`. |
| `insert(chunk_text, embedding, *, source, tags, fingerprint, skip_duplicates)` | Low-level single insert. Returns row id or `None` if duplicate skipped. |
| `fingerprint_exists(fingerprint)` | Check for existing SHA256. |
| `find_duplicates()` | Returns `{fingerprint: [id, ...]}` for all duplicated fingerprints. |
| `query(embedding, top_k)` | KNN via `vec0 MATCH`. Returns `list[MemoryResult]` ordered by ascending distance. |
| `count()` | Total chunk count. |
| `list_sources()` | Distinct non-null sources, sorted. |
| `close()` | Close connection. Supports context manager (`with MemoryStore() as s:`). |

**Return types:**

```python
@dataclass
class MemoryResult:
    id: int
    chunk_text: str
    source: str | None
    tags: list[str]
    created_at: datetime
    fingerprint: str
    distance: float
```

**Deduplication detail:**
- `add()` computes `sha256(chunk_text.encode()).hexdigest()` for every chunk.
- Checks `fingerprint_exists()` before calling `embed_batch`.
- Only new chunks are embedded — duplicate chunks never reach the API.

**sqlite-vec KNN query:**
```sql
SELECT m.id, m.chunk_text, m.source, m.tags, m.created_at, m.fingerprint,
       v.distance
FROM memory_vss v
JOIN memories m ON m.id = v.rowid
WHERE v.embedding MATCH ?
  AND k = ?
ORDER BY v.distance
```

---

### `mem/retriever.py`

Re-ranks KNN results using a semantic + recency blended score.

**Key class:** `Retriever`

```python
Retriever(store: MemoryStore, semantic_weight: float = 0.8, recency_weight: float = 0.2)
```

Raises `ValueError` if weights don't sum to 1.0 (within 1e-9 tolerance).

**Method:**
```python
retrieve(embedding: list[float], top_k: int = 5, *, now: datetime | None = None) -> list[RetrievedResult]
```

1. Fetches `2 × top_k` candidates from `store.query()`.
2. Scores each:
   ```python
   semantic = max(0.0, 1.0 - distance)
   recency  = 1.0 / (1.0 + days_since_created)
   final    = semantic_weight * semantic + recency_weight * recency
   ```
3. Sorts by `final` descending, returns first `top_k`.

**Oversampling rationale:** Fetching `2k` candidates before re-ranking gives recency scoring room to surface recent items that ranked just outside `k` on semantic distance alone.

**Return type:**
```python
@dataclass
class RetrievedResult:
    memory: MemoryResult
    semantic_score: float
    recency_score: float
    final_score: float
```

**Recency decay behaviour:**
| Age | Recency score |
|-----|--------------|
| 0 seconds | 1.00 |
| 1 day | 0.50 |
| 6 days | 0.14 |
| 30 days | 0.032 |
| 1 year | 0.003 |

The `now` parameter allows injecting a fixed time in tests.

---

### `mem/cli.py`

Click-based CLI. Entry point: `mem.cli:main` (registered as `mem` in pyproject.toml).

**Commands:**

| Command | Key options | Behaviour |
|---------|-------------|-----------|
| `store` | `--file`, `--stdin`, `--source`, `--tag` | Calls `MemoryStore.add()` |
| `query` | `-k`, `--semantic-weight`, `--recency-weight`, `--json` | Embeds query, calls `Retriever.retrieve()`, formats output |
| `list` | `-n`, `--json` | Recent entries from `memories` table |
| `stats` | `--json` | Count, sources, oldest/newest timestamps |
| `dedup` | `--dry-run` | Calls `store.find_duplicates()`, reports |

**Helper functions:**
```python
_store() -> MemoryStore         # creates MemoryStore(~/.mem/memory.db)
_score_color(score) -> str      # "green" ≥0.7, "yellow" ≥0.4, "red" otherwise
```

**Coloured query output** uses `click.style()`. JSON output uses `json.dumps()`.

---

### `eval/run.py`

Offline retrieval evaluation harness. Not part of the user-facing CLI.

**Measures:** Precision@5 per query across 5 topics.

**Data files (eval/):**
- `items.json`: 50 items, 10 per topic (transformers, vector_databases, python_packaging, retrieval_augmented_generation, sqlite). Each has `id`, `text`, `topic`.
- `queries.json`: 10 queries, 2 per topic. Each has `id`, `text`, `topic`.

**Pipeline:**
```python
build_store(db_path, items)
# → embeds all items, stores with source=topic

run_queries(db_path, queries, items)
# → for each query: embed → retrieve top5 (semantic_weight=0.9)
#                → count results whose source matches query.topic
#                → precision@5 = correct / 5

print_report(results)
# → per-topic averages, overall average, bar chart, PASS/FAIL vs 0.7
```

**CLI:**
```bash
uv run python eval/run.py [--db PATH] [--no-cache]
```

**Pass threshold:** overall precision@5 ≥ 0.70.

---

## Testing

Tests live in `tests/`. All OpenAI API calls are mocked — no API key needed.

```bash
uv run pytest                  # all tests
uv run pytest tests/test_chunker.py -v
```

**Mocking strategy:**
- `embedder` tests patch `openai.OpenAI` and return deterministic fake embeddings.
- `store` tests use `tmp_path` fixtures for isolated SQLite databases.
- `retriever` tests inject synthetic `MemoryResult` objects.
- `cli` tests use `click.testing.CliRunner` and mock `MemoryStore`.

**Coverage by module:**

| Module | Tests | Focus areas |
|--------|-------|-------------|
| chunker | 40+ | Size constraints, overlap, sentence boundaries, edge cases |
| embedder | 10+ | Batching, API key validation, order preservation |
| store | 30+ | Schema, insert, dedup, query, lifecycle |
| retriever | 20+ | Recency decay, weight validation, ranking |
| cli | 20+ | All commands, JSON output, flag combinations |
| dedup | 10+ | Fingerprint uniqueness, duplicate detection |
| eval | 15+ | Data files, pipeline, precision metrics |

---

## Key Design Decisions

**Why sentence-aligned chunking?**
Splitting at arbitrary token offsets cuts sentences mid-clause, producing fragments that embed poorly and retrieve inconsistently. Sentence-aligned splits preserve grammatical and semantic units.

**Why deduplicate before embedding?**
Embedding API calls have per-token cost and latency. Checking SHA256 fingerprints before the API call avoids re-embedding content already in the store, which is common when ingesting overlapping documents or re-running ingestion.

**Why blend semantic + recency?**
Pure semantic search can bury recently added memories behind older, more topically saturated content. Recency scoring (with a low default weight of 0.2) gently surfaces newer material without overwhelming relevance.

**Why oversampling (2×top_k)?**
Re-ranking by recency can pull items from positions just below top_k in semantic ranking. Fetching 2×top_k candidates before re-ranking ensures the final list is genuinely the best blend of relevance and freshness.

**Why SQLite + sqlite-vec instead of a dedicated vector DB?**
- Zero external infrastructure — a single file.
- Trivially portable and backupable.
- Full SQL access for auditing and debugging.
- sqlite-vec's vec0 virtual table provides efficient approximate KNN at personal-scale data volumes (thousands to low millions of chunks).

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI embedding API authentication |

---

## Extending the System

**Adding a new embedding model:**
Edit `embedder.py`. Change `_MODEL` and `_DIM`. If using a non-OpenAI provider, replace the client creation in `_client()` and the `embeddings.create()` call. Update the vec0 table dimension in `store.py`'s `CREATE VIRTUAL TABLE` statement.

**Adding tag filtering to queries:**
Add a `--tags` option to `mem query`. In `store.py`, extend the KNN query to JOIN against `memories` and filter by `json_each(tags)` before or after re-ranking.

**Implementing dedup deletion:**
In `store.py`, add a `delete(id: int)` method that removes from both `memories` and `memory_vss`. In `cli.py`, wire `mem dedup` (without `--dry-run`) to call it on the duplicate ids returned by `find_duplicates()`.

**Adding a REST API:**
Wrap `MemoryStore` and `Retriever` in a FastAPI app. The store is already safe to use as a context manager. The main concern is thread safety for concurrent writes — consider `check_same_thread=False` or a connection pool.
