# personal-memory

A personal semantic memory system backed by SQLite. Store text as vector embeddings and retrieve it later using natural language queries — with results ranked by both semantic similarity and recency.

## What it does

`mem` lets you build a local, searchable knowledge base from any text: notes, articles, transcripts, code snippets. You query it in plain language and get back the most relevant chunks, ranked by how closely they match your query and how recently they were stored.

```
mem store "Transformers use self-attention to model token relationships across sequences" -s ml-notes -t transformers
mem query "how does attention work in LLMs?"
# → 1. [0.82] ml-notes  Transformers use self-attention to…
```

---

## Requirements

- **Python ≥ 3.12**
- **OpenAI API key** — used for embeddings (`text-embedding-3-small`)
- **uv** (recommended) or pip

Optional but recommended:
- `tiktoken` — for accurate token counting (falls back to `len/4` heuristic without it)

---

## Installation

```bash
git clone https://github.com/yourusername/personal-memory
cd personal-memory

# With uv (recommended)
uv sync
uv run mem --help

# With pip
pip install -e .
mem --help
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

The database is created automatically at `~/.mem/memory.db` on first use.

---

## CLI Reference

### `mem store`

Store text in memory. Chunks long text automatically. Deduplicates by content hash before calling the embedding API.

```
mem store [TEXT] [OPTIONS]

Arguments:
  TEXT          Inline text to store (omit if using --file or --stdin)

Options:
  -f, --file PATH       Read text from a file
  --stdin               Read text from stdin
  -s, --source LABEL    Label for this memory's origin (e.g. "article", "journal")
  -t, --tag TAG         Tag to attach; can be repeated
```

**Examples:**

```bash
# Inline text
mem store "The CAP theorem states you can only guarantee two of: consistency, availability, partition tolerance" \
  -s distributed-systems -t cap -t databases

# From a file
mem store -f ~/notes/paper.txt -s research -t rag

# From stdin
cat transcript.txt | mem store --stdin -s podcast -t ai

# Multiple tags
mem store "Rust ownership prevents data races at compile time" -t rust -t safety -t systems
```

**Output:**
```
Stored 3 chunk(s). Skipped 0 duplicate(s).
```

---

### `mem query`

Search memory using a natural language query. Results are ranked by a blend of semantic similarity and recency.

```
mem query TEXT [OPTIONS]

Arguments:
  TEXT          Query text

Options:
  -k, --top-k INT             Number of results to return (default: 5)
  --semantic-weight FLOAT     Weight for semantic similarity, 0–1 (default: 0.8)
  --recency-weight FLOAT      Weight for recency score, 0–1 (default: 0.2)
  --json                      Output results as JSON
```

Weights must sum to 1.0. Higher `--semantic-weight` prioritises relevance; higher `--recency-weight` surfaces newer memories.

**Examples:**

```bash
# Basic query
mem query "how does transformer attention work?"

# Get more results
mem query "database indexing strategies" -k 10

# Prioritise recent memories
mem query "what did I read about rust?" --semantic-weight 0.6 --recency-weight 0.4

# JSON output for scripting
mem query "vector databases" --json | jq '.[0].chunk_text'
```

**Default output:**
```
1. [0.824] ml-notes                          [tags: transformers, ml]
   Transformers use self-attention to model token relationships across…

2. [0.712] research                          [tags: rag]
   Retrieval-augmented generation combines a vector store with a language…
```

Score colour coding: green ≥ 0.7 · yellow ≥ 0.4 · red < 0.4

**JSON output fields:**
```json
[
  {
    "rank": 1,
    "final_score": 0.824,
    "semantic_score": 0.91,
    "recency_score": 0.98,
    "source": "ml-notes",
    "tags": ["transformers", "ml"],
    "chunk_text": "Transformers use self-attention…",
    "created_at": "2024-03-09T14:22:10Z"
  }
]
```

---

### `mem list`

Show recent memory entries.

```
mem list [OPTIONS]

Options:
  -n INT      Number of entries to show (default: 20)
  --json      Output as JSON
```

**Example:**
```bash
mem list -n 10
mem list --json | jq '.[].source' | sort | uniq -c
```

---

### `mem stats`

Show a summary of your memory store.

```
mem stats [OPTIONS]

Options:
  --json      Output as JSON
```

**Example output:**
```
Total chunks : 142
Sources      : distributed-systems, ml-notes, podcast, research
Oldest       : 2024-01-15T10:30:45Z
Newest       : 2024-03-09T14:22:10Z
```

---

### `mem dedup`

Find chunks with identical content stored more than once.

```
mem dedup [OPTIONS]

Options:
  --dry-run   Report duplicates without deleting (default behaviour)
```

**Example output:**
```
Found 2 duplicated fingerprint(s) — 2 redundant row(s).
  abc123def456...  ids=[5, 12]
  9f8e7d6c5b4a...  ids=[33, 47]
```

---

## Typical Workflows

### Daily notes

```bash
# Morning: store what you read
mem store -f ~/reading/article.txt -s reading -t $(date +%Y-%m-%d)

# Later: find it
mem query "what were the key points about distributed consensus?"
```

### Research ingestion

```bash
# Ingest a batch of papers
for f in ~/papers/*.txt; do
  mem store -f "$f" -s papers -t research -t $(basename "$f" .txt)
done

# Query across all of them
mem query "Byzantine fault tolerance approaches" -k 10
```

### Piping from other tools

```bash
# Store a web page (after stripping HTML)
curl -s https://example.com/article | html2text | mem store --stdin -s web

# Store clipboard contents (macOS)
pbpaste | mem store --stdin -s clipboard -t snippet
```

---

## How Chunking Works

Long text is split into overlapping chunks before embedding. Each chunk is at most 256 tokens (roughly 200 words), with ~32 tokens of overlap with the next chunk. Splits happen at sentence boundaries where possible, so chunks rarely cut mid-sentence.

You don't need to pre-split your text — just pass the full document.

---

## How Retrieval Works

When you run `mem query`:

1. Your query is embedded into a 1536-dimensional vector.
2. The 10 nearest chunks (by L2 distance) are retrieved from the vector store.
3. Each candidate is scored:
   - **Semantic score** = `1 - L2_distance` (how similar in meaning)
   - **Recency score** = `1 / (1 + days_since_stored)` (how recent)
   - **Final score** = `0.8 × semantic + 0.2 × recency`
4. Candidates are re-ranked by final score, and the top 5 are returned.

You can adjust the balance between relevance and recency with `--semantic-weight` / `--recency-weight`.

---

## Data Storage

- **Database:** `~/.mem/memory.db` (SQLite with sqlite-vec extension)
- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dimensions)
- **Deduplication:** SHA256 fingerprint of each chunk is stored; duplicate content is never re-embedded

The database is a standard SQLite file. You can inspect it with any SQLite browser, back it up with `cp`, or move it elsewhere and set a custom path in the source if needed.

---

## Running Tests

```bash
uv run pytest
```

Tests mock the OpenAI API — no API key required to run the test suite.

---

## Running the Eval Harness

The eval harness measures retrieval precision@5 across 5 knowledge topics (transformers, vector databases, Python packaging, RAG, SQLite). Target is ≥ 0.70 overall.

```bash
uv run python eval/run.py
```

This requires a live OpenAI API key and will make embedding API calls.

---

## Limitations

- No tag-based filtering in `mem query` (semantic search only)
- `mem dedup` reports duplicates but does not delete them yet
- Embedding model is fixed to OpenAI `text-embedding-3-small`
- No REST API — CLI only

---

## License

MIT
