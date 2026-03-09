"""sqlite-vec backed memory store.

Schema
------
memories      – metadata table (text, source, tags, fingerprint, created_at)
memory_vss    – vec0 virtual table for KNN embedding search
"""

import hashlib
import json
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sqlite_vec

_DEFAULT_DB = Path.home() / ".mem" / "memory.db"
_EMBEDDING_DIM = 1536  # text-embedding-3-small / ada-002


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MemoryResult:
    id: int
    chunk_text: str
    source: str | None
    tags: list[str]
    created_at: datetime
    fingerprint: str
    distance: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack(embedding: list[float]) -> bytes:
    """Pack a float list into little-endian IEEE-754 bytes for sqlite-vec."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(f"""
        CREATE TABLE IF NOT EXISTS memories (
            id          INTEGER PRIMARY KEY,
            chunk_text  TEXT    NOT NULL,
            source      TEXT,
            tags        TEXT    DEFAULT '[]',
            created_at  TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            fingerprint TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vss
            USING vec0(embedding FLOAT[{_EMBEDDING_DIM}]);
    """)
    con.commit()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MemoryStore:
    """Persistent vector store backed by sqlite-vec."""

    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        self._con = _open_db(db_path)
        _ensure_schema(self._con)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def fingerprint_exists(self, fingerprint: str) -> bool:
        """Return True if a chunk with this sha256 fingerprint is already stored."""
        row = self._con.execute(
            "SELECT 1 FROM memories WHERE fingerprint = ? LIMIT 1", (fingerprint,)
        ).fetchone()
        return row is not None

    def insert(
        self,
        chunk_text: str,
        embedding: list[float],
        *,
        source: str | None = None,
        tags: list[str] | None = None,
        fingerprint: str | None = None,
        skip_duplicates: bool = False,
    ) -> int | None:
        """Insert a chunk and its embedding.

        Returns the new row id, or None if skip_duplicates=True and the
        fingerprint already exists in the DB.
        """
        if skip_duplicates and fingerprint and self.fingerprint_exists(fingerprint):
            return None

        tags_json = json.dumps(tags or [])
        cur = self._con.execute(
            """
            INSERT INTO memories (chunk_text, source, tags, fingerprint)
            VALUES (?, ?, ?, ?)
            """,
            (chunk_text, source, tags_json, fingerprint),
        )
        row_id = cur.lastrowid
        self._con.execute(
            "INSERT INTO memory_vss (rowid, embedding) VALUES (?, ?)",
            (row_id, _pack(embedding)),
        )
        self._con.commit()
        return row_id

    def find_duplicates(self) -> dict[str, list[int]]:
        """Return fingerprints that appear more than once, mapped to their row ids.

        Used by `mem dedup --dry-run`.
        """
        rows = self._con.execute(
            """
            SELECT fingerprint, GROUP_CONCAT(id) AS ids, COUNT(*) AS cnt
            FROM memories
            WHERE fingerprint IS NOT NULL
            GROUP BY fingerprint
            HAVING cnt > 1
            """
        ).fetchall()
        return {
            r["fingerprint"]: [int(i) for i in r["ids"].split(",")]
            for r in rows
        }

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[MemoryResult]:
        """Return the top-k most similar memories, ordered by ascending distance."""
        rows = self._con.execute(
            """
            SELECT
                m.id,
                m.chunk_text,
                m.source,
                m.tags,
                m.created_at,
                m.fingerprint,
                v.distance
            FROM memory_vss v
            JOIN memories m ON m.id = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            (_pack(embedding), top_k),
        ).fetchall()

        results: list[MemoryResult] = []
        for r in rows:
            results.append(
                MemoryResult(
                    id=r["id"],
                    chunk_text=r["chunk_text"],
                    source=r["source"],
                    tags=json.loads(r["tags"] or "[]"),
                    created_at=datetime.fromisoformat(r["created_at"]),
                    fingerprint=r["fingerprint"] or "",
                    distance=r["distance"],
                )
            )
        return results

    def count(self) -> int:
        """Total number of stored chunks."""
        row = self._con.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0]

    def list_sources(self) -> list[str]:
        """Distinct non-null sources, alphabetically sorted."""
        rows = self._con.execute(
            "SELECT DISTINCT source FROM memories WHERE source IS NOT NULL ORDER BY source"
        ).fetchall()
        return [r[0] for r in rows]

    def add(
        self,
        text: str,
        *,
        source: str | None = None,
        tags: list[str] | None = None,
        max_tokens: int = 256,
        overlap: int = 32,
    ) -> tuple[int, int]:
        """Chunk text, embed non-duplicate chunks, and store them.

        Returns (stored, skipped) where skipped counts duplicate chunks
        whose fingerprint was already in the DB.
        """
        from mem.chunker import chunk
        from mem.embedder import embed_batch

        chunks = chunk(text, max_tokens=max_tokens, overlap=overlap)
        if not chunks:
            return 0, 0

        # Compute fingerprints up-front so we can skip duplicates before
        # paying for embedding API calls.
        fingerprints = [hashlib.sha256(c.encode()).hexdigest() for c in chunks]
        new_chunks = [(c, fp) for c, fp in zip(chunks, fingerprints)
                      if not self.fingerprint_exists(fp)]
        skipped = len(chunks) - len(new_chunks)

        if not new_chunks:
            return 0, skipped

        embeddings = embed_batch([c for c, _ in new_chunks])

        for (chunk_text, fp), embedding in zip(new_chunks, embeddings):
            self.insert(chunk_text, embedding, source=source, tags=tags, fingerprint=fp)

        return len(new_chunks), skipped

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()
