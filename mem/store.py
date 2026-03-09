"""sqlite-vec backed memory store.

Schema
------
memories      – metadata table (text, source, tags, fingerprint, created_at)
memory_vss    – vec0 virtual table for KNN embedding search
"""

import json
import sqlite3
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

    def insert(
        self,
        chunk_text: str,
        embedding: list[float],
        *,
        source: str | None = None,
        tags: list[str] | None = None,
        fingerprint: str | None = None,
    ) -> int:
        """Insert a chunk and its embedding.  Returns the new row id."""
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()
