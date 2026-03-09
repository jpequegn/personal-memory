"""Tests for the sqlite-vec memory store."""

import math
import random
import tempfile
from pathlib import Path

import pytest
from mem.store import MemoryStore, _pack, _EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with MemoryStore(db) as s:
        yield s


def _rand_embedding(seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(_EMBEDDING_DIM)]
    # L2-normalise so distances are comparable
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _unit(dim: int, idx: int) -> list[float]:
    """Standard basis vector e_idx (padded to dim with zeros)."""
    v = [0.0] * dim
    v[idx % dim] = 1.0
    return v


# ---------------------------------------------------------------------------
# Schema / lifecycle
# ---------------------------------------------------------------------------

class TestStoreLifecycle:
    def test_empty_store_count_zero(self, store):
        assert store.count() == 0

    def test_close_and_reopen(self, tmp_path):
        db = tmp_path / "reopen.db"
        with MemoryStore(db) as s:
            s.insert("hello", _rand_embedding(0))
        with MemoryStore(db) as s:
            assert s.count() == 1

    def test_default_db_path_created(self, tmp_path, monkeypatch):
        """MemoryStore creates the parent directory automatically."""
        nested = tmp_path / "a" / "b" / "mem.db"
        with MemoryStore(nested) as s:
            assert nested.exists()


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

class TestInsert:
    def test_insert_returns_positive_id(self, store):
        row_id = store.insert("chunk text", _rand_embedding(1))
        assert row_id > 0

    def test_count_increments(self, store):
        for i in range(5):
            store.insert(f"chunk {i}", _rand_embedding(i))
        assert store.count() == 5

    def test_insert_with_metadata(self, store):
        store.insert(
            "with meta",
            _rand_embedding(2),
            source="wiki",
            tags=["ai", "memory"],
            fingerprint="abc123",
        )
        results = store.query(_rand_embedding(2), top_k=1)
        assert results[0].source == "wiki"
        assert results[0].tags == ["ai", "memory"]
        assert results[0].fingerprint == "abc123"

    def test_insert_20_chunks_acceptance(self, store):
        """Acceptance: insert 20 chunks, query returns ranked results."""
        query_vec = _unit(_EMBEDDING_DIM, 0)
        # Insert 20 chunks; chunk at index 0 is identical to the query
        for i in range(20):
            store.insert(f"chunk {i}", _unit(_EMBEDDING_DIM, i), source=f"src-{i}")
        assert store.count() == 20
        results = store.query(query_vec, top_k=5)
        assert len(results) == 5
        # The nearest chunk should be chunk 0 (distance ≈ 0)
        assert results[0].chunk_text == "chunk 0"
        assert results[0].distance == pytest.approx(0.0, abs=1e-5)
        # Results must be ordered by ascending distance
        distances = [r.distance for r in results]
        assert distances == sorted(distances)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_empty_store_returns_empty(self, store):
        assert store.query(_rand_embedding(0)) == []

    def test_query_top_k_respected(self, store):
        for i in range(10):
            store.insert(f"c{i}", _rand_embedding(i))
        results = store.query(_rand_embedding(0), top_k=3)
        assert len(results) == 3

    def test_nearest_is_self(self, store):
        vec = _rand_embedding(42)
        store.insert("needle", vec)
        for i in range(5):
            store.insert(f"hay {i}", _rand_embedding(i))
        results = store.query(vec, top_k=1)
        assert results[0].chunk_text == "needle"
        assert results[0].distance == pytest.approx(0.0, abs=1e-5)

    def test_result_fields_populated(self, store):
        vec = _rand_embedding(7)
        store.insert("text", vec, source="src", tags=["t"], fingerprint="fp")
        r = store.query(vec, top_k=1)[0]
        assert r.chunk_text == "text"
        assert r.source == "src"
        assert r.tags == ["t"]
        assert r.fingerprint == "fp"
        assert r.created_at is not None
        assert isinstance(r.distance, float)

    def test_results_ordered_by_distance(self, store):
        query_vec = _unit(_EMBEDDING_DIM, 0)
        for i in range(8):
            store.insert(f"c{i}", _unit(_EMBEDDING_DIM, i))
        results = store.query(query_vec, top_k=8)
        distances = [r.distance for r in results]
        assert distances == sorted(distances)


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_list_sources_empty(self, store):
        assert store.list_sources() == []

    def test_list_sources_distinct_sorted(self, store):
        for src in ["zebra", "alpha", "alpha", "beta", None]:
            store.insert("x", _rand_embedding(), source=src)
        assert store.list_sources() == ["alpha", "beta", "zebra"]

    def test_count_after_inserts(self, store):
        for _ in range(7):
            store.insert("x", _rand_embedding())
        assert store.count() == 7
