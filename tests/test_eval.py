"""Tests for the eval harness (mocked embeddings)."""

from __future__ import annotations

import json
import math
import random
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mem.store import MemoryStore

HERE = Path(__file__).parent.parent / "eval"
_DIM = 1536


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _topic_vec(topic: str, noise: float = 0.02) -> list[float]:
    """Deterministic per-topic vector with optional small noise."""
    topics = ["transformers", "vector_databases", "python_packaging",
              "retrieval_augmented_generation", "sqlite"]
    idx = topics.index(topic) if topic in topics else 0
    rng = random.Random(idx * 1000 + hash(topic) % 1000)
    # Basis vector for the topic + small noise
    v = [1.0 if i == idx * (_DIM // len(topics)) else 0.0 for i in range(_DIM)]
    v = [x + rng.gauss(0, noise) for x in v]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _make_embed_batch(items: list[dict], queries: list[dict] | None = None):
    """Return embed_batch side_effect that uses topic-aware vectors."""
    text_to_topic: dict[str, str] = {item["text"]: item["topic"] for item in items}
    if queries:
        for q in queries:
            text_to_topic[q["query"]] = q["topic"]

    def _embed(texts):
        return [_topic_vec(text_to_topic.get(t, "transformers")) for t in texts]

    return _embed


# ---------------------------------------------------------------------------
# Data file tests
# ---------------------------------------------------------------------------

class TestDataFiles:
    def test_items_file_exists(self):
        assert (HERE / "items.json").exists()

    def test_queries_file_exists(self):
        assert (HERE / "queries.json").exists()

    def test_items_count(self):
        items = json.loads((HERE / "items.json").read_text())
        assert len(items) == 50

    def test_items_five_topics(self):
        items = json.loads((HERE / "items.json").read_text())
        topics = {i["topic"] for i in items}
        assert len(topics) == 5

    def test_items_ten_per_topic(self):
        items = json.loads((HERE / "items.json").read_text())
        from collections import Counter
        counts = Counter(i["topic"] for i in items)
        assert all(c == 10 for c in counts.values())

    def test_queries_count(self):
        queries = json.loads((HERE / "queries.json").read_text())
        assert len(queries) == 10

    def test_queries_two_per_topic(self):
        queries = json.loads((HERE / "queries.json").read_text())
        from collections import Counter
        counts = Counter(q["topic"] for q in queries)
        assert all(c == 2 for c in counts.values())

    def test_items_have_required_fields(self):
        items = json.loads((HERE / "items.json").read_text())
        for item in items:
            assert "id" in item
            assert "topic" in item
            assert "text" in item

    def test_queries_have_required_fields(self):
        queries = json.loads((HERE / "queries.json").read_text())
        for q in queries:
            assert "id" in q
            assert "topic" in q
            assert "query" in q


# ---------------------------------------------------------------------------
# Eval pipeline tests (mocked embeddings)
# ---------------------------------------------------------------------------

class TestEvalPipeline:
    @pytest.fixture
    def eval_db(self, tmp_path):
        return tmp_path / "eval.db"

    @pytest.fixture
    def items(self):
        return json.loads((HERE / "items.json").read_text())

    @pytest.fixture
    def queries(self):
        return json.loads((HERE / "queries.json").read_text())

    def test_build_store_populates_db(self, eval_db, items, queries):
        from eval.run import build_store
        with patch("mem.embedder.embed_batch", side_effect=_make_embed_batch(items, queries)):
            build_store(eval_db, items)
        with MemoryStore(eval_db) as s:
            assert s.count() == 50

    def test_build_store_skips_if_already_populated(self, eval_db, items, queries):
        from eval.run import build_store
        embed_fn = _make_embed_batch(items, queries)
        with patch("mem.embedder.embed_batch", side_effect=embed_fn) as m:
            build_store(eval_db, items)
            call_count_first = m.call_count
            build_store(eval_db, items)
            assert m.call_count == call_count_first

    def test_run_queries_returns_all_queries(self, eval_db, items, queries):
        from eval.run import build_store, run_queries
        embed_fn = _make_embed_batch(items, queries)
        with patch("mem.embedder.embed_batch", side_effect=embed_fn):
            build_store(eval_db, items)
            results = run_queries(eval_db, queries, items)
        assert len(results) == len(queries)

    def test_precision_fields_present(self, eval_db, items, queries):
        from eval.run import build_store, run_queries
        embed_fn = _make_embed_batch(items, queries)
        with patch("mem.embedder.embed_batch", side_effect=embed_fn):
            build_store(eval_db, items)
            results = run_queries(eval_db, queries, items)
        for r in results.values():
            assert "precision_at_k" in r
            assert 0.0 <= r["precision_at_k"] <= 1.0
            assert "correct" in r
            assert "topic" in r

    def test_topic_aware_embeddings_achieve_high_precision(self, eval_db, items, queries):
        """With topic-clustered embeddings, precision@5 should be >= 0.7."""
        from eval.run import build_store, run_queries
        embed_fn = _make_embed_batch(items, queries)
        with patch("mem.embedder.embed_batch", side_effect=embed_fn):
            build_store(eval_db, items)
            results = run_queries(eval_db, queries, items)

        scores = [r["precision_at_k"] for r in results.values()]
        overall = sum(scores) / len(scores)
        assert overall >= 0.7, f"Expected overall precision >= 0.7, got {overall:.2f}"

    def test_print_report_returns_float(self, eval_db, items, queries, capsys):
        from eval.run import build_store, print_report, run_queries
        embed_fn = _make_embed_batch(items, queries)
        with patch("mem.embedder.embed_batch", side_effect=embed_fn):
            build_store(eval_db, items)
            results = run_queries(eval_db, queries, items)
        score = print_report(results)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
