"""Tests for the Retriever (semantic + recency blending)."""

import math
import random
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from mem.retriever import Retriever, RetrievedResult, _recency_score
from mem.store import MemoryResult, MemoryStore

_DIM = 1536


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_vec(seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(_DIM)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _unit(idx: int) -> list[float]:
    v = [0.0] * _DIM
    v[idx % _DIM] = 1.0
    return v


def _mem(
    id: int = 1,
    chunk_text: str = "text",
    distance: float = 0.0,
    created_at: datetime | None = None,
    source: str | None = None,
) -> MemoryResult:
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    return MemoryResult(
        id=id,
        chunk_text=chunk_text,
        source=source,
        tags=[],
        created_at=created_at,
        fingerprint="",
        distance=distance,
    )


def _store_returning(results: list[MemoryResult]) -> MemoryStore:
    """Return a mock MemoryStore whose query() returns `results`."""
    store = MagicMock(spec=MemoryStore)
    store.query.return_value = results
    return store


# ---------------------------------------------------------------------------
# _recency_score unit tests
# ---------------------------------------------------------------------------

class TestRecencyScore:
    def test_just_created_is_one(self):
        now = datetime.now(timezone.utc)
        assert _recency_score(now, now=now) == pytest.approx(1.0)

    def test_one_day_old(self):
        now = datetime.now(timezone.utc)
        created = now - timedelta(days=1)
        assert _recency_score(created, now=now) == pytest.approx(0.5)

    def test_older_is_lower(self):
        now = datetime.now(timezone.utc)
        score_1d = _recency_score(now - timedelta(days=1), now=now)
        score_7d = _recency_score(now - timedelta(days=7), now=now)
        assert score_1d > score_7d

    def test_naive_datetime_treated_as_utc(self):
        now = datetime.now(timezone.utc)
        naive = now.replace(tzinfo=None)
        # Should not raise and should return a sensible value
        score = _recency_score(naive, now=now)
        assert 0.0 < score <= 1.0

    def test_far_future_rounds_to_one(self):
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=1)  # days < 0 → clamped to 0
        assert _recency_score(future, now=now) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Retriever construction
# ---------------------------------------------------------------------------

class TestRetrieverConstruction:
    def test_default_weights(self):
        store = _store_returning([])
        r = Retriever(store)
        assert r.semantic_weight == pytest.approx(0.8)
        assert r.recency_weight == pytest.approx(0.2)

    def test_custom_weights(self):
        store = _store_returning([])
        r = Retriever(store, semantic_weight=0.6, recency_weight=0.4)
        assert r.semantic_weight == pytest.approx(0.6)

    def test_weights_must_sum_to_one(self):
        store = _store_returning([])
        with pytest.raises(ValueError, match="must equal 1.0"):
            Retriever(store, semantic_weight=0.5, recency_weight=0.3)


# ---------------------------------------------------------------------------
# Retriever.retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_empty_store_returns_empty(self):
        store = _store_returning([])
        r = Retriever(store)
        assert r.retrieve(_rand_vec()) == []

    def test_returns_retrieved_result_objects(self):
        store = _store_returning([_mem(distance=0.1)])
        r = Retriever(store)
        results = r.retrieve(_rand_vec(), top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievedResult)

    def test_top_k_respected(self):
        mems = [_mem(id=i, distance=float(i) * 0.05) for i in range(10)]
        store = _store_returning(mems)
        r = Retriever(store)
        results = r.retrieve(_rand_vec(), top_k=3)
        assert len(results) == 3

    def test_fetches_2x_candidates(self):
        """Retriever fetches top_k*2 from the vector store."""
        store = _store_returning([])
        r = Retriever(store)
        r.retrieve(_rand_vec(), top_k=5)
        store.query.assert_called_once_with(_rand_vec(), top_k=10)

    def test_results_ordered_by_final_score_descending(self):
        mems = [_mem(id=i, distance=float(i) * 0.1) for i in range(6)]
        store = _store_returning(mems)
        r = Retriever(store)
        results = r.retrieve(_rand_vec(), top_k=5)
        scores = [res.final_score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_computed(self):
        now = datetime.now(timezone.utc)
        mem = _mem(distance=0.2, created_at=now - timedelta(days=1))
        store = _store_returning([mem])
        r = Retriever(store)
        result = r.retrieve(_rand_vec(), top_k=1, now=now)[0]
        assert result.semantic_score == pytest.approx(0.8)
        assert result.recency_score == pytest.approx(0.5)
        expected_final = 0.8 * 0.8 + 0.2 * 0.5
        assert result.final_score == pytest.approx(expected_final)

    # ------------------------------------------------------------------
    # Acceptance criterion
    # ------------------------------------------------------------------

    def test_newer_ranks_higher_with_identical_embeddings(self):
        """Two chunks with identical embeddings: newer must rank higher."""
        now = datetime.now(timezone.utc)
        old_mem = _mem(id=1, chunk_text="old", distance=0.0,
                       created_at=now - timedelta(days=30))
        new_mem = _mem(id=2, chunk_text="new", distance=0.0,
                       created_at=now - timedelta(days=0))

        store = _store_returning([old_mem, new_mem])
        r = Retriever(store)
        results = r.retrieve(_rand_vec(), top_k=2, now=now)

        assert results[0].memory.chunk_text == "new"
        assert results[1].memory.chunk_text == "old"

    def test_semantic_weight_dominates_when_high(self):
        """With semantic_weight=1.0, the closer embedding wins regardless of age."""
        now = datetime.now(timezone.utc)
        old_close = _mem(id=1, chunk_text="close-old", distance=0.1,
                         created_at=now - timedelta(days=365))
        new_far = _mem(id=2, chunk_text="far-new", distance=0.9,
                       created_at=now)

        store = _store_returning([old_close, new_far])
        r = Retriever(store, semantic_weight=1.0, recency_weight=0.0)
        results = r.retrieve(_rand_vec(), top_k=2, now=now)
        assert results[0].memory.chunk_text == "close-old"
