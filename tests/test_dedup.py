"""Tests for deduplication: fingerprint-based skip on insert and add."""

import math
import random
from unittest.mock import patch

import pytest

from mem.store import MemoryStore

_DIM = 1536


def _rand_vec(seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(_DIM)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _patch_embed(texts):
    return [_rand_vec(i) for i in range(len(texts))]


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with MemoryStore(db) as s:
        yield s


# ---------------------------------------------------------------------------
# fingerprint_exists
# ---------------------------------------------------------------------------

class TestFingerprintExists:
    def test_false_when_empty(self, store):
        assert store.fingerprint_exists("abc") is False

    def test_true_after_insert(self, store):
        store.insert("hello", _rand_vec(0), fingerprint="fp1")
        assert store.fingerprint_exists("fp1") is True

    def test_false_for_different_fingerprint(self, store):
        store.insert("hello", _rand_vec(0), fingerprint="fp1")
        assert store.fingerprint_exists("fp2") is False


# ---------------------------------------------------------------------------
# insert skip_duplicates
# ---------------------------------------------------------------------------

class TestInsertSkipDuplicates:
    def test_skip_returns_none(self, store):
        store.insert("text", _rand_vec(0), fingerprint="fp")
        result = store.insert("text", _rand_vec(1), fingerprint="fp",
                              skip_duplicates=True)
        assert result is None

    def test_skip_does_not_increase_count(self, store):
        store.insert("text", _rand_vec(0), fingerprint="fp")
        store.insert("text", _rand_vec(1), fingerprint="fp",
                     skip_duplicates=True)
        assert store.count() == 1

    def test_no_skip_flag_allows_duplicate(self, store):
        store.insert("text", _rand_vec(0), fingerprint="fp")
        store.insert("text", _rand_vec(1), fingerprint="fp")
        assert store.count() == 2

    def test_skip_without_fingerprint_still_inserts(self, store):
        # No fingerprint means no dedup check
        r = store.insert("text", _rand_vec(0), fingerprint=None,
                         skip_duplicates=True)
        assert r is not None
        assert store.count() == 1


# ---------------------------------------------------------------------------
# add() deduplication
# ---------------------------------------------------------------------------

class TestAddDedup:
    def test_second_run_stores_zero_chunks(self, store):
        """Acceptance: store same text twice — second run stores 0 new chunks."""
        text = "The sky is blue. Water is wet. Fire is hot."
        with patch("mem.embedder.embed_batch", side_effect=_patch_embed):
            stored1, skipped1 = store.add(text, source="first")
        assert stored1 > 0
        assert skipped1 == 0

        with patch("mem.embedder.embed_batch", side_effect=_patch_embed):
            stored2, skipped2 = store.add(text, source="second")
        assert stored2 == 0
        assert skipped2 == stored1

    def test_partial_overlap_stores_only_new(self, store):
        # Pre-seed one chunk with a known fingerprint
        store.insert("shared chunk", _rand_vec(0), fingerprint="shared-fp")

        # Build a text whose first chunk has that same fingerprint — simulate
        # via add() by pre-inserting one of the chunks that add() would produce.
        # Instead, test at the insert level with skip_duplicates directly.
        r1 = store.insert("shared chunk", _rand_vec(1), fingerprint="shared-fp",
                          skip_duplicates=True)
        r2 = store.insert("unique chunk", _rand_vec(2), fingerprint="unique-fp",
                          skip_duplicates=True)
        assert r1 is None   # skipped
        assert r2 is not None  # stored
        assert store.count() == 2  # original + unique

    def test_skipped_chunks_not_re_embedded(self, store):
        """Duplicate chunks must not be sent to the embedding API."""
        text = "Unique sentence here. Another sentence follows."
        with patch("mem.embedder.embed_batch", side_effect=_patch_embed) as m:
            store.add(text)
        first_call_count = m.call_count

        with patch("mem.embedder.embed_batch", side_effect=_patch_embed) as m:
            store.add(text)
        # Second run: all chunks are duplicates → embed_batch never called
        assert m.call_count == 0

    def test_total_count_unchanged_after_duplicate_add(self, store):
        text = "One. Two. Three."
        with patch("mem.embedder.embed_batch", side_effect=_patch_embed):
            stored, _ = store.add(text)
        count_after_first = store.count()

        with patch("mem.embedder.embed_batch", side_effect=_patch_embed):
            store.add(text)
        assert store.count() == count_after_first


# ---------------------------------------------------------------------------
# find_duplicates
# ---------------------------------------------------------------------------

class TestFindDuplicates:
    def test_empty_db_returns_empty(self, store):
        assert store.find_duplicates() == {}

    def test_no_duplicates_returns_empty(self, store):
        store.insert("a", _rand_vec(0), fingerprint="fp1")
        store.insert("b", _rand_vec(1), fingerprint="fp2")
        assert store.find_duplicates() == {}

    def test_detects_duplicates(self, store):
        store.insert("x", _rand_vec(0), fingerprint="dup")
        store.insert("x", _rand_vec(1), fingerprint="dup")
        dupes = store.find_duplicates()
        assert "dup" in dupes
        assert len(dupes["dup"]) == 2

    def test_ignores_null_fingerprints(self, store):
        store.insert("x", _rand_vec(0), fingerprint=None)
        store.insert("x", _rand_vec(1), fingerprint=None)
        assert store.find_duplicates() == {}
