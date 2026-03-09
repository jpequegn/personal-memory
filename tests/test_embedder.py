"""Tests for the embedding pipeline (embedder + MemoryStore.add)."""

import hashlib
import math
import random
from unittest.mock import MagicMock, patch

import pytest

from mem.embedder import embed, embed_batch, _BATCH_SIZE, _MODEL
from mem.store import MemoryStore

_DIM = 1536


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedding(seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(_DIM)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _make_openai_response(texts: list[str], seed_offset: int = 0):
    """Build a mock openai embeddings response."""
    items = []
    for i, _ in enumerate(texts):
        item = MagicMock()
        item.index = i
        item.embedding = _fake_embedding(i + seed_offset)
        items.append(item)
    resp = MagicMock()
    resp.data = items
    return resp


# ---------------------------------------------------------------------------
# embed / embed_batch unit tests (mocked API)
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    def test_empty_input_returns_empty(self):
        assert embed_batch([]) == []

    def test_single_text_returns_one_embedding(self):
        with patch("mem.embedder._client") as mock_client_fn:
            client = MagicMock()
            mock_client_fn.return_value = client
            client.embeddings.create.return_value = _make_openai_response(["hello"])

            result = embed_batch(["hello"])

        assert len(result) == 1
        assert len(result[0]) == _DIM

    def test_multiple_texts_within_batch(self):
        texts = [f"text {i}" for i in range(5)]
        with patch("mem.embedder._client") as mock_client_fn:
            client = MagicMock()
            mock_client_fn.return_value = client
            client.embeddings.create.return_value = _make_openai_response(texts)

            result = embed_batch(texts)

        assert len(result) == 5
        assert all(len(v) == _DIM for v in result)

    def test_uses_correct_model(self):
        with patch("mem.embedder._client") as mock_client_fn:
            client = MagicMock()
            mock_client_fn.return_value = client
            client.embeddings.create.return_value = _make_openai_response(["x"])

            embed_batch(["x"])

        client.embeddings.create.assert_called_once()
        call_kwargs = client.embeddings.create.call_args
        assert call_kwargs.kwargs.get("model") == _MODEL or call_kwargs.args[0] == _MODEL or \
               call_kwargs.kwargs.get("model") == _MODEL

    def test_large_input_split_into_batches(self):
        n = _BATCH_SIZE + 10
        texts = [f"t{i}" for i in range(n)]

        call_count = 0
        def fake_create(model, input):
            nonlocal call_count
            call_count += 1
            return _make_openai_response(input, seed_offset=call_count * 1000)

        with patch("mem.embedder._client") as mock_client_fn:
            client = MagicMock()
            mock_client_fn.return_value = client
            client.embeddings.create.side_effect = fake_create

            result = embed_batch(texts)

        assert len(result) == n
        assert call_count == 2  # ceil((BATCH_SIZE+10) / BATCH_SIZE)

    def test_embed_single_delegates_to_batch(self):
        with patch("mem.embedder.embed_batch") as mock_batch:
            mock_batch.return_value = [_fake_embedding(0)]
            result = embed("hello world")

        mock_batch.assert_called_once_with(["hello world"])
        assert result == _fake_embedding(0)

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            from mem import embedder
            embedder._client()


# ---------------------------------------------------------------------------
# MemoryStore.add integration (mocked embedder)
# ---------------------------------------------------------------------------

class TestMemoryStoreAdd:
    @pytest.fixture
    def store(self, tmp_path):
        db = tmp_path / "test.db"
        with MemoryStore(db) as s:
            yield s

    def _patch_embed(self, texts):
        return [_fake_embedding(i) for i in range(len(texts))]

    def test_add_returns_chunk_count(self, store):
        with patch("mem.embedder.embed_batch", side_effect=self._patch_embed):
            stored, skipped = store.add("Hello world. This is a short text.", source="test")
        assert stored >= 1
        assert skipped == 0

    def test_add_stores_chunks_in_db(self, store):
        article = " ".join(["word"] * 300)  # ~300 words → multiple chunks
        with patch("mem.embedder.embed_batch", side_effect=self._patch_embed):
            stored, _ = store.add(article, source="article", tags=["test"])
        assert store.count() == stored
        assert stored > 1

    def test_add_sets_fingerprint(self, store):
        text = "A single sentence that fits in one chunk."
        with patch("mem.embedder.embed_batch", side_effect=self._patch_embed):
            store.add(text, source="fp-test")
        results = store.query(_fake_embedding(0), top_k=1)
        assert results[0].fingerprint != ""
        # Verify it's a sha256 hex string (64 chars)
        assert len(results[0].fingerprint) == 64

    def test_add_propagates_source_and_tags(self, store):
        with patch("mem.embedder.embed_batch", side_effect=self._patch_embed):
            store.add("Some text here.", source="wiki", tags=["a", "b"])
        results = store.query(_fake_embedding(0), top_k=1)
        assert results[0].source == "wiki"
        assert results[0].tags == ["a", "b"]

    def test_add_empty_text_returns_zero(self, store):
        with patch("mem.embedder.embed_batch", side_effect=self._patch_embed):
            stored, skipped = store.add("")
        assert stored == 0
        assert skipped == 0
        assert store.count() == 0

    def test_add_1000_word_article_acceptance(self, store, tmp_path):
        """Acceptance: store a 1000-word article, confirm all chunks in DB."""
        words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 100).split()
        article = " ".join(words[:1000])

        with patch("mem.embedder.embed_batch", side_effect=self._patch_embed):
            stored, _ = store.add(article, source="1000-word-article")

        assert stored > 0
        assert store.count() == stored
        assert "1000-word-article" in store.list_sources()
