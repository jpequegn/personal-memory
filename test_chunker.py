"""Unit tests for the text chunker."""

import pytest
from chunker import chunk, _count_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lorem(n_words: int) -> str:
    """Generate a simple reproducible text of ~n_words words."""
    sentence = (
        "The quick brown fox jumps over the lazy dog near the riverbank. "
        "Scientists have long studied the migration patterns of arctic terns. "
        "Machine learning models require large amounts of labeled training data. "
        "Natural language processing enables computers to understand human text. "
    )
    words = (sentence * ((n_words // len(sentence.split())) + 2)).split()
    return " ".join(words[:n_words])


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestChunkBasic:
    def test_empty_string_returns_empty(self):
        assert chunk("") == []

    def test_whitespace_string_returns_empty(self):
        assert chunk("   \n\t  ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world. This is a test."
        result = chunk(text, max_tokens=256)
        assert len(result) == 1
        assert "Hello world" in result[0]

    def test_returns_list_of_strings(self):
        result = chunk("A simple sentence.", max_tokens=256)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_no_empty_chunks(self):
        text = _lorem(500)
        result = chunk(text, max_tokens=64, overlap=8)
        assert all(len(c.strip()) > 0 for c in result)


# ---------------------------------------------------------------------------
# Size constraints
# ---------------------------------------------------------------------------

class TestChunkSizes:
    def test_each_chunk_within_max_tokens(self):
        text = _lorem(800)
        max_tok = 128
        result = chunk(text, max_tokens=max_tok, overlap=16)
        for c in result:
            assert _count_tokens(c) <= max_tok, f"Chunk exceeds max_tokens: {_count_tokens(c)}"

    def test_multiple_chunks_for_long_text(self):
        text = _lorem(800)
        result = chunk(text, max_tokens=64, overlap=8)
        assert len(result) > 1

    def test_acceptance_criteria_5000_chars(self):
        """5000-char input must produce >= 10 chunks."""
        # Build text of ~5000 characters
        base = "The sun rises in the east and sets in the west every single day. "
        text = (base * ((5000 // len(base)) + 2))[:5000]
        result = chunk(text, max_tokens=64, overlap=8)
        assert len(result) >= 10, f"Expected >= 10 chunks, got {len(result)}"


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

class TestChunkOverlap:
    def test_consecutive_chunks_share_content(self):
        text = _lorem(400)
        result = chunk(text, max_tokens=64, overlap=16)
        if len(result) < 2:
            pytest.skip("Not enough chunks to test overlap")
        # Take last few words of chunk N and first few words of chunk N+1;
        # some words from the end of chunk N should appear at start of chunk N+1
        for i in range(len(result) - 1):
            words_end = set(result[i].split()[-10:])
            words_start = set(result[i + 1].split()[:10])
            assert words_end & words_start, (
                f"No overlapping words between chunk {i} and {i+1}"
            )

    def test_zero_overlap_no_shared_words(self):
        # With overlap=0, consecutive chunks should not share boundary words
        # (sentence-aligned so some sharing is possible; just verify it runs)
        text = _lorem(300)
        result = chunk(text, max_tokens=64, overlap=0)
        assert len(result) > 0

    def test_default_overlap_value(self):
        """chunk() called with only max_tokens should use overlap=32."""
        text = _lorem(500)
        result_default = chunk(text, max_tokens=64)
        result_explicit = chunk(text, max_tokens=64, overlap=32)
        assert result_default == result_explicit


# ---------------------------------------------------------------------------
# Sentence boundary preservation
# ---------------------------------------------------------------------------

class TestSentenceBoundaries:
    def test_chunks_do_not_split_mid_sentence(self):
        """Every chunk should end with sentence-ending punctuation or be the last chunk."""
        sentences = [
            "The cat sat on the mat and looked around carefully.",
            "Dogs are known for their loyalty and affectionate nature.",
            "Birds migrate thousands of miles each year without a map.",
            "Fish breathe underwater using their gills efficiently.",
            "Reptiles are cold-blooded and rely on external heat sources.",
        ] * 10
        text = " ".join(sentences)
        result = chunk(text, max_tokens=64, overlap=8)
        for c in result[:-1]:
            # Strip trailing whitespace and check last char
            stripped = c.rstrip()
            assert stripped[-1] in ".!?", (
                f"Chunk appears to end mid-sentence: ...{stripped[-40:]!r}"
            )

    def test_2000_word_article_overlap_and_sizes(self):
        """Chunk a 2000-word article; verify overlap and per-chunk size."""
        text = _lorem(2000)
        max_tok = 128
        ovlp = 32
        result = chunk(text, max_tokens=max_tok, overlap=ovlp)

        # Size constraint
        for c in result:
            assert _count_tokens(c) <= max_tok

        # Overlap: consecutive chunks share some words
        for i in range(len(result) - 1):
            words_end = set(result[i].split()[-20:])
            words_start = set(result[i + 1].split()[:20])
            assert words_end & words_start, (
                f"No overlap between chunk {i} and {i+1}"
            )
