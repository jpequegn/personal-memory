"""
Text chunker: split input into overlapping chunks suitable for embedding.

Strategy:
- Split on sentence boundaries first (preserves semantic coherence)
- Group sentences into chunks of max_tokens size
- Add overlap by including trailing tokens from previous chunk
"""

import re


def _count_tokens(text: str) -> int:
    """Approximate token count using chars/4 heuristic."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving punctuation."""
    # Split on sentence-ending punctuation followed by whitespace or end of string
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def chunk(text: str, max_tokens: int = 256, overlap: int = 32) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Input text to split.
        max_tokens: Maximum tokens per chunk (default 256).
        overlap: Number of tokens to overlap between consecutive chunks (default 32).

    Returns:
        List of text chunks, each <= max_tokens, with overlap between consecutive chunks.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)

        # If a single sentence exceeds max_tokens, split it by words
        if sentence_tokens > max_tokens:
            # Flush current buffer first
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0
            # Hard-split the long sentence by words
            words = sentence.split()
            word_buf: list[str] = []
            word_tokens = 0
            for word in words:
                wt = _count_tokens(word)
                if word_tokens + wt > max_tokens and word_buf:
                    chunks.append(" ".join(word_buf))
                    # Keep overlap words for next chunk
                    overlap_words = _trim_to_tokens(word_buf, overlap)
                    word_buf = overlap_words + [word]
                    word_tokens = _count_tokens(" ".join(word_buf))
                else:
                    word_buf.append(word)
                    word_tokens += wt
            if word_buf:
                current_sentences = word_buf
                current_tokens = word_tokens
            continue

        if current_tokens + sentence_tokens > max_tokens and current_sentences:
            # Emit current chunk
            chunks.append(" ".join(current_sentences))
            # Build overlap: take trailing tokens from the just-emitted sentences
            overlap_sentences = _trailing_sentences(current_sentences, overlap)
            current_sentences = overlap_sentences + [sentence]
            current_tokens = _count_tokens(" ".join(current_sentences))
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def _trim_to_tokens(words: list[str], max_tok: int) -> list[str]:
    """Return trailing words from `words` that fit within `max_tok` tokens."""
    result: list[str] = []
    for word in reversed(words):
        candidate = [word] + result
        if _count_tokens(" ".join(candidate)) <= max_tok:
            result = candidate
        else:
            break
    return result


def _trailing_sentences(sentences: list[str], max_tok: int) -> list[str]:
    """Return trailing sentences that fit within `max_tok` tokens."""
    result: list[str] = []
    for s in reversed(sentences):
        candidate = [s] + result
        if _count_tokens(" ".join(candidate)) <= max_tok:
            result = candidate
        else:
            break
    return result
