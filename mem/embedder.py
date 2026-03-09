"""Embedding pipeline: text → list[float] via OpenAI text-embedding-3-small."""

import os

_MODEL = "text-embedding-3-small"
_DIM = 1536
_BATCH_SIZE = 100


def _client():
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def embed(text: str) -> list[float]:
    """Embed a single string. Returns a 1536-dim float list."""
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed up to _BATCH_SIZE texts in one API call.

    Splits larger lists into batches automatically.
    """
    if not texts:
        return []

    client = _client()
    results: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = client.embeddings.create(model=_MODEL, input=batch)
        # Response items are ordered by index
        batch_vecs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        results.extend(batch_vecs)

    return results
