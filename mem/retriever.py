"""Retriever: blends semantic similarity with recency for re-ranked results.

final_score = semantic_weight * semantic_score + recency_weight * recency_score

Where:
  semantic_score = 1 - distance          (vec0 L2 distance; lower = more similar)
  recency_score  = 1 / (1 + days_since)  (decays toward 0 as memory ages)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from mem.store import MemoryResult, MemoryStore

_DEFAULT_SEMANTIC_WEIGHT = 0.8
_DEFAULT_RECENCY_WEIGHT = 0.2


@dataclass
class RetrievedResult:
    memory: MemoryResult
    semantic_score: float
    recency_score: float
    final_score: float


def _recency_score(created_at: datetime, now: datetime | None = None) -> float:
    """1 / (1 + days_since_created).  Returns 1.0 for a memory created right now."""
    if now is None:
        now = datetime.now(timezone.utc)
    # Make both timezone-aware for comparison
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    days = max(0.0, (now - created_at).total_seconds() / 86_400)
    return 1.0 / (1.0 + days)


class Retriever:
    """Wraps MemoryStore.query() and re-ranks results using a blended score."""

    def __init__(
        self,
        store: MemoryStore,
        semantic_weight: float = _DEFAULT_SEMANTIC_WEIGHT,
        recency_weight: float = _DEFAULT_RECENCY_WEIGHT,
    ) -> None:
        if abs(semantic_weight + recency_weight - 1.0) > 1e-9:
            raise ValueError("semantic_weight + recency_weight must equal 1.0")
        self.store = store
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight

    def retrieve(
        self,
        embedding: list[float],
        top_k: int = 5,
        *,
        now: datetime | None = None,
    ) -> list[RetrievedResult]:
        """Query the store and return top_k results re-ranked by blended score.

        Fetches 2×top_k candidates from the vector index to give the re-ranker
        enough candidates to work with when recency pulls older items down.
        """
        candidates = self.store.query(embedding, top_k=top_k * 2)
        if not candidates:
            return []

        scored: list[RetrievedResult] = []
        for mem in candidates:
            # vec0 returns L2 distance; convert to a [0,1] similarity score.
            # Distance is always >= 0; we cap semantic_score at 1 for distance=0.
            sem = max(0.0, 1.0 - mem.distance)
            rec = _recency_score(mem.created_at, now=now)
            final = self.semantic_weight * sem + self.recency_weight * rec
            scored.append(RetrievedResult(
                memory=mem,
                semantic_score=sem,
                recency_score=rec,
                final_score=final,
            ))

        scored.sort(key=lambda r: r.final_score, reverse=True)
        return scored[:top_k]
