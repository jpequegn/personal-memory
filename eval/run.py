#!/usr/bin/env python3
"""Eval: precision@5 for the personal-memory retrieval pipeline.

Protocol
--------
1. Embed and store all 50 items from items.json into a temp DB.
2. For each query in queries.json, retrieve top-5 results.
3. Count how many retrieved chunks belong to the expected topic.
4. precision@5 = correct / 5  (per query), averaged per topic and overall.

Target: overall precision@5 >= 0.7

Usage
-----
    uv run python eval/run.py
    uv run python eval/run.py --no-cache   # re-embed even if DB exists
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ITEMS_FILE = HERE / "items.json"
QUERIES_FILE = HERE / "queries.json"
K = 5


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def build_store(db_path: Path, items: list[dict]) -> None:
    """Embed items and insert into the store (skip if already populated)."""
    from mem.embedder import embed_batch
    from mem.store import MemoryStore

    with MemoryStore(db_path) as store:
        if store.count() == len(items):
            print(f"  Store already has {len(items)} items — skipping embedding.")
            return

        print(f"  Embedding {len(items)} items …", flush=True)
        texts = [item["text"] for item in items]
        embeddings = embed_batch(texts)

        for item, embedding in zip(items, embeddings):
            import hashlib
            fp = hashlib.sha256(item["text"].encode()).hexdigest()
            store.insert(
                item["text"],
                embedding,
                source=item["topic"],
                tags=[item["topic"]],
                fingerprint=fp,
            )
        print(f"  Stored {store.count()} items.")


def run_queries(db_path: Path, queries: list[dict], items: list[dict]) -> dict:
    """Run all queries and return per-query precision scores."""
    from mem.embedder import embed_batch
    from mem.retriever import Retriever
    from mem.store import MemoryStore

    # Build lookup: chunk_text → topic
    text_to_topic = {item["text"]: item["topic"] for item in items}

    print(f"\n  Running {len(queries)} queries …", flush=True)
    query_texts = [q["query"] for q in queries]
    query_embeddings = embed_batch(query_texts)

    results_by_query: dict[int, dict] = {}

    with MemoryStore(db_path) as store:
        retriever = Retriever(store, semantic_weight=0.9, recency_weight=0.1)
        for query, embedding in zip(queries, query_embeddings):
            retrieved = retriever.retrieve(embedding, top_k=K)
            correct = sum(
                1 for r in retrieved
                if text_to_topic.get(r.memory.chunk_text) == query["topic"]
            )
            precision = correct / K
            results_by_query[query["id"]] = {
                "query": query["query"],
                "topic": query["topic"],
                "precision_at_k": precision,
                "correct": correct,
                "retrieved": [r.memory.chunk_text[:60] for r in retrieved],
            }

    return results_by_query


def print_report(results: dict, k: int = K) -> float:
    """Print per-topic and overall precision@k. Returns overall score."""
    by_topic: dict[str, list[float]] = {}
    for r in results.values():
        by_topic.setdefault(r["topic"], []).append(r["precision_at_k"])

    print(f"\n{'─' * 60}")
    print(f"  Precision@{k} results")
    print(f"{'─' * 60}")

    all_scores: list[float] = []
    for topic, scores in sorted(by_topic.items()):
        avg = sum(scores) / len(scores)
        all_scores.extend(scores)
        bar = "█" * int(avg * 20)
        print(f"  {topic:<35} {avg:.2f}  {bar}")

    overall = sum(all_scores) / len(all_scores)
    print(f"{'─' * 60}")
    print(f"  {'Overall':<35} {overall:.2f}")
    print(f"{'─' * 60}")

    target = 0.7
    status = "✓ PASS" if overall >= target else "✗ FAIL"
    print(f"\n  Target ≥ {target:.2f}  →  {status} ({overall:.2f})\n")
    return overall


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval eval.")
    parser.add_argument("--db", type=Path, default=None,
                        help="Path to eval DB (default: temp file).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-embed even if DB already exists.")
    args = parser.parse_args()

    items = load_json(ITEMS_FILE)
    queries = load_json(QUERIES_FILE)

    print(f"\npersonal-memory eval  ({len(items)} items, {len(queries)} queries, k={K})")
    print("=" * 60)

    use_temp = args.db is None
    if use_temp:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(tmp.name)
        tmp.close()
    else:
        db_path = args.db

    if args.no_cache and db_path.exists():
        db_path.unlink()

    print(f"\n[1/2] Building store  →  {db_path}")
    build_store(db_path, items)

    print("\n[2/2] Querying")
    results = run_queries(db_path, queries, items)

    overall = print_report(results)

    if use_temp:
        db_path.unlink(missing_ok=True)

    return 0 if overall >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
