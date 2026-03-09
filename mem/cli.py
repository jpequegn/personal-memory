"""CLI entrypoint for personal-memory (mem)."""

from __future__ import annotations

import json as _json
import sys

import click

from mem.store import MemoryStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _store() -> MemoryStore:
    return MemoryStore()


def _score_color(score: float) -> str:
    if score >= 0.7:
        return "green"
    if score >= 0.4:
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option("0.1.0")
def cli() -> None:
    """Personal semantic memory backed by sqlite-vec."""


# ---------------------------------------------------------------------------
# mem store
# ---------------------------------------------------------------------------

@cli.command("store")
@click.argument("text", required=False)
@click.option("--file", "-f", "filepath", type=click.Path(exists=True),
              help="Ingest a file instead of inline text.")
@click.option("--stdin", "from_stdin", is_flag=True,
              help="Read text from stdin.")
@click.option("--source", "-s", default=None, help="Label for this memory's source.")
@click.option("--tag", "-t", "tags", multiple=True, help="Tags (repeatable).")
def store_cmd(
    text: str | None,
    filepath: str | None,
    from_stdin: bool,
    source: str | None,
    tags: tuple[str, ...],
) -> None:
    """Store text in memory.

    Provide TEXT inline, --file PATH, or --stdin.
    """
    if from_stdin:
        content = sys.stdin.read()
    elif filepath:
        with open(filepath) as fh:
            content = fh.read()
        if source is None:
            source = filepath
    elif text:
        content = text
    else:
        raise click.UsageError("Provide TEXT, --file, or --stdin.")

    with _store() as s:
        stored, skipped = s.add(content, source=source, tags=list(tags))

    click.echo(f"Stored {stored} chunk(s).", err=False)
    if skipped:
        click.echo(f"Skipped {skipped} duplicate chunk(s).", err=False)


# ---------------------------------------------------------------------------
# mem query
# ---------------------------------------------------------------------------

@cli.command("query")
@click.argument("text")
@click.option("-k", "--top-k", default=5, show_default=True,
              help="Number of results to return.")
@click.option("--semantic-weight", default=0.8, show_default=True,
              help="Weight for semantic similarity (0–1).")
@click.option("--recency-weight", default=0.2, show_default=True,
              help="Weight for recency (0–1). Must sum to 1 with --semantic-weight.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def query_cmd(
    text: str,
    top_k: int,
    semantic_weight: float,
    recency_weight: float,
    as_json: bool,
) -> None:
    """Query memory by semantic similarity."""
    from mem.embedder import embed
    from mem.retriever import Retriever

    query_vec = embed(text)

    with _store() as s:
        retriever = Retriever(s, semantic_weight=semantic_weight,
                              recency_weight=recency_weight)
        results = retriever.retrieve(query_vec, top_k=top_k)

    if as_json:
        output = [
            {
                "id": r.memory.id,
                "chunk_text": r.memory.chunk_text,
                "source": r.memory.source,
                "tags": r.memory.tags,
                "created_at": r.memory.created_at.isoformat(),
                "semantic_score": round(r.semantic_score, 4),
                "recency_score": round(r.recency_score, 4),
                "final_score": round(r.final_score, 4),
            }
            for r in results
        ]
        click.echo(_json.dumps(output, indent=2))
        return

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        score_str = click.style(f"{r.final_score:.3f}", fg=_score_color(r.final_score))
        source_str = click.style(r.memory.source or "—", dim=True)
        click.echo(f"\n{i}. [{score_str}] {source_str}")
        click.echo(f"   {r.memory.chunk_text}")


# ---------------------------------------------------------------------------
# mem list
# ---------------------------------------------------------------------------

@cli.command("list")
@click.option("-n", default=20, show_default=True, help="Number of entries to show.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def list_cmd(n: int, as_json: bool) -> None:
    """Show recent memory entries."""
    with _store() as s:
        rows = s._con.execute(
            "SELECT id, chunk_text, source, tags, created_at FROM memories "
            "ORDER BY created_at DESC LIMIT ?",
            (n,),
        ).fetchall()

    if as_json:
        click.echo(_json.dumps(
            [dict(r) for r in rows], indent=2, default=str
        ))
        return

    if not rows:
        click.echo("No memories stored yet.")
        return

    for r in rows:
        ts = click.style(r["created_at"][:19], dim=True)
        src = click.style(r["source"] or "—", dim=True)
        preview = r["chunk_text"][:80].replace("\n", " ")
        click.echo(f"[{r['id']}] {ts}  {src}\n    {preview}")


# ---------------------------------------------------------------------------
# mem stats
# ---------------------------------------------------------------------------

@cli.command("stats")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def stats_cmd(as_json: bool) -> None:
    """Show memory store statistics."""
    with _store() as s:
        count = s.count()
        sources = s.list_sources()
        row = s._con.execute(
            "SELECT MIN(created_at) AS oldest, MAX(created_at) AS newest FROM memories"
        ).fetchone()

    data = {
        "total_chunks": count,
        "sources": sources,
        "oldest": row["oldest"],
        "newest": row["newest"],
    }

    if as_json:
        click.echo(_json.dumps(data, indent=2))
        return

    click.echo(f"Total chunks : {count}")
    click.echo(f"Sources      : {', '.join(sources) if sources else '—'}")
    click.echo(f"Oldest       : {data['oldest'] or '—'}")
    click.echo(f"Newest       : {data['newest'] or '—'}")


# ---------------------------------------------------------------------------
# mem dedup
# ---------------------------------------------------------------------------

@cli.command("dedup")
@click.option("--dry-run", is_flag=True,
              help="Report duplicates without deleting.")
def dedup_cmd(dry_run: bool) -> None:
    """Find (and optionally report) duplicate chunks in the DB."""
    with _store() as s:
        dupes = s.find_duplicates()

    if not dupes:
        click.echo("No duplicates found.")
        return

    total_extra = sum(len(ids) - 1 for ids in dupes.values())
    click.echo(
        f"Found {len(dupes)} duplicated fingerprint(s) — "
        f"{total_extra} redundant row(s)."
    )
    if dry_run:
        for fp, ids in dupes.items():
            click.echo(f"  {fp[:16]}…  ids={ids}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    cli()


if __name__ == "__main__":
    main()
