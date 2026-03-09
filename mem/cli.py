"""CLI entrypoint for personal-memory (mem)."""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mem",
        description="Personal semantic memory backed by sqlite-vec.",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # mem store
    store_p = sub.add_parser("store", help="Store text in memory.")
    store_p.add_argument("text", help="Text to store.")

    # mem query
    query_p = sub.add_parser("query", help="Query memory by semantic similarity.")
    query_p.add_argument("text", help="Query text.")
    query_p.add_argument(
        "-k", "--top-k", type=int, default=5, metavar="K",
        help="Number of results to return (default: 5).",
    )

    # mem dedup
    dedup_p = sub.add_parser("dedup", help="Find duplicate chunks in the DB.")
    dedup_p.add_argument(
        "--dry-run", action="store_true",
        help="Report duplicates without deleting them.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "store":
        # Placeholder — wired up in issue #7 (CLI)
        print(f"[store] Not yet implemented. Text: {args.text!r}")
        return 0

    if args.command == "query":
        # Placeholder — wired up in issue #7 (CLI)
        print(f"[query] Not yet implemented. Query: {args.text!r}, top_k={args.top_k}")
        return 0

    if args.command == "dedup":
        from mem.store import MemoryStore
        with MemoryStore() as store:
            dupes = store.find_duplicates()
        if not dupes:
            print("No duplicates found.")
            return 0
        total_extra = sum(len(ids) - 1 for ids in dupes.values())
        print(f"Found {len(dupes)} duplicated fingerprint(s) — {total_extra} redundant row(s).")
        if args.dry_run:
            for fp, ids in dupes.items():
                print(f"  {fp[:16]}…  ids={ids}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
