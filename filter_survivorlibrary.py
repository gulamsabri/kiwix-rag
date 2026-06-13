#!/usr/bin/env python3
"""
Filter low-quality chunks from survivorlibrary ChromaDB collections.

Targets two categories of noise:
  1. Classified ads / equipment-for-sale pages from vintage magazines
  2. Conspiracy/misinformation keywords

Source JSONL files on the SSD are untouched — deletions are reversible by re-indexing.

Usage:
    python filter_survivorlibrary.py --dry-run         # report only, no changes
    python filter_survivorlibrary.py                   # delete flagged chunks
    python filter_survivorlibrary.py --show 5          # print sample bad chunks in dry-run
    python filter_survivorlibrary.py --threshold 2     # require score >= N to delete (default 2)
    python filter_survivorlibrary.py --db /path/to/db
"""

import argparse
import re
import sys
from pathlib import Path

import chromadb
from chunk_filter import score_chunk, DEFAULT_THRESHOLD


def process_collection(col, threshold: int, dry_run: bool, show_samples: int) -> tuple[int, int]:
    """Return (total_chunks, flagged_count)."""
    total = col.count()
    print(f"\n  {col.name}  ({total:,} chunks)")

    batch_size = 500
    to_delete: list[str] = []
    samples: list[dict] = []

    for offset in range(0, total, batch_size):
        result = col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = result["ids"]
        docs = result["documents"]
        metas = result["metadatas"]

        for chunk_id, text, meta in zip(ids, docs, metas):
            source = (meta or {}).get("source", "")
            s, reasons = score_chunk(text)
            if s >= threshold:
                to_delete.append(chunk_id)
                if show_samples and len(samples) < show_samples:
                    samples.append({
                        "source": source,
                        "score": s,
                        "reasons": reasons,
                        "text": text,
                    })

        pct = min(offset + batch_size, total)
        print(f"\r    scanned {pct:,}/{total:,} …", end="", flush=True)

    print(f"\r    scanned {total:,}/{total:,} — {len(to_delete):,} flagged ({100*len(to_delete)/(total or 1):.1f}%)")

    if samples:
        print(f"\n  --- Sample flagged chunks ---")
        for s in samples:
            snippet = s["text"].replace("\n", " ").strip()[:250]
            print(f"  source: {s['source']}")
            print(f"  score:  {s['score']}  reasons: {s['reasons'][:3]}")
            print(f"  text:   {snippet}{'...' if len(s['text']) > 250 else ''}")
            print()

    if not dry_run and to_delete:
        delete_batch = 500
        for i in range(0, len(to_delete), delete_batch):
            col.delete(ids=to_delete[i:i+delete_batch])
        print(f"  Deleted {len(to_delete):,} chunks.")
    elif dry_run and to_delete:
        print(f"  [dry-run] Would delete {len(to_delete):,} chunks. Re-run without --dry-run to commit.")

    return total, len(to_delete)


def main():
    parser = argparse.ArgumentParser(description="Filter low-quality chunks from survivorlibrary collections.")
    parser.add_argument("--db", default=str(Path(__file__).parent / "vector_db"),
                        help="ChromaDB path (default: ./vector_db)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report flagged chunks without deleting")
    parser.add_argument("--show", type=int, default=0, metavar="N",
                        help="Print N sample flagged chunks per collection (default: 0)")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"Minimum score to flag a chunk for deletion (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--collection", default="",
                        help="Filter only this collection name (default: all survivorlibrary_* collections)")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.db)
    all_names = [c.name for c in client.list_collections()]

    if args.collection:
        target_names = [n for n in all_names if n == args.collection]
        if not target_names:
            print(f"Collection '{args.collection}' not found.", file=sys.stderr)
            sys.exit(1)
    else:
        target_names = sorted(n for n in all_names if "survivorlibrary" in n)

    if not target_names:
        print("No survivorlibrary collections found.", file=sys.stderr)
        sys.exit(1)

    mode = "DRY RUN — no changes will be made" if args.dry_run else "LIVE — chunks will be deleted"
    print(f"Mode:       {mode}")
    print(f"DB:         {args.db}")
    print(f"Threshold:  score >= {args.threshold}")
    print(f"Collections ({len(target_names)}): {', '.join(target_names)}")

    grand_total = 0
    grand_flagged = 0

    for name in target_names:
        col = client.get_collection(name)
        total, flagged = process_collection(col, args.threshold, args.dry_run, args.show)
        grand_total += total
        grand_flagged += flagged

    print(f"\n{'='*60}")
    print(f"Total chunks:   {grand_total:,}")
    print(f"Total flagged:  {grand_flagged:,}  ({100*grand_flagged/(grand_total or 1):.1f}%)")
    if args.dry_run:
        print(f"Remaining:      {grand_total - grand_flagged:,}  (if you run without --dry-run)")
    else:
        print(f"Remaining:      {grand_total - grand_flagged:,}")


if __name__ == "__main__":
    main()
