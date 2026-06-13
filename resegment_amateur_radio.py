#!/usr/bin/env python3
"""
Split survivorlibrary_amateur_radio into decade sub-collections.

73 Magazine ran 1960–2003. Splitting by decade keeps each collection well under
the Pi's --max-collection-size limit while preserving semantic coherence
(radio technology evolved significantly each decade: tubes → solid-state →
digital → packet/internet era).

New collections:
  survivorlibrary_amateur_radio_1960s  — 1960–1969 issues
  survivorlibrary_amateur_radio_1970s  — 1970–1979 issues
  survivorlibrary_amateur_radio_1980s  — 1980–1989 issues
  survivorlibrary_amateur_radio_1990s  — 1990–2003 issues + nav/index pages

Usage:
    python resegment_amateur_radio.py --dry-run
    python resegment_amateur_radio.py
    python resegment_amateur_radio.py --drop-old
"""

import argparse
import re
import sys
from pathlib import Path

import chromadb

SOURCE_COLLECTION = "survivorlibrary_amateur_radio"
DECADES = ["1960s", "1970s", "1980s", "1990s"]
TARGET_PREFIX = "survivorlibrary_amateur_radio"
_TARGET_NAMES = {f"{TARGET_PREFIX}_{d}" for d in DECADES}

FETCH_BATCH = 200   # small: fetching full embedding vectors

_YEAR_RE = re.compile(r'73_magazine_(\d{4})')


def classify(source: str) -> str:
    m = _YEAR_RE.search(source)
    if not m:
        return "1990s"   # nav/index pages → most recent bucket
    year = int(m.group(1))
    if year < 1970:
        return "1960s"
    if year < 1980:
        return "1970s"
    if year < 1990:
        return "1980s"
    return "1990s"


def build(client, dry_run: bool) -> dict[str, int]:
    try:
        src = client.get_collection(SOURCE_COLLECTION)
    except Exception:
        print(f"ERROR: source collection '{SOURCE_COLLECTION}' not found.", file=sys.stderr)
        sys.exit(1)

    total = src.count()
    print(f"Source: {SOURCE_COLLECTION}  ({total:,} chunks)\n")

    counts = {d: 0 for d in DECADES}

    if not dry_run:
        targets = {
            d: client.get_or_create_collection(
                f"{TARGET_PREFIX}_{d}", metadata={"hnsw:space": "cosine"}
            )
            for d in DECADES
        }
        buffers = {
            d: {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
            for d in DECADES
        }

    offset = 0
    fetched = 0
    while offset < total:
        result = src.get(
            limit=FETCH_BATCH,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        if not result["ids"]:
            break

        for chunk_id, emb, doc, meta in zip(
            result["ids"], result["embeddings"],
            result["documents"], result["metadatas"],
        ):
            decade = classify((meta or {}).get("source", ""))
            counts[decade] += 1

            if not dry_run:
                buf = buffers[decade]
                buf["ids"].append(chunk_id)
                buf["embeddings"].append(emb)
                buf["documents"].append(doc)
                buf["metadatas"].append(meta)

                if len(buf["ids"]) >= 500:
                    targets[decade].upsert(**buf)
                    buf["ids"].clear(); buf["embeddings"].clear()
                    buf["documents"].clear(); buf["metadatas"].clear()

        fetched += len(result["ids"])
        offset += len(result["ids"])
        print(f"\r  {fetched:,} / {total:,}", end="", flush=True)

    print()

    if not dry_run:
        for d, buf in buffers.items():
            if buf["ids"]:
                targets[d].upsert(**buf)

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Split survivorlibrary_amateur_radio into decade sub-collections."
    )
    parser.add_argument("--db", default=str(Path(__file__).parent / "vector_db"),
                        help="ChromaDB path (default: ./vector_db)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count only; no writes")
    parser.add_argument("--drop-old", action="store_true",
                        help="Delete the source collection after successful build")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.db)

    existing = {c.name for c in client.list_collections()}
    existing_targets = [n for n in _TARGET_NAMES if n in existing]
    if existing_targets and not args.dry_run:
        print("NOTE: target collections already exist; chunks will be upserted (safe to rerun):")
        for n in sorted(existing_targets):
            print(f"  {n}")
        print()

    mode = "DRY RUN" if args.dry_run else "LIVE BUILD"
    print(f"Mode: {mode}")
    print(f"DB:   {args.db}\n")

    counts = build(client, args.dry_run)

    total = sum(counts.values())
    print(f"\n{'='*60}")
    print(f"{'Decade':<12}  {'Chunks':>8}  {'%':>5}  Target collection")
    print(f"{'-'*60}")
    for d in DECADES:
        n = counts[d]
        pct = 100 * n / total if total else 0
        print(f"  {d:<10}  {n:>8,}  {pct:>4.1f}%  {TARGET_PREFIX}_{d}")
    print(f"  {'TOTAL':<10}  {total:>8,}")

    if args.dry_run:
        print("\n[dry-run] No collections written.")
        return

    print("\nVerifying...")
    ok = True
    for d in DECADES:
        col = client.get_collection(f"{TARGET_PREFIX}_{d}")
        actual = col.count()
        status = "✓" if actual == counts[d] else f"MISMATCH (expected {counts[d]})"
        print(f"  {TARGET_PREFIX}_{d}: {actual:,}  {status}")
        if actual != counts[d]:
            ok = False

    if ok and args.drop_old:
        client.delete_collection(SOURCE_COLLECTION)
        print(f"\nDropped {SOURCE_COLLECTION}")
    elif not ok:
        print("\nWARNING: count mismatches — source NOT deleted.")


if __name__ == "__main__":
    main()
