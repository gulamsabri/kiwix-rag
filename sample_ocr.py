#!/usr/bin/env python3
"""
Quick OCR quality check for survivorlibrary collections.

Samples chunks from each survivorlibrary_* collection and reports:
  - collection name, chunk count
  - mean chunk length
  - estimated garble rate (non-ASCII + runs of non-word chars)
  - a few randomly sampled chunks to eyeball

Usage:
    python sample_ocr.py                     # uses default vector_db path
    python sample_ocr.py --db /path/to/db
    python sample_ocr.py --show 5            # print 5 sample chunks per collection
"""

import argparse
import random
import re
import unicodedata
from pathlib import Path

import chromadb


def garble_score(text: str) -> float:
    """Fraction of characters that look like OCR noise."""
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127 and unicodedata.category(c) not in ("Ll", "Lu", "Lt", "Lm", "Lo"))
    # runs of 3+ consecutive non-word, non-space characters
    garbage_runs = len(re.findall(r'[^\w\s]{3,}', text))
    return (non_ascii + garbage_runs * 3) / max(len(text), 1)


def main():
    parser = argparse.ArgumentParser(description="Sample OCR quality from survivorlibrary collections.")
    parser.add_argument("--db", default=str(Path(__file__).parent / "vector_db"),
                        help="ChromaDB path (default: ./vector_db)")
    parser.add_argument("--show", type=int, default=2, metavar="N",
                        help="Sample chunks to print per collection (default: 2)")
    parser.add_argument("--sample", type=int, default=200, metavar="N",
                        help="Chunks to fetch for statistics (default: 200)")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.db)
    all_names = [c.name for c in client.list_collections()]
    surv_names = sorted(n for n in all_names if "survivorlibrary" in n)

    if not surv_names:
        print("No survivorlibrary collections found in the database.")
        return

    print(f"Found {len(surv_names)} survivorlibrary collection(s)\n{'=' * 60}")

    total_chunks = 0
    total_garble = 0.0
    collections_checked = 0

    for name in surv_names:
        col = client.get_collection(name)
        count = col.count()
        total_chunks += count

        fetch_n = min(args.sample, count)
        result = col.get(limit=fetch_n, include=["documents"])
        docs = result["documents"]

        lengths = [len(d) for d in docs]
        scores = [garble_score(d) for d in docs]
        mean_len = sum(lengths) / len(lengths) if lengths else 0
        mean_garble = sum(scores) / len(scores) if scores else 0
        high_garble = sum(1 for s in scores if s > 0.05)

        total_garble += mean_garble
        collections_checked += 1

        print(f"\n{name}")
        print(f"  chunks:          {count:,}")
        print(f"  mean length:     {mean_len:.0f} chars")
        print(f"  mean garble:     {mean_garble:.3f}  (>0.05 = suspicious)")
        print(f"  high-garble:     {high_garble}/{fetch_n} chunks ({100*high_garble/(fetch_n or 1):.0f}%)")

        if args.show > 0:
            samples = random.sample(docs, min(args.show, len(docs)))
            for i, chunk in enumerate(samples, 1):
                snippet = chunk.replace("\n", " ").strip()
                print(f"  --- sample {i} ---")
                print(f"  {snippet[:300]}{'...' if len(snippet) > 300 else ''}")

    print(f"\n{'=' * 60}")
    print(f"Total chunks across {len(surv_names)} collection(s): {total_chunks:,}")
    if collections_checked:
        print(f"Mean garble rate across collections: {total_garble/collections_checked:.3f}")


if __name__ == "__main__":
    main()
