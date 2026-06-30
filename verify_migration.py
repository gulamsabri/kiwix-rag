#!/usr/bin/env python3
"""Verify ChromaDB → pgvector migration: per-collection counts + spot-check vectors.

Usage:
    python verify_migration.py --chroma-path ./vector_db --dsn postgresql:///kiwix_rag
"""

import argparse
import random
import sys
from pathlib import Path

import chromadb

from pg_client import PgClient

SPOT_CHECK = 10


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ChromaDB → pgvector migration.")
    parser.add_argument("--chroma-path", required=True)
    parser.add_argument("--dsn", required=True)
    args = parser.parse_args()

    chroma_path = Path(args.chroma_path).expanduser().resolve()
    cclient = chromadb.PersistentClient(path=str(chroma_path))
    pg = PgClient(args.dsn)
    exit_code = 0

    try:
        chroma_names = {c.name for c in cclient.list_collections()}
        pg_names = set(pg.list_collections())
        only_chroma = chroma_names - pg_names
        only_pg = pg_names - chroma_names
        if only_chroma:
            print(f"FAIL: in ChromaDB but not pgvector: {sorted(only_chroma)}", file=sys.stderr)
            exit_code = 1
        if only_pg:
            print(f"WARN: in pgvector but not ChromaDB: {sorted(only_pg)}", file=sys.stderr)

        for name in sorted(chroma_names & pg_names):
            ccol = cclient.get_collection(name)
            ccount = ccol.count()
            pcount = pg.count(name)
            if ccount != pcount:
                print(f"FAIL [{name}]: chroma={ccount:,} pg={pcount:,} (Δ={ccount - pcount:+,})",
                      file=sys.stderr)
                exit_code = 1
                continue
            # Spot check: pull SPOT_CHECK random ids, compare vectors + metadata
            total = ccount
            if total == 0:
                print(f"OK   [{name}] empty (0 vectors)")
                continue
            sample_n = min(SPOT_CHECK, total)
            offsets = sorted(random.sample(range(total), sample_n))
            for off in offsets:
                cbatch = ccol.get(include=["embeddings", "documents", "metadatas"],
                                 limit=1, offset=off)
                cid = cbatch["ids"][0]
                cemb = cbatch["embeddings"][0]
                cdoc = cbatch["documents"][0]
                cmeta = cbatch["metadatas"][0]
                # Fetch the same id from pgvector
                with pg._pool.connection() as conn:
                    row = conn.execute(
                        "SELECT document, source, title, is_accepted, embedding "
                        "FROM chunks WHERE collection = %s AND id = %s",
                        (name, str(cid)),
                    ).fetchone()
                if row is None:
                    print(f"FAIL [{name}] id={cid} missing from pgvector", file=sys.stderr)
                    exit_code = 1
                    continue
                if row[0] != cdoc:
                    print(f"FAIL [{name}] id={cid} document mismatch", file=sys.stderr)
                    exit_code = 1
                if row[1] != cmeta.get("source", ""):
                    print(f"FAIL [{name}] id={cid} source mismatch", file=sys.stderr)
                    exit_code = 1
                if row[2] != cmeta.get("title", ""):
                    print(f"FAIL [{name}] id={cid} title mismatch", file=sys.stderr)
                    exit_code = 1
                # Compare embeddings byte-for-byte (both float32)
                import numpy as np
                pvec = np.asarray(row[4])
                cvec = np.asarray(cemb, dtype=np.float32)
                if pvec.shape != cvec.shape or not np.array_equal(pvec, cvec):
                    print(f"FAIL [{name}] id={cid} embedding mismatch", file=sys.stderr)
                    exit_code = 1
            print(f"OK   [{name}] {ccount:,} vectors, {sample_n} spot-checked")

        if exit_code == 0:
            print("\nAll collections verified. ✓", flush=True)
        else:
            print("\nVerification FAILED — see above.", file=sys.stderr)
        return exit_code
    finally:
        pg.close()


if __name__ == "__main__":
    sys.exit(main())
