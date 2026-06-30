#!/usr/bin/env python3
"""One-time offline migration: ChromaDB → Postgres + pgvector.

Reads vectors via the ChromaDB Python API (NOT the backing SQLite — vectors
live in HNSW binary segment files and are only readable through the API),
streams them in batches to PgClient.upsert, marks each collection imported in
collections_registry. Resumable: a collection is skipped only when its
`imported_at` is set (i.e. the previous run completed fully). A crash
mid-collection leaves `imported_at = NULL`, so the rerun re-imports from
offset 0 — safe because upsert is idempotent (ON CONFLICT DO UPDATE).

Usage:
    python migrate_chroma_to_pg.py --chroma-path ./vector_db --dsn postgresql:///kiwix_rag
    python migrate_chroma_to_pg.py ... --rebuild devdocs_en_git_2025_04_chunks
"""

import argparse
import sys
import time
from pathlib import Path

import chromadb

from pg_client import PgClient

BATCH_SIZE = 1000


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate ChromaDB vectors to pgvector.")
    parser.add_argument("--chroma-path", required=True, help="Path to the ChromaDB directory")
    parser.add_argument("--dsn", required=True, help="Postgres DSN, e.g. postgresql:///kiwix_rag")
    parser.add_argument("--rebuild", metavar="NAME", default=None,
                        help="Drop and re-import a single collection (ignores imported_at)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only migrate the first N collections (debug)")
    args = parser.parse_args()

    chroma_path = Path(args.chroma_path).expanduser().resolve()
    if not chroma_path.exists():
        print(f"Error: chroma path not found: {chroma_path}", file=sys.stderr)
        return 1

    print(f"Opening ChromaDB at {chroma_path} ...", flush=True)
    cclient = chromadb.PersistentClient(path=str(chroma_path))
    collections = [c.name for c in cclient.list_collections()]
    print(f"  {len(collections)} collections in ChromaDB", flush=True)

    if args.limit:
        collections = collections[: args.limit]

    pg = PgClient(args.dsn)
    try:
        for name in collections:
            if args.rebuild and args.rebuild != name:
                continue
            col = cclient.get_collection(name)
            if args.rebuild == name:
                print(f"[{name}] --rebuild: dropping existing partition", flush=True)
                pg.delete_collection(name)
                pg.create_collection(name)
            elif pg.is_imported(name):
                existing_count = pg.count(name)
                print(f"[{name}] already imported ({existing_count:,} vectors) — skip", flush=True)
                continue
            elif pg.count(name) > 0:
                print(f"[{name}] partial import ({pg.count(name):,} vectors, imported_at NULL) — re-importing from start", flush=True)
                pg.create_collection(name)
            else:
                pg.create_collection(name)

            handle = pg.get_collection(name)
            total = col.count()
            print(f"[{name}] {total:,} vectors → pgvector (batch={BATCH_SIZE})", flush=True)
            t0 = time.time()
            migrated = 0
            offset = 0
            while offset < total:
                batch = col.get(
                    include=["embeddings", "documents", "metadatas"],
                    limit=BATCH_SIZE,
                    offset=offset,
                )
                if not batch["ids"]:
                    break
                metas = [
                    {
                        "source": m.get("source", ""),
                        "title": m.get("title", ""),
                        "is_accepted": m.get("is_accepted", False),
                    }
                    for m in batch["metadatas"]
                ]
                handle.upsert(
                    [str(i) for i in batch["ids"]],
                    [e.tolist() for e in batch["embeddings"]],
                    batch["documents"],
                    metas,
                )
                migrated += len(batch["ids"])
                offset += BATCH_SIZE
                rate = migrated / max(1.0, time.time() - t0)
                print(f"\r  {migrated:,}/{total:,} ({rate:.0f}/s)", end="", flush=True)
            elapsed = time.time() - t0
            print(f"\n  done in {elapsed:.0f}s — {migrated:,} vectors", flush=True)

            # Mark imported
            with pg._pool.connection() as conn:
                conn.execute(
                    "UPDATE collections_registry SET imported_at = now(), vector_count = %s "
                    "WHERE collection = %s",
                    (migrated, name),
                )
            print(f"  marked imported_at", flush=True)

        print("\nMigration complete.", flush=True)
        return 0
    finally:
        pg.close()


if __name__ == "__main__":
    sys.exit(main())
