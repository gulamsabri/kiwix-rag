#!/usr/bin/env python3
"""
Embed chunks from a .jsonl file and store them in a persistent ChromaDB collection.

Usage:
    python build_index.py <chunks.jsonl> [options]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # use cached model, no network check on startup

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"  # 80MB, fast, runs well on Pi 5
BATCH_SIZE = 256


def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Embed .jsonl chunks into a persistent ChromaDB vector index."
    )
    parser.add_argument("jsonl_file", help="Path to the .jsonl chunks file")
    parser.add_argument(
        "--db", "-d",
        default=str(Path(__file__).parent / "vector_db"),
        help="Directory for the ChromaDB database (default: ./vector_db next to this script)",
    )
    parser.add_argument(
        "--collection", "-c",
        help="Collection name (default: derived from the input filename)",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_file).expanduser().resolve()
    if not jsonl_path.exists():
        print(f"Error: file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    db_path = Path(args.db).expanduser().resolve()
    db_path.mkdir(parents=True, exist_ok=True)

    collection_name = args.collection or jsonl_path.stem.replace("-", "_").replace(".", "_")

    print(f"Input:      {jsonl_path}")
    print(f"Database:   {db_path}")
    print(f"Collection: {collection_name}")
    print(f"Model:      {EMBED_MODEL}")
    print()

    print("Loading chunks...")
    chunks = load_chunks(jsonl_path)
    print(f"  {len(chunks):,} chunks loaded")
    print()

    print(f"Loading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)
    print("  Model ready")
    print()

    client = chromadb.PersistentClient(path=str(db_path))
    # Replace existing collection so re-runs are idempotent
    client.delete_collection(collection_name) if collection_name in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding and indexing {len(chunks):,} chunks in batches of {BATCH_SIZE}...")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(
            ids=[str(i + j) for j in range(len(batch))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": c["source"], "title": c["title"],
                        "is_accepted": c.get("is_accepted", False)} for c in batch],
        )
        done = min(i + BATCH_SIZE, len(chunks))
        print(f"\r  {done:,} / {len(chunks):,}", end="", flush=True)

    print(f"\n\nDone — {collection.count():,} vectors in collection '{collection_name}'")
    print(f"Database saved to {db_path}")


if __name__ == "__main__":
    main()
