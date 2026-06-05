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

os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"  # 80MB, fast, runs well on Pi 5
BATCH_SIZE = 256
COPY_BATCH = 1000  # batch size when swapping temp → final collection


def iter_chunks(jsonl_path: Path):
    """Stream chunks from a JSONL file one at a time."""
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(jsonl_path: Path) -> int:
    with open(jsonl_path, "rb") as f:
        return sum(1 for line in f if line.strip())


def swap_collection(client, build_name: str, final_name: str, total: int) -> None:
    """
    Promote build_name → final_name without re-embedding.
    ChromaDB has no rename, so we copy vectors in batches then delete the temp.
    """
    existing = [c.name for c in client.list_collections()]
    if final_name in existing:
        client.delete_collection(final_name)

    src = client.get_collection(build_name)
    dst = client.get_or_create_collection(final_name, metadata={"hnsw:space": "cosine"})

    print(f"Promoting temp collection → '{final_name}'...")
    offset = 0
    while True:
        result = src.get(
            limit=COPY_BATCH,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        if not result["ids"]:
            break
        dst.add(
            ids=result["ids"],
            embeddings=result["embeddings"],
            documents=result["documents"],
            metadatas=result["metadatas"],
        )
        offset += len(result["ids"])
        if offset < total:
            print(f"\r  {offset:,} / {total:,} copied", end="", flush=True)

    client.delete_collection(build_name)
    print(f"\r  {dst.count():,} vectors promoted.          ")


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
    parser.add_argument(
        "--replace", action="store_true",
        help="Replace an existing collection (default: error if it already exists)",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_file).expanduser().resolve()
    if not jsonl_path.exists():
        print(f"Error: file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    db_path = Path(args.db).expanduser().resolve()
    db_path.mkdir(parents=True, exist_ok=True)

    collection_name = args.collection or jsonl_path.stem.replace("-", "_").replace(".", "_")
    build_name = f"{collection_name}__building"

    print(f"Input:      {jsonl_path}")
    print(f"Database:   {db_path}")
    print(f"Collection: {collection_name}")
    print(f"Model:      {EMBED_MODEL}")
    print()

    client = chromadb.PersistentClient(path=str(db_path))
    existing = [c.name for c in client.list_collections()]

    if collection_name in existing and not args.replace:
        print(f"Error: collection '{collection_name}' already exists.", file=sys.stderr)
        print("Use --replace to overwrite it.", file=sys.stderr)
        sys.exit(1)

    # Clean up any leftover temp collection from a previous interrupted run
    if build_name in existing:
        print(f"Removing leftover temp collection '{build_name}' from a previous interrupted run...")
        client.delete_collection(build_name)

    print("Counting chunks...")
    total = count_lines(jsonl_path)
    print(f"  {total:,} chunks")
    print()

    print(f"Loading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)
    print("  Model ready")
    print()

    # Build into a temp collection so the live collection is untouched until success
    collection = client.get_or_create_collection(
        build_name,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding and indexing {total:,} chunks in batches of {BATCH_SIZE}...")
    done = 0
    batch_texts, batch_meta, batch_ids = [], [], []

    for chunk in iter_chunks(jsonl_path):
        batch_texts.append(chunk["text"])
        batch_meta.append({
            "source": chunk["source"],
            "title": chunk["title"],
            "is_accepted": chunk.get("is_accepted", False),
        })
        batch_ids.append(str(done))
        done += 1

        if len(batch_texts) >= BATCH_SIZE:
            embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )
            print(f"\r  {done:,} / {total:,}", end="", flush=True)
            batch_texts, batch_meta, batch_ids = [], [], []

    if batch_texts:
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        print(f"\r  {done:,} / {total:,}", end="", flush=True)

    print(f"\n\nEmbedding complete — {collection.count():,} vectors in temp collection.")
    print()

    swap_collection(client, build_name, collection_name, total)

    final = client.get_collection(collection_name)
    print(f"Done — {final.count():,} vectors in collection '{collection_name}'")
    print(f"Database saved to {db_path}")


if __name__ == "__main__":
    main()
