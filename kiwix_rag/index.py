from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterator

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import chromadb
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 256
COPY_BATCH = 1000
ID_FETCH_BATCH = 10_000


def iter_chunks(jsonl_path: Path) -> Iterator[dict]:
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(jsonl_path: Path) -> int:
    with open(jsonl_path, "rb") as f:
        return sum(1 for line in f if line.strip())


class Indexer:
    """Embed JSONL chunks and store them in a persistent ChromaDB collection."""

    def __init__(self, db_path: Path | str, embed_model: str = "all-MiniLM-L6-v2") -> None:
        self.db_path = Path(db_path)
        self.embed_model = embed_model
        self._model: SentenceTransformer | None = None
        self._client: chromadb.PersistentClient | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embed_model)
        return self._model

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.db_path))
        return self._client

    def build(
        self,
        jsonl_path: Path | str,
        collection_name: str | None = None,
        replace: bool = False,
    ) -> int:
        """Embed and index a JSONL file. Returns number of chunks indexed."""
        jsonl_path = Path(jsonl_path)
        name = collection_name or jsonl_path.stem.replace("-", "_").replace(".", "_")
        build_name = f"{name}__building"

        existing = [c.name for c in self.client.list_collections()]
        if name in existing and not replace:
            raise RuntimeError(
                f"Collection '{name}' already exists. Pass replace=True to overwrite."
            )

        if build_name in existing:
            self.client.delete_collection(build_name)

        total = count_lines(jsonl_path)
        print(f"Embedding {total:,} chunks → '{name}'...")

        collection = self.client.get_or_create_collection(
            build_name, metadata={"hnsw:space": "cosine"}
        )

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
                raw = self.model.encode(batch_texts, show_progress_bar=False)
                embeddings = raw.tolist() if hasattr(raw, "tolist") else raw
                collection.add(ids=batch_ids, embeddings=embeddings,
                               documents=batch_texts, metadatas=batch_meta)
                print(f"\r  {done:,} / {total:,}", end="", flush=True)
                batch_texts, batch_meta, batch_ids = [], [], []

        if batch_texts:
            raw = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings = raw.tolist() if hasattr(raw, "tolist") else raw
            collection.add(ids=batch_ids, embeddings=embeddings,
                           documents=batch_texts, metadatas=batch_meta)
            print(f"\r  {done:,} / {total:,}", end="", flush=True)

        print(f"\n\nEmbedding complete — {collection.count():,} vectors.")
        self._swap_collection(build_name, name, total)
        return total

    def _iter_ids(self, collection) -> list[str]:
        ids: list[str] = []
        offset = 0
        while True:
            page = collection.get(limit=ID_FETCH_BATCH, offset=offset, include=[])
            if not page["ids"]:
                break
            ids.extend(page["ids"])
            offset += len(page["ids"])
        return ids

    def _copy_collection(self, src_name: str, dst_name: str) -> None:
        src = self.client.get_collection(src_name)
        dst = self.client.get_or_create_collection(dst_name, metadata={"hnsw:space": "cosine"})
        all_ids = self._iter_ids(src)
        for i in range(0, len(all_ids), COPY_BATCH):
            batch = all_ids[i : i + COPY_BATCH]
            result = src.get(ids=batch, include=["embeddings", "documents", "metadatas"])
            dst.add(ids=result["ids"], embeddings=result["embeddings"],
                    documents=result["documents"], metadatas=result["metadatas"])

    def _swap_collection(self, build_name: str, final_name: str, total: int) -> None:
        """Atomic-ish promotion: build_name → final_name with backup."""
        backup_name = f"{final_name}__prev"
        existing = {c.name for c in self.client.list_collections()}

        if backup_name in existing:
            raise RuntimeError(
                f"Found leftover backup '{backup_name}' from an interrupted promotion.\n"
                f"Inspect '{final_name}' and '{backup_name}', delete the bad one, then retry."
            )

        if final_name in existing:
            print(f"Backing up existing '{final_name}' → '{backup_name}'...")
            self._copy_collection(final_name, backup_name)
            self.client.delete_collection(final_name)

        print(f"Promoting temp collection → '{final_name}'...")
        src = self.client.get_collection(build_name)
        dst = self.client.get_or_create_collection(final_name, metadata={"hnsw:space": "cosine"})
        copied = 0
        for i in range(0, total, COPY_BATCH):
            batch_ids = [str(j) for j in range(i, min(i + COPY_BATCH, total))]
            result = src.get(ids=batch_ids, include=["embeddings", "documents", "metadatas"])
            dst.add(ids=result["ids"], embeddings=result["embeddings"],
                    documents=result["documents"], metadatas=result["metadatas"])
            copied += len(result["ids"])
            if copied < total:
                print(f"\r  {copied:,} / {total:,} copied", end="", flush=True)

        final_count = dst.count()
        if final_count != total:
            raise RuntimeError(
                f"Promotion count mismatch: expected {total}, got {final_count}. "
                f"Previous index preserved at '{backup_name}'."
            )

        self.client.delete_collection(build_name)
        if backup_name in {c.name for c in self.client.list_collections()}:
            self.client.delete_collection(backup_name)
        print(f"\r  {final_count:,} vectors promoted.          ")
