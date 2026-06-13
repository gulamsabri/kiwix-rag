from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import chromadb
from sentence_transformers import SentenceTransformer


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(f"[{c['title']}]\n{c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


class Retriever:
    """Query a set of ChromaDB collections and return ranked chunks."""

    ACCEPTED_BOOST = 0.85  # multiply distance by this for accepted answers

    def __init__(
        self,
        db_path: Path | str,
        embed_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.db_path = Path(db_path)
        self._embedder = SentenceTransformer(embed_model)
        self._client = chromadb.PersistentClient(path=str(self.db_path))

    @property
    def embedder(self) -> SentenceTransformer:
        return self._embedder

    @property
    def client(self) -> chromadb.PersistentClient:
        return self._client

    def retrieve(self, query: str, collections: list, k: int = 5) -> list[dict]:
        """
        Query each collection, merge results, apply accepted-answer boost,
        deduplicate, sort by effective distance, and return top-k chunks.
        """
        vec = self._embedder.encode([query]).tolist()
        candidates: list[dict] = []

        for col in collections:
            try:
                results = col.query(
                    query_embeddings=vec,
                    n_results=k,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                continue
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                candidates.append({
                    "text": doc,
                    "source": meta["source"],
                    "title": meta["title"],
                    "dist": dist,
                    "is_accepted": meta.get("is_accepted", False),
                    "zim": col.name.removesuffix("_chunks"),
                })

        for c in candidates:
            if c.get("is_accepted"):
                c["dist"] *= self.ACCEPTED_BOOST

        candidates.sort(key=lambda c: c["dist"])
        seen: set[tuple] = set()
        chunks: list[dict] = []
        for c in candidates:
            key = (c["source"], c["text"][:80])
            if key not in seen:
                seen.add(key)
                chunks.append(c)
            if len(chunks) >= k:
                break
        return chunks
