#!/usr/bin/env python3
"""
RAG query interface — ask questions answered from your Kiwix vector index.

Usage:
    python rag.py                          # interactive chat mode
    python rag.py "how do I undo a commit" # single question and exit
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # use cached model, no network check on startup

import chromadb
import requests
from sentence_transformers import SentenceTransformer

# ── defaults (override with CLI flags) ────────────────────────────────────────
DB_PATH = Path(__file__).parent / "vector_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "phi3:mini"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "context passages provided. If the context does not contain enough "
    "information to answer, say so. Be concise and accurate."
)


def retrieve(query: str, collections: list, embedder, k: int) -> list[dict]:
    vec = embedder.encode([query]).tolist()
    candidates = []
    for collection in collections:
        results = collection.query(query_embeddings=vec, n_results=k, include=["documents", "metadatas", "distances"])
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            candidates.append({"text": doc, "source": meta["source"], "title": meta["title"], "dist": dist})

    # sort by distance (lower = more similar), deduplicate, take top-k
    candidates.sort(key=lambda c: c["dist"])
    seen = set()
    chunks = []
    for c in candidates:
        key = (c["source"], c["text"][:80])
        if key not in seen:
            seen.add(key)
            chunks.append(c)
        if len(chunks) >= k:
            break
    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[{c['title']}]\n{c['text']}" for c in chunks
    )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def stream_answer(prompt: str, model: str, ollama_url: str) -> str:
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": True,
    }
    full = []
    try:
        with requests.post(f"{ollama_url}/api/generate", json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                full.append(token)
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        print("\n[Error] Could not reach Ollama. Is the server running?")
        print(f"  Start it with: ollama serve")
        sys.exit(1)
    print()
    return "".join(full)


def print_sources(chunks: list[dict]) -> None:
    seen = []
    for c in chunks:
        entry = f"  {c['title']} ({c['source']})"
        if entry not in seen:
            seen.append(entry)
    print("\nSources:")
    for s in seen:
        print(s)


def ask(question: str, collections: list, embedder, args) -> None:
    chunks = retrieve(question, collections, embedder, args.top_k)
    if not chunks:
        print("No relevant content found in the index.")
        return
    prompt = build_prompt(question, chunks)
    print()
    stream_answer(prompt, args.model, args.ollama_url)
    print_sources(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ask questions answered from your Kiwix index.")
    parser.add_argument("question", nargs="?", help="Question to answer (omit for interactive mode)")
    parser.add_argument("--db", default=str(DB_PATH), help=f"ChromaDB path (default: {DB_PATH})")
    parser.add_argument("--collection", "-c", action="append", dest="collections",
                        metavar="NAME", help="Collection(s) to search (default: all). Repeat to search multiple.")
    parser.add_argument("--model", "-m", default=LLM_MODEL, help=f"Ollama model (default: {LLM_MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help=f"Ollama base URL (default: {OLLAMA_URL})")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Chunks to retrieve per query (default: {TOP_K})")
    args = parser.parse_args()

    print("Loading embedding model...", end=" ", flush=True)
    embedder = SentenceTransformer(EMBED_MODEL)
    print("ready")

    client = chromadb.PersistentClient(path=str(Path(args.db).expanduser()))
    available = [c.name for c in client.list_collections()]
    if not available:
        print("No collections found. Run build_index.py first.")
        sys.exit(1)

    if args.collections:
        missing = [n for n in args.collections if n not in available]
        if missing:
            print(f"Error: collection(s) not found: {', '.join(missing)}")
            print(f"Available: {', '.join(available)}")
            sys.exit(1)
        names = args.collections
    else:
        names = available  # search everything

    collections = [client.get_collection(n) for n in names]
    total_vectors = sum(c.count() for c in collections)
    print(f"Collections: {', '.join(names)}")
    print(f"Total vectors: {total_vectors:,} | Model: {args.model}\n")

    if args.question:
        ask(args.question, collections, embedder, args)
        return

    # interactive mode
    print("Ask questions about your Kiwix content. Type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        ask(question, collections, embedder, args)
        print()


if __name__ == "__main__":
    main()
