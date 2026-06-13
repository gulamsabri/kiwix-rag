#!/usr/bin/env python3
"""
CLI entry points for kiwix-rag.

Installed as:
  kiwix-extract → extract_main()
  kiwix-index   → index_main()
  kiwix-query   → query_main()
  kiwix-serve   → serve_main()
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


# ── kiwix-extract ─────────────────────────────────────────────────────────────

def extract_main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text chunks from a Kiwix .zim file for RAG ingestion."
    )
    parser.add_argument("zim_file", help="Path to the .zim file")
    parser.add_argument("--output", "-o", help="Output .jsonl file")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--ocr-engine", default="tesseract", choices=["tesseract", "easyocr"])
    parser.add_argument("--entry-offset", type=int, default=0)
    parser.add_argument("--entry-limit", type=int, default=0)
    parser.add_argument("--filter", action="store_true", dest="quality_filter")
    args = parser.parse_args()

    import libzim
    from kiwix_rag.extract import ZimExtractor
    from kiwix_rag.filter import ChunkFilter

    zim_path = Path(args.zim_file).expanduser().resolve()
    if not zim_path.exists():
        print(f"Error: file not found: {zim_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    elif args.entry_offset > 0 or args.entry_limit > 0:
        output_path = zim_path.parent / f"{zim_path.stem}_e{args.entry_offset:08d}_chunks.jsonl"
    else:
        output_path = zim_path.parent / f"{zim_path.stem}_chunks.jsonl"

    ocr_engine = None
    if args.ocr:
        from kiwix_rag.ocr import load_engine
        try:
            ocr_engine = load_engine(args.ocr_engine)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    quality_filter = ChunkFilter() if args.quality_filter else None
    extractor = ZimExtractor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        ocr_engine=ocr_engine,
        quality_filter=quality_filter,
    )

    archive = libzim.Archive(zim_path)
    print(f"Input:  {zim_path}")
    print(f"Output: {output_path}")
    print(f"Entries in archive: {archive.all_entry_count:,}")
    print()

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for chunk in extractor.iter_chunks(
            archive,
            entry_offset=args.entry_offset,
            entry_limit=args.entry_limit,
        ):
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1

    print(f"\nDone — {count:,} chunks written to {output_path}")


# ── kiwix-index ───────────────────────────────────────────────────────────────

def index_main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed .jsonl chunks into a persistent ChromaDB vector index."
    )
    parser.add_argument("jsonl_file", help="Path to the .jsonl chunks file")
    parser.add_argument("--db", "-d", default=None,
                        help="ChromaDB directory (default: from config.yaml or ./vector_db)")
    parser.add_argument("--collection", "-c", default=None,
                        help="Collection name (default: derived from filename)")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="Path to config.yaml (default: auto-discover)")
    args = parser.parse_args()

    from kiwix_rag.config import Config
    from kiwix_rag.index import Indexer

    cfg = Config.load(Path(args.config) if args.config else None)
    db_path = Path(args.db).expanduser() if args.db else cfg.db_path

    jsonl_path = Path(args.jsonl_file).expanduser().resolve()
    if not jsonl_path.exists():
        print(f"Error: file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Input:      {jsonl_path}")
    print(f"Database:   {db_path}")
    print(f"Model:      {cfg.embed_model}")
    print()

    idx = Indexer(db_path, embed_model=cfg.embed_model)
    try:
        total = idx.build(jsonl_path, collection_name=args.collection, replace=args.replace)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Done — {total:,} vectors indexed to {db_path}")


# ── kiwix-query ───────────────────────────────────────────────────────────────

def query_main() -> None:
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    parser = argparse.ArgumentParser(description="Ask questions from your Kiwix index.")
    parser.add_argument("question", nargs="?", help="Question (omit for interactive mode)")
    parser.add_argument("--db", default=None)
    parser.add_argument("--collection", "-c", action="append", dest="collections", metavar="NAME")
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--config", default=None, metavar="FILE")
    args = parser.parse_args()

    import requests
    import chromadb
    from kiwix_rag.config import Config
    from kiwix_rag.retrieval import Retriever, build_prompt
    from kiwix_rag.groups import SYSTEM_PROMPT

    cfg_overrides = {k: v for k, v in [
        ("db_path", args.db), ("llm_model", args.model),
        ("ollama_url", args.ollama_url), ("top_k", args.top_k),
    ] if v is not None}
    cfg = Config.load(Path(args.config) if args.config else None, **cfg_overrides)

    print("Loading embedding model...", end=" ", flush=True)
    retriever = Retriever(cfg.db_path, cfg.embed_model)
    print("ready")

    available = [c.name for c in retriever.client.list_collections()]
    if not available:
        print("No collections found. Run kiwix-index first.")
        sys.exit(1)

    if args.collections:
        missing = [n for n in args.collections if n not in available]
        if missing:
            print(f"Error: collection(s) not found: {', '.join(missing)}")
            sys.exit(1)
        names = args.collections
    else:
        names = available

    collections = [retriever.client.get_collection(n) for n in names]
    print(f"Collections: {', '.join(names)}")
    print(f"Model: {cfg.llm_model}\n")

    def ask(question: str) -> None:
        chunks = retriever.retrieve(question, collections, k=cfg.top_k)
        if not chunks:
            print("No relevant content found.")
            return
        payload = {
            "model": cfg.llm_model,
            "system": SYSTEM_PROMPT,
            "prompt": build_prompt(question, chunks),
            "stream": True,
        }
        print()
        try:
            with requests.post(f"{cfg.ollama_url}/api/generate",
                               json=payload, stream=True, timeout=cfg.timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    tok = json.loads(line)
                    print(tok.get("response", ""), end="", flush=True)
                    if tok.get("done"):
                        break
        except requests.exceptions.ConnectionError:
            print("\n[Error] Could not reach Ollama. Start it with: ollama serve")
            return
        except requests.exceptions.ReadTimeout:
            print("\n[Error] Ollama request timed out.")
            return
        print()
        seen, sources = [], []
        for c in chunks:
            e = f"  {c['title']} ({c['source']})"
            if e not in seen:
                seen.append(e)
                sources.append(e)
        print("\nSources:")
        for s in sources:
            print(s)

    if args.question:
        ask(args.question)
        return

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
            break
        ask(question)
        print()


# ── kiwix-serve ───────────────────────────────────────────────────────────────

def serve_main() -> None:
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    parser = argparse.ArgumentParser(description="Kiwix RAG web interface.")
    parser.add_argument("--db", default=None)
    parser.add_argument("--collection", "-c", action="append", dest="collections", metavar="NAME")
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-groups", type=int, default=None)
    parser.add_argument("--route-threshold", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-per-group", type=int, default=None)
    parser.add_argument("--max-cache-size", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default=None, metavar="FILE")
    args = parser.parse_args()

    from kiwix_rag.config import Config
    from kiwix_rag.server import create_app

    cfg_overrides = {k: v for k, v in [
        ("db_path", args.db), ("llm_model", args.model),
        ("embed_model", args.embed_model), ("ollama_url", args.ollama_url),
        ("top_k", args.top_k), ("top_groups", args.top_groups),
        ("route_threshold", args.route_threshold), ("timeout", args.timeout),
        ("max_per_group", args.max_per_group), ("max_cache_size", args.max_cache_size),
        ("host", args.host), ("port", args.port),
    ] if v is not None}
    cfg = Config.load(Path(args.config) if args.config else None, **cfg_overrides)

    app = create_app(cfg)
    print(f"Model: {cfg.llm_model} | top_k={cfg.top_k}")
    print(f"Listening on http://{cfg.host}:{cfg.port}\n")
    app.run(host=cfg.host, port=cfg.port, threaded=True)
