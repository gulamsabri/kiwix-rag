from __future__ import annotations
import json
import os
import time
import threading
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import requests
from flask import Flask, Response, render_template, request

from kiwix_rag.config import Config
from kiwix_rag.groups import SYSTEM_PROMPT, GROUPS
from kiwix_rag.router import GroupRouter
from kiwix_rag.collection_size import CollectionSizer
from kiwix_rag.retrieval import Retriever, build_prompt

_GROUP_TTL = 600  # seconds before idle collection evicted from cache


class CollectionCache:
    """Byte-budgeted cache for ChromaDB collection handles.

    Resident memory is bounded by total on-disk index bytes (a proxy for RAM),
    not by count. The current request's working set (the names passed to one
    get() call) is never evicted; only collections from previous queries are.
    A collection that cannot fit is skipped and logged, except a single
    collection larger than the whole budget, which is loaded alone (best effort).
    """

    def __init__(self, client, max_bytes: int, size_fn) -> None:
        self._client = client
        self._max_bytes = max_bytes
        self._size_fn = size_fn
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _resident_bytes(self) -> int:
        return sum(e["bytes"] for e in self._cache.values())

    def get(self, names: list[str]) -> dict:
        now = time.time()
        working = set(names)
        with self._lock:
            for n in names:
                if n in self._cache:
                    self._cache[n]["last_used"] = now
                    continue
                need = self._size_fn(n)
                while self._resident_bytes() + need > self._max_bytes:
                    evictable = [k for k in self._cache if k not in working]
                    if not evictable:
                        break
                    lru = min(evictable, key=lambda k: self._cache[k]["last_used"])
                    del self._cache[lru]
                fits = self._resident_bytes() + need <= self._max_bytes
                if fits or not self._cache:
                    if not fits:
                        print(
                            f"  [cache] best-effort load {n} ({need / 1e9:.1f} GB) "
                            f"— exceeds budget alone",
                            flush=True,
                        )
                    self._cache[n] = {
                        "col": self._client.get_collection(n),
                        "bytes": need,
                        "last_used": now,
                    }
                else:
                    print(
                        f"  [cache] skipped {n} ({need / 1e9:.1f} GB) — over budget",
                        flush=True,
                    )
            return {n: self._cache[n]["col"] for n in names if n in self._cache}

    def evict_stale(self, ttl: float = _GROUP_TTL) -> None:
        now = time.time()
        with self._lock:
            stale = [n for n, e in self._cache.items() if now - e["last_used"] > ttl]
            for n in stale:
                del self._cache[n]
            if stale:
                print(f"  [cache] evicted {len(stale)}: {', '.join(stale)}", flush=True)


def _eviction_daemon(cache: CollectionCache) -> None:
    while True:
        time.sleep(60)
        cache.evict_stale()


def create_app(
    config: Config,
    retriever: Retriever | None = None,
    router: GroupRouter | None = None,
) -> Flask:
    """
    Flask application factory.

    Pass retriever and router for testing (avoids loading real models).
    In production, omit them — create_app will initialize them from config.
    """
    templates_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(templates_dir))

    if retriever is None:
        retriever = Retriever(config.db_path, config.embed_model)
        _client = retriever.client
    else:
        _client = getattr(retriever, "client", None)

    if router is None:
        if _client is None:
            raise ValueError(
                "Cannot auto-build router: retriever._client is None. "
                "Inject a router or pass a full Retriever."
            )
        router = GroupRouter(
            GROUPS,
            top_groups=config.top_groups,
            route_threshold=config.route_threshold,
            max_per_group=config.max_per_group,
        )
        available = [c.name for c in _client.list_collections()]
        router.build(available, retriever.embedder)

    sizer = CollectionSizer(config.db_path)
    col_cache = CollectionCache(
        _client, max_bytes=config.max_cache_bytes, size_fn=sizer.size
    )
    threading.Thread(target=_eviction_daemon, args=(col_cache,), daemon=True).start()

    def _retrieve_for_query(question: str) -> list[dict]:
        q_norm = retriever.embedder.encode([question], normalize_embeddings=True)
        groups = router.route(q_norm[0])
        seen: set[str] = set()
        names_to_load: list[str] = []
        for g in groups:
            selected = router.select_collections(
                router.group_cols.get(g, []), question
            )
            for name in selected:
                if name not in seen:
                    seen.add(name)
                    names_to_load.append(name)
        col_map = col_cache.get(names_to_load)
        cols = [col_map[n] for n in names_to_load if n in col_map]
        print(
            f"  groups: {[g for g in groups if g != '_other']} → {len(cols)} collections",
            flush=True,
        )
        return retriever.retrieve(question, cols, k=config.top_k)

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/ask")
    def ask():
        question = request.args.get("q", "").strip()

        def generate():
            if not question:
                yield "data: [DONE]\n\n"
                return
            print(f"Q: {question}", flush=True)
            chunks = _retrieve_for_query(question)
            if not chunks:
                yield _sse({"token": "No relevant content found in the index."})
                yield "data: [DONE]\n\n"
                return
            payload = {
                "model": config.llm_model,
                "system": SYSTEM_PROMPT,
                "prompt": build_prompt(question, chunks),
                "stream": True,
                "keep_alive": -1,
            }
            try:
                with requests.post(
                    f"{config.ollama_url}/api/generate",
                    json=payload, stream=True, timeout=config.timeout,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield _sse({"token": token})
                        if chunk.get("done"):
                            break
            except requests.exceptions.ConnectionError:
                yield _sse({"token": "\n[Error: could not reach Ollama — is it running?]"})
                yield "data: [DONE]\n\n"
                return
            except requests.exceptions.ReadTimeout:
                yield _sse({"token": "\n[Error: Ollama timed out]"})
                yield "data: [DONE]\n\n"
                return
            except Exception as e:
                yield _sse({"token": f"\n[Error: {e}]"})
                yield "data: [DONE]\n\n"
                return

            seen_sources, sources = [], []
            for c in chunks:
                entry = {"title": c["title"], "source": c["source"], "zim": c.get("zim", "")}
                if entry not in seen_sources:
                    seen_sources.append(entry)
                    sources.append(entry)
            yield _sse({"sources": sources})
            yield "data: [DONE]\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/ask", methods=["POST"])
    def api_ask():
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return {"error": "no question provided"}, 400
        t0 = time.time()
        print(f"Q (api): {question}", flush=True)
        chunks = _retrieve_for_query(question)
        if not chunks:
            return {
                "answer": "", "sources": [], "elapsed": round(time.time() - t0, 1),
                "error": "no relevant content found",
            }
        payload = {
            "model": config.llm_model,
            "system": SYSTEM_PROMPT,
            "prompt": build_prompt(question, chunks),
            "stream": False,
            "keep_alive": -1,
        }
        try:
            resp = requests.post(
                f"{config.ollama_url}/api/generate",
                json=payload, timeout=config.timeout,
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "")
        except Exception as e:
            return {"error": str(e)}, 500

        seen_sources, sources = [], []
        for c in chunks:
            entry = {"title": c["title"], "source": c["source"]}
            if entry not in seen_sources:
                seen_sources.append(entry)
                sources.append(entry)
        return {"answer": answer, "sources": sources, "elapsed": round(time.time() - t0, 1)}

    return app
