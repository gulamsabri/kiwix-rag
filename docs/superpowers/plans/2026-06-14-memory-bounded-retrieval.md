# Memory-Bounded, Leak-Free Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop survivorlibrary content from polluting every answer and prevent OOM, by bounding per-query collection memory with a byte budget, fixing the cache so it never drops the current query's collections, and searching `_other` only as a fallback.

**Architecture:** Replace `CollectionCache`'s count-based eviction with a byte-budgeted cache that protects the current request's working set. A new `CollectionSizer` maps each collection to its on-disk index size (read from `chroma.sqlite3`, sized lazily). `GroupRouter.route()` returns `_other` only on the below-threshold fallback path. Then a verified data cleanup drops 7 redundant leftover collections, and the service is retuned (`--max-cache-bytes`, `--max-per-group 5`).

**Tech Stack:** Python 3.13, ChromaDB, Flask, pytest, sentence-transformers, numpy.

**Spec:** `docs/superpowers/specs/2026-06-14-memory-bounded-retrieval-design.md`

---

## File Structure

- **Create** `kiwix_rag/collection_size.py` — `dir_bytes()` + `CollectionSizer` (name → index bytes).
- **Create** `tests/test_collection_size.py` — sizer unit tests.
- **Create** `tests/test_collection_cache.py` — byte-budget cache unit tests.
- **Modify** `kiwix_rag/config.py` — add `max_cache_bytes` field.
- **Modify** `kiwix_rag/server.py` — rewrite `CollectionCache` (byte budget); wire `CollectionSizer` into `create_app`.
- **Modify** `kiwix_rag/router.py` — `route()` returns `_other` fallback-only.
- **Modify** `kiwix_rag/cli.py` — `serve_main`: replace `--max-cache-size` with `--max-cache-bytes`.
- **Modify** `tests/test_config.py`, `tests/test_router.py` — new tests.
- **Modify** `kiwix-rag.service`, `config.example.yaml` — retune flags/keys.
- **Ops (Tasks 8–9)** — leftover-collection verification + drop; deploy + eval.

Run the whole suite with: `source ~/kiwix-rag/bin/activate && pytest -q` (baseline: 48 passing).

---

## Task 1: Config — add `max_cache_bytes`

**Files:**
- Modify: `kiwix_rag/config.py:25`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
def test_max_cache_bytes_default():
    from kiwix_rag.config import Config
    assert Config().max_cache_bytes == 11_000_000_000


def test_max_cache_bytes_override_coerces_int():
    from kiwix_rag.config import Config
    cfg = Config.load(max_cache_bytes="5000000000")
    assert cfg.max_cache_bytes == 5_000_000_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -k max_cache_bytes -v`
Expected: FAIL (`AttributeError: 'Config' object has no attribute 'max_cache_bytes'` / unexpected keyword).

- [ ] **Step 3: Add the field**

In `kiwix_rag/config.py`, add after the `max_cache_size` line (currently line 25):

```python
    max_cache_size: int = 15
    max_cache_bytes: int = 11_000_000_000
```

(Keep `max_cache_size` for backward compatibility with existing config.yaml/env; it is no longer used by the cache.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -k max_cache_bytes -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add kiwix_rag/config.py tests/test_config.py
git commit -m "feat: add max_cache_bytes config field"
```

---

## Task 2: `CollectionSizer` — collection name → on-disk index bytes

**Files:**
- Create: `kiwix_rag/collection_size.py`
- Test: `tests/test_collection_size.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collection_size.py`:

```python
import sqlite3
from pathlib import Path
from kiwix_rag.collection_size import dir_bytes, CollectionSizer


def test_dir_bytes_sums_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "b.bin").write_bytes(b"y" * 250)
    assert dir_bytes(tmp_path) == 350


def test_dir_bytes_ignores_subdirs_and_missing(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.bin").write_bytes(b"z" * 999)
    assert dir_bytes(tmp_path) == 10
    assert dir_bytes(tmp_path / "does_not_exist") == 0


def _make_chroma_db(root: Path):
    con = sqlite3.connect(str(root / "chroma.sqlite3"))
    con.execute("CREATE TABLE collections (id TEXT, name TEXT)")
    con.execute("CREATE TABLE segments (id TEXT, collection TEXT)")
    con.execute("INSERT INTO collections VALUES ('cid1', 'col_a')")
    con.execute("INSERT INTO segments VALUES ('seg1', 'cid1')")
    con.execute("INSERT INTO segments VALUES ('seg2', 'cid1')")
    con.commit()
    con.close()
    for seg, n in (("seg1", 1000), ("seg2", 500)):
        d = root / seg
        d.mkdir()
        (d / "data.bin").write_bytes(b"x" * n)


def test_sizer_sums_all_segments_for_collection(tmp_path):
    _make_chroma_db(tmp_path)
    sizer = CollectionSizer(tmp_path)
    assert sizer.size("col_a") == 1500


def test_sizer_unknown_collection_is_zero(tmp_path):
    _make_chroma_db(tmp_path)
    assert CollectionSizer(tmp_path).size("missing") == 0


def test_sizer_missing_db_is_zero(tmp_path):
    # No chroma.sqlite3 at all (e.g. test/dev environment)
    assert CollectionSizer(tmp_path).size("anything") == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_collection_size.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'kiwix_rag.collection_size'`).

- [ ] **Step 3: Implement the module**

Create `kiwix_rag/collection_size.py`:

```python
from __future__ import annotations
import sqlite3
from pathlib import Path


def dir_bytes(path: Path) -> int:
    """Total size in bytes of files directly under `path` (non-recursive)."""
    if not path.is_dir():
        return 0
    total = 0
    for child in path.iterdir():
        if child.is_file():
            total += child.stat().st_size
    return total


class CollectionSizer:
    """Map a ChromaDB collection name to its on-disk index size in bytes.

    Reads the collection->segment mapping from chroma.sqlite3 once, then sizes
    each collection's segment directories lazily on first request and caches
    the result. Robust to a missing/incomplete DB (returns 0).
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._seg_map = self._load_segment_map()
        self._cache: dict[str, int] = {}

    def _load_segment_map(self) -> dict[str, list[str]]:
        sqlite_path = self._db_path / "chroma.sqlite3"
        if not sqlite_path.exists():
            return {}
        try:
            con = sqlite3.connect(str(sqlite_path))
            rows = con.execute(
                "SELECT c.name, s.id FROM segments s "
                "JOIN collections c ON s.collection = c.id"
            ).fetchall()
            con.close()
        except sqlite3.Error:
            return {}
        mapping: dict[str, list[str]] = {}
        for name, seg_id in rows:
            mapping.setdefault(name, []).append(seg_id)
        return mapping

    def size(self, name: str) -> int:
        if name in self._cache:
            return self._cache[name]
        total = sum(
            dir_bytes(self._db_path / seg) for seg in self._seg_map.get(name, [])
        )
        self._cache[name] = total
        return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_collection_size.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add kiwix_rag/collection_size.py tests/test_collection_size.py
git commit -m "feat: add CollectionSizer for on-disk collection index sizes"
```

---

## Task 3: Byte-budgeted `CollectionCache`

**Files:**
- Modify: `kiwix_rag/server.py:21-53` (the `CollectionCache` class)
- Test: `tests/test_collection_cache.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collection_cache.py`:

```python
from unittest.mock import MagicMock
from kiwix_rag.server import CollectionCache


def make_cache(sizes, max_bytes):
    client = MagicMock()
    client.get_collection.side_effect = lambda n: f"COL:{n}"
    return CollectionCache(client, max_bytes=max_bytes, size_fn=lambda n: sizes[n])


def test_loads_all_when_within_budget():
    cache = make_cache({"a": 2, "b": 3}, max_bytes=10)
    got = cache.get(["a", "b"])
    assert got == {"a": "COL:a", "b": "COL:b"}


def test_skips_collections_over_budget_keeps_working_set():
    # Budget fits 2 of these 3 same-call collections; the LAST is skipped,
    # the earlier (working-set) ones are NOT evicted.
    cache = make_cache({"a": 5, "b": 5, "c": 5}, max_bytes=10)
    got = cache.get(["a", "b", "c"])
    assert set(got) == {"a", "b"}
    assert "c" not in got


def test_cross_query_evicts_non_current():
    cache = make_cache({"a": 6, "b": 6}, max_bytes=8)
    cache.get(["a"])            # a resident
    got = cache.get(["b"])      # b needs room -> a (non-current) evicted
    assert got == {"b": "COL:b"}
    assert "a" not in cache._cache


def test_single_collection_over_budget_loads_alone():
    cache = make_cache({"big": 99}, max_bytes=10)
    got = cache.get(["big"])
    assert got == {"big": "COL:big"}


def test_resident_never_exceeds_budget_for_multi_query():
    cache = make_cache({"a": 6, "b": 6, "c": 6}, max_bytes=8)
    cache.get(["a"])
    cache.get(["b"])
    cache.get(["c"])
    assert cache._resident_bytes() <= 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_collection_cache.py -v`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'max_bytes'`).

- [ ] **Step 3: Rewrite `CollectionCache`**

In `kiwix_rag/server.py`, replace the entire `CollectionCache` class (lines 21-53) with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_collection_cache.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add kiwix_rag/server.py tests/test_collection_cache.py
git commit -m "feat: byte-budgeted CollectionCache that protects the working set"
```

---

## Task 4: Wire `CollectionSizer` into `create_app`

**Files:**
- Modify: `kiwix_rag/server.py` (imports + line 97 cache construction)
- Test: `tests/test_server.py` (existing suite must still pass)

- [ ] **Step 1: Update the import**

In `kiwix_rag/server.py`, alongside the existing `from kiwix_rag.router import GroupRouter`:

```python
from kiwix_rag.router import GroupRouter
from kiwix_rag.collection_size import CollectionSizer
```

- [ ] **Step 2: Replace the cache construction**

Replace line 97:

```python
    col_cache = CollectionCache(_client, max_size=config.max_cache_size)
```

with:

```python
    sizer = CollectionSizer(config.db_path)
    col_cache = CollectionCache(
        _client, max_bytes=config.max_cache_bytes, size_fn=sizer.size
    )
```

- [ ] **Step 3: Run the existing server tests**

Run: `pytest tests/test_server.py -v`
Expected: PASS (5 passed). `CollectionSizer` returns 0 for the test's non-existent `vector_db`, so every collection "fits" and the mocked client loads normally.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: all pass (baseline 48 + new tests).

- [ ] **Step 5: Commit**

```bash
git add kiwix_rag/server.py
git commit -m "feat: wire CollectionSizer into create_app cache"
```

---

## Task 5: Router — `_other` fallback-only

**Files:**
- Modify: `kiwix_rag/router.py:69-72`
- Test: `tests/test_router.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_router.py`:

```python
def test_confident_route_excludes_other():
    r = GroupRouter(GROUPS, top_groups=2, route_threshold=0.20)
    r.group_cols = {"web": ["devdocs_en_react_chunks"], "_other": ["scifi_chunks"]}
    r._group_embs = {"web": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # dot=1.0 >= threshold
    result = r.route(q)
    assert "web" in result
    assert "_other" not in result


def test_below_threshold_route_includes_other():
    r = GroupRouter(GROUPS, top_groups=2, route_threshold=0.20)
    r.group_cols = {"web": ["devdocs_en_react_chunks"], "_other": ["scifi_chunks"]}
    r._group_embs = {"web": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # dot=0.0 < threshold
    result = r.route(q)
    assert "_other" in result
```

- [ ] **Step 2: Run tests to verify the first fails**

Run: `pytest tests/test_router.py -k "confident_route_excludes_other or below_threshold_route_includes_other" -v`
Expected: `test_confident_route_excludes_other` FAILS (today `_other` is always appended); `test_below_threshold_route_includes_other` passes.

- [ ] **Step 3: Make the change**

In `kiwix_rag/router.py`, replace the confident-branch return (lines 69-72):

```python
        selected = [g for g, s in ranked[: self.top_groups] if s >= best - 0.1]
        if "_other" in self.group_cols:
            selected.append("_other")
        return selected
```

with:

```python
        # Confident match: search only the top groups. _other (unassigned
        # collections) is reserved for the below-threshold fallback above, so
        # it never pollutes correctly-routed queries.
        return [g for g, s in ranked[: self.top_groups] if s >= best - 0.1]
```

(Leave the below-threshold `fallback` branch on lines 63-67 unchanged — it still appends `_other`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router.py -v`
Expected: PASS (all, including the two new tests). Note `test_route_returns_all_when_no_embeddings` still passes (that path returns early at the `if not self._group_embs` guard).

- [ ] **Step 5: Commit**

```bash
git add kiwix_rag/router.py tests/test_router.py
git commit -m "feat: route _other only on below-threshold fallback"
```

---

## Task 6: CLI — replace `--max-cache-size` with `--max-cache-bytes`

**Files:**
- Modify: `kiwix_rag/cli.py:255` and `:269`

- [ ] **Step 1: Replace the argument definition**

In `kiwix_rag/cli.py`, replace line 255:

```python
    parser.add_argument("--max-cache-size", type=int, default=None)
```

with:

```python
    parser.add_argument("--max-cache-bytes", type=int, default=None,
                        help="Max resident collection-index bytes (default ~11 GB)")
```

- [ ] **Step 2: Replace the override mapping**

In the `cfg_overrides` list (line 269), replace:

```python
        ("max_per_group", args.max_per_group), ("max_cache_size", args.max_cache_size),
```

with:

```python
        ("max_per_group", args.max_per_group), ("max_cache_bytes", args.max_cache_bytes),
```

- [ ] **Step 3: Verify the CLI parses**

Run: `python -c "from kiwix_rag.cli import serve_main; import sys; sys.argv=['x','--help']; serve_main()" 2>&1 | grep max-cache`
Expected: shows `--max-cache-bytes` and NOT `--max-cache-size`.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add kiwix_rag/cli.py
git commit -m "feat: serve CLI uses --max-cache-bytes instead of --max-cache-size"
```

---

## Task 7: Retune service file + example config

**Files:**
- Modify: `kiwix-rag.service`
- Modify: `config.example.yaml`

- [ ] **Step 1: Update the service ExecStart**

In `kiwix-rag.service`, change the ExecStart args:
- `--max-cache-size 2` → `--max-cache-bytes 11000000000`
- `--max-per-group 3` → `--max-per-group 5`

Resulting ExecStart tail:

```
    --model llama3.1:8b-rag \
    --max-cache-bytes 11000000000 \
    --max-per-group 5
```

Leave `MemoryMax=13G` and everything else unchanged.

- [ ] **Step 2: Update example config**

In `config.example.yaml`, if a `max_cache_size:` key exists, add beneath it:

```yaml
max_cache_bytes: 11000000000   # ~11 GB resident collection-index budget
```

- [ ] **Step 3: Sanity-check the service file parses the flags**

Run: `grep -E "max-cache-bytes|max-per-group|MemoryMax" kiwix-rag.service`
Expected: shows `--max-cache-bytes 11000000000`, `--max-per-group 5`, `MemoryMax=13G`.

- [ ] **Step 4: Commit**

```bash
git add kiwix-rag.service config.example.yaml
git commit -m "ops: retune service to byte budget (11GB) and max-per-group 5"
```

---

## Task 8: Verify and drop redundant leftover collections (ops — Pi)

> Runs against the live DB as a **standalone script outside the kiwix-rag
> service cgroup**, so `MemoryMax=13G` does NOT protect it — a careless full-DB
> sweep here can trigger a *global* OOM and freeze the Pi (this is what killed
> the earlier `count()` sweep). The script below is safe because it uses ONLY
> metadata operations — `.count()` and `.get(where=...)` hit the sqlite metadata
> segment and do **not** load the multi-GB HNSW vector indexes (only `.query()`
> does, and we never call it). It touches ~12 collections, not all 151.
> Requires the Pi powered on and SSD mounted — confirm with the user first.

The 8 suspects (matched no group, lived in `_other`):
`survivorlibrary_com_en_all_2025_03_e0000{1000,1500,1500__building,2000,2500,3000,3500}_chunks`.

- [ ] **Step 1: Generate a redundancy report**

Save as `scripts/check_leftovers.py` and run on the Pi
(`ssh pi@meshpi.local 'source ~/kiwix-rag/bin/activate && python /mnt/ssd/kiwix-rag-project/scripts/check_leftovers.py'`):

```python
import chromadb

DB = "/mnt/ssd/vector_db"
LEFTOVERS = [
    "survivorlibrary_com_en_all_2025_03_e00001000_chunks",
    "survivorlibrary_com_en_all_2025_03_e00001500_chunks",
    "survivorlibrary_com_en_all_2025_03_e00001500_chunks__building",
    "survivorlibrary_com_en_all_2025_03_e00002000_chunks",
    "survivorlibrary_com_en_all_2025_03_e00002500_chunks",
    "survivorlibrary_com_en_all_2025_03_e00003000_chunks",
    "survivorlibrary_com_en_all_2025_03_e00003500_chunks",
]
TOPIC = [
    "survivorlibrary_reference", "survivorlibrary_engineering",
    "survivorlibrary_medicine", "survivorlibrary_agriculture",
]
client = chromadb.PersistentClient(path=DB)
existing = {c.name for c in client.list_collections()}

for name in LEFTOVERS:
    if name not in existing:
        print(f"{name}: ALREADY GONE")
        continue
    col = client.get_collection(name)
    sample = col.get(limit=5, include=["metadatas"])
    sources = {m.get("source") for m in sample["metadatas"] if m.get("source")}
    print(f"\n{name}: count={col.count():,} sample_sources={sources}")
    for tname in TOPIC:
        if tname not in existing:
            continue
        tcol = client.get_collection(tname)
        for s in sources:
            hit = tcol.get(where={"source": s}, limit=1, include=["metadatas"])
            if hit["ids"]:
                print(f"   redundant: source {s!r} present in {tname}")
                break
```

- [ ] **Step 2: Decide per collection**

Read the report. A leftover is **redundant** if each sampled source is found in a topic collection. Record which are redundant and which (if any) are NOT — the not-redundant ones were never merged and must be resegmented, not dropped.

- [ ] **Step 3: Drop the confirmed-redundant collections (Pi)**

For each confirmed-redundant `NAME`, on the Pi:

```bash
ssh pi@meshpi.local 'source ~/kiwix-rag/bin/activate && python -c "
import chromadb
c = chromadb.PersistentClient(path=\"/mnt/ssd/vector_db\")
for n in [\"NAME1\", \"NAME2\"]:
    c.delete_collection(n); print(\"dropped\", n)
"'
```

- [ ] **Step 4: Drop the same collections on the Mac source DB**

```bash
source ~/kiwix-rag/bin/activate && python -c "
import chromadb
c = chromadb.PersistentClient(path='vector_db')
for n in ['NAME1', 'NAME2']:
    try:
        c.delete_collection(n); print('dropped', n)
    except Exception as e:
        print('skip', n, e)
"
```

- [ ] **Step 5: Resegment any not-yet-merged parts (only if Step 2 found non-redundant leftovers)**

```bash
source ~/kiwix-rag/bin/activate && python resegment_survivorlibrary.py --drop-old > /tmp/resegment_live.log 2>&1 &
```

Watch `/tmp/resegment_live.log` for the per-collection summary and a "dropping source collections" line. If this runs, re-sync the affected collections to the Pi afterward.

- [ ] **Step 6: Commit the helper script**

```bash
git add scripts/check_leftovers.py
git commit -m "ops: add leftover-collection redundancy check script"
```

---

## Task 9: Deploy and verify (ops)

> Requires the Pi on + SSD mounted, and the Ollama host healthy (the eval can't
> pass while `/api/generate` 500s). Confirm both with the user first.

- [ ] **Step 1: Confirm Ollama backend is reachable**

```bash
curl -s -m 10 http://cori-desktop.local:11434/api/tags | head -c 200 || echo "OLLAMA UNREACHABLE"
```

Expected: JSON model list. If unreachable, stop and resolve Ollama before continuing (out of scope for this plan).

- [ ] **Step 2: Sync package + service to the Pi**

With the SSD on the Mac: `bash update_pi.sh --scripts`. With the SSD on the Pi instead, `scp` the package + service over (per the established workflow). Then on the Pi:

```bash
ssh pi@meshpi.local 'source ~/kiwix-rag/bin/activate && pip install /mnt/ssd/kiwix-rag-project/'
```

- [ ] **Step 3: Install the new service file (needs sudo — user runs in a TTY)**

```
ssh -t pi@meshpi.local "sudo cp /mnt/ssd/kiwix-rag-project/kiwix-rag.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl restart kiwix-rag && echo RESTARTED"
```

- [ ] **Step 4: Verify the service came up with the new flags**

```bash
ssh pi@meshpi.local 'systemctl show kiwix-rag -p ExecStart --no-pager | grep -oE "max-cache-bytes [0-9]+|max-per-group [0-9]+"; systemctl is-active kiwix-rag'
```

Expected: `max-cache-bytes 11000000000`, `max-per-group 5`, `active`.

- [ ] **Step 5: Spot-check retrieval routes to the right domain**

Ask the previously-broken questions and confirm sources are correct-domain, not survivorlibrary:

```bash
for q in "How do I use the useEffect hook in React?" "How do I expose a port in Docker?"; do
  echo "Q: $q"
  curl -s -m 320 http://meshpi.local:5000/api/ask -H 'Content-Type: application/json' \
    -d "{\"question\": \"$q\"}" | python3 -c "import sys,json; d=json.load(sys.stdin); print([s['source'] for s in d.get('sources',[])])"
done
```

Expected: React → `devdocs_en_react*`; Docker → `devdocs`/`serverfault` sources; no `survivorlibrary` sources. (To see which group each routed to, `ssh pi@meshpi.local 'journalctl -u kiwix-rag -n 20 --no-pager | grep groups:'` afterward.)

- [ ] **Step 6: Re-run the eval groups (user runs in their own terminal)**

```bash
for g in survivorlibrary_law survivorlibrary_anatomy survivorlibrary_electrical survivorlibrary_mining; do
  python eval.py --url http://meshpi.local:5000 --group "$g"
done
```

Confirm modern-domain questions no longer return survivorlibrary, scores recover, and no OOM (`ssh pi@meshpi.local 'systemctl show kiwix-rag -p NRestarts --value'` stays 0).

- [ ] **Step 7: Final commit / branch ready for PR**

The refactor + this fix are now ready to push as one PR (per the chosen sequencing).

---

## Notes for the implementer

- **Do not** widen `retrieve()` — it already accepts whatever collections the cache returns; the budget is enforced upstream in `CollectionCache.get`.
- The `names_to_load` list in `server.py:_retrieve_for_query` is already in route-ranked, name-relevance order, so the cache loads the highest-priority collections first within budget — no change needed there.
- Keep the Ollama timeout/500 issue out of scope; it is infra, tracked separately.
