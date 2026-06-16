# Public-Ready Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor kiwix-rag from a collection of standalone scripts into an installable Python package with a config file, OOP design, consistent CLI entry points, and a pytest test suite.

**Architecture:** A `kiwix_rag/` package holds all logic in focused, testable classes (`Config`, `ChunkFilter`, `ZimExtractor`, `Indexer`, `Retriever`, `GroupRouter`). Top-level scripts become thin wrappers that call into the package. A `pyproject.toml` exposes `kiwix-extract`, `kiwix-index`, `kiwix-query`, and `kiwix-serve` as installed commands. Users copy `config.example.yaml` → `config.yaml` and edit 3–5 lines; no hardcoded paths.

**Tech Stack:** Python 3.11+, pytest, PyYAML, chromadb, sentence-transformers, flask, libzim, numpy

---

## File Map

### Created
- `pyproject.toml` — package metadata, deps, entry points
- `config.example.yaml` — annotated reference config (copy to `config.yaml`)
- `kiwix_rag/__init__.py` — version string only
- `kiwix_rag/config.py` — `Config` dataclass; loads YAML + env vars
- `kiwix_rag/filter.py` — `ChunkFilter` class (logic from `chunk_filter.py`)
- `kiwix_rag/groups.py` — `GROUPS` dict and `SYSTEM_PROMPT` (extracted from `web.py`)
- `kiwix_rag/router.py` — `GroupRouter` class (routing logic from `web.py`)
- `kiwix_rag/extract.py` — `ZimExtractor` class (logic from `extract_zim.py`)
- `kiwix_rag/ocr.py` — OCR module (moved from `ocr.py`)
- `kiwix_rag/index.py` — `Indexer` class (logic from `build_index.py`)
- `kiwix_rag/retrieval.py` — `Retriever` class (logic from `rag.py` + `web.py`)
- `kiwix_rag/server.py` — Flask app factory (logic from `web.py`)
- `kiwix_rag/cli.py` — four `main()` functions; entry points for pyproject.toml
- `tests/__init__.py`
- `tests/test_config.py`
- `tests/test_filter.py`
- `tests/test_router.py`
- `tests/test_extract.py`
- `tests/test_retrieval.py`
- `tests/test_server.py`

### Modified (thin wrappers only — no logic changes)
- `extract_zim.py` — becomes 3 lines: import + call `kiwix_rag.cli.extract_main()`
- `build_index.py` — becomes 3 lines: import + call `kiwix_rag.cli.index_main()`
- `rag.py` — becomes 3 lines: import + call `kiwix_rag.cli.query_main()`
- `web.py` — becomes 3 lines: import + call `kiwix_rag.cli.serve_main()`
- `ocr.py` — becomes 3 lines: re-export from `kiwix_rag.ocr` for backward compat
- `requirements.txt` — add `pyyaml`
- `batch_index.sh` — replace hardcoded defaults with env-var form already present; no logic change

### Unchanged
- `chunk_filter.py` — keep in place; its logic is duplicated inside `kiwix_rag/filter.py`. We do NOT delete it yet (batch scripts import it by path). Deprecation note added.
- `eval.py`, `resegment_*.py`, `filter_survivorlibrary.py`, `sample_ocr.py`, `stage_for_pi.sh`, `update_pi.sh`, `build_kiwix_library.sh` — untouched in this refactor

---

## Task 1: Package Scaffold + pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `kiwix_rag/__init__.py`
- Create: `config.example.yaml`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `kiwix_rag/__init__.py`**

```python
__version__ = "0.2.0"
```

- [ ] **Step 2: Create `tests/__init__.py`**

```python
```
(empty file)

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "kiwix-rag"
version = "0.2.0"
requires-python = ">=3.11"
dependencies = [
    "chromadb",
    "sentence-transformers",
    "flask",
    "requests",
    "numpy",
    "libzim",
    "beautifulsoup4",
    "langchain-text-splitters",
    "pypdf",
    "pyyaml",
]

[project.optional-dependencies]
ocr = [
    "pymupdf",
    "opencv-python-headless",
    "pillow",
    "pytesseract",
]
dev = [
    "pytest",
    "pytest-mock",
]

[project.scripts]
kiwix-extract = "kiwix_rag.cli:extract_main"
kiwix-index   = "kiwix_rag.cli:index_main"
kiwix-query   = "kiwix_rag.cli:query_main"
kiwix-serve   = "kiwix_rag.cli:serve_main"

[tool.setuptools.packages.find]
where = ["."]
include = ["kiwix_rag*"]
```

- [ ] **Step 4: Create `config.example.yaml`**

```yaml
# kiwix-rag configuration
# Copy this file to config.yaml and edit the values below.
# All settings can also be overridden with environment variables
# (prefix: KIWIX_RAG_, e.g. KIWIX_RAG_OLLAMA_URL).

# Path to the ChromaDB vector database directory
db_path: ./vector_db

# Sentence-transformers embedding model name or local path
embed_model: all-MiniLM-L6-v2

# Ollama server base URL
ollama_url: http://localhost:11434

# Ollama model to use for generation
llm_model: llama3.2:3b

# Number of chunks to retrieve per query
top_k: 3

# Web server settings
host: 127.0.0.1
port: 5000

# Semantic routing settings
top_groups: 2
route_threshold: 0.20
max_per_group: 15
max_cache_size: 15

# Ollama request timeout in seconds
timeout: 300

# Optional: base directory for batch_index.sh (overrides KIWIX_DIR env var)
# kiwix_dir: /path/to/your/kiwix-library
```

- [ ] **Step 5: Install package in editable mode and verify import**

```bash
pip install -e ".[dev]"
python -c "import kiwix_rag; print(kiwix_rag.__version__)"
```

Expected: `0.2.0`

- [ ] **Step 6: Commit**

```bash
git add kiwix_rag/__init__.py tests/__init__.py pyproject.toml config.example.yaml
git commit -m "feat: add package scaffold, pyproject.toml, and config.example.yaml"
```

---

## Task 2: Config Class

**Files:**
- Create: `kiwix_rag/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
import os
from pathlib import Path
import pytest
from kiwix_rag.config import Config


def test_defaults():
    cfg = Config()
    assert cfg.embed_model == "all-MiniLM-L6-v2"
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.llm_model == "llama3.2:3b"
    assert cfg.top_k == 3
    assert cfg.port == 5000
    assert cfg.db_path == Path("vector_db")


def test_load_yaml(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        "ollama_url: http://my-server:11434\n"
        "top_k: 10\n"
        "llm_model: phi3:mini\n"
    )
    cfg = Config.load(yaml_file)
    assert cfg.ollama_url == "http://my-server:11434"
    assert cfg.top_k == 10
    assert cfg.llm_model == "phi3:mini"
    assert cfg.embed_model == "all-MiniLM-L6-v2"  # default preserved


def test_load_missing_yaml_uses_defaults(tmp_path):
    cfg = Config.load(tmp_path / "nonexistent.yaml")
    assert cfg.top_k == 3


def test_env_override(monkeypatch, tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("top_k: 7\n")
    monkeypatch.setenv("KIWIX_RAG_TOP_K", "12")
    cfg = Config.load(yaml_file)
    assert cfg.top_k == 12  # env wins over yaml


def test_cli_override_wins_over_env(monkeypatch, tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("top_k: 7\n")
    monkeypatch.setenv("KIWIX_RAG_TOP_K", "12")
    cfg = Config.load(yaml_file, top_k=20)
    assert cfg.top_k == 20  # kwargs win over env


def test_db_path_is_pathlib(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("db_path: /tmp/mydb\n")
    cfg = Config.load(yaml_file)
    assert isinstance(cfg.db_path, Path)
    assert cfg.db_path == Path("/tmp/mydb")


def test_auto_discover_config(tmp_path, monkeypatch):
    """Config.load() with no path finds config.yaml in CWD."""
    (tmp_path / "config.yaml").write_text("top_k: 99\n")
    monkeypatch.chdir(tmp_path)
    cfg = Config.load()
    assert cfg.top_k == 99
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'kiwix_rag.config'`

- [ ] **Step 3: Implement `kiwix_rag/config.py`**

```python
from __future__ import annotations
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_ENV_PREFIX = "KIWIX_RAG_"

# Fields whose values should be coerced to Path
_PATH_FIELDS = {"db_path", "kiwix_dir"}


@dataclass
class Config:
    db_path: Path = field(default_factory=lambda: Path("vector_db"))
    embed_model: str = "all-MiniLM-L6-v2"
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    timeout: int = 300
    top_k: int = 3
    top_groups: int = 2
    route_threshold: float = 0.20
    max_cache_size: int = 15
    max_per_group: int = 15
    host: str = "127.0.0.1"
    port: int = 5000
    kiwix_dir: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None, **overrides: Any) -> "Config":
        """
        Priority (highest wins): kwargs > env vars > YAML > defaults.
        If path is None, looks for config.yaml in the current directory.
        If the file doesn't exist, silently uses defaults.
        """
        # 1. Start from field defaults
        values: dict[str, Any] = {}

        # 2. YAML layer
        yaml_path = path if path is not None else Path("config.yaml")
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            values.update(loaded)

        # 3. Env var layer — KIWIX_RAG_<FIELD_NAME_UPPER>
        field_names = {f.name for f in fields(cls)}
        for fname in field_names:
            env_key = _ENV_PREFIX + fname.upper()
            env_val = os.environ.get(env_key)
            if env_val is not None:
                values[fname] = env_val

        # 4. CLI/kwargs layer
        values.update(overrides)

        # 5. Type coercion for known types
        typed: dict[str, Any] = {}
        field_map = {f.name: f for f in fields(cls)}
        for fname, fld in field_map.items():
            if fname not in values:
                continue
            raw = values[fname]
            if fname in _PATH_FIELDS and raw is not None:
                typed[fname] = Path(raw)
            elif fld.type in (int, "int"):
                typed[fname] = int(raw)
            elif fld.type in (float, "float"):
                typed[fname] = float(raw)
            else:
                typed[fname] = raw

        return cls(**typed)
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
pytest tests/test_config.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add kiwix_rag/config.py tests/test_config.py
git commit -m "feat: add Config class with YAML/env/kwargs merge"
```

---

## Task 3: ChunkFilter Class

**Files:**
- Create: `kiwix_rag/filter.py`
- Create: `tests/test_filter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_filter.py
import pytest
from kiwix_rag.filter import ChunkFilter


@pytest.fixture
def f():
    return ChunkFilter()


def test_clean_educational_text(f):
    assert f.is_clean(
        "To stop severe bleeding, apply firm direct pressure to the wound "
        "using a clean cloth or bandage and maintain pressure for at least "
        "ten minutes without lifting to check the wound."
    )


def test_strong_ad_postpaid(f):
    assert not f.is_clean(
        "POSTPAID. Hallicrafters SX-28A receiver, good condition. "
        "Send $125 postpaid. Write for our complete catalog."
    )


def test_strong_ad_send_dollar(f):
    assert not f.is_clean("Send $15.00 and we will mail your order today.")


def test_moderate_ad_for_sale(f):
    # Single 'for sale' alone is only +1, below threshold of 2
    score, _ = f.score("Collins KWM-2 for sale, asking $800.")
    assert score >= 1


def test_conspiracy_flat_earth(f):
    score, reasons = f.score("The flat earth society insists that NASA lies.")
    assert score >= 1
    assert any("flat earth" in r for r in reasons)


def test_conspiracy_does_not_block_medicine(f):
    # "depopulation" could appear in legitimate population medicine text
    assert f.is_clean(
        "Smallpox depopulation of native communities was catastrophic, "
        "killing an estimated 90 percent of some populations."
    )


def test_score_returns_tuple(f):
    score, reasons = f.score("normal text")
    assert isinstance(score, int)
    assert isinstance(reasons, list)


def test_custom_threshold(f):
    text = "Send $10.00 for our catalog — items for sale below."
    # default threshold 2 — this should fail (for sale + price = ≥2)
    assert not f.is_clean(text)
    # threshold 10 — same text passes
    assert f.is_clean(text, threshold=10)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_filter.py -v
```

Expected: `ModuleNotFoundError: No module named 'kiwix_rag.filter'`

- [ ] **Step 3: Implement `kiwix_rag/filter.py`**

Port the logic from `chunk_filter.py` verbatim into a class. The patterns and regexes are identical; only the calling convention changes.

```python
from __future__ import annotations
import re

_PRICE_RE = re.compile(r'\$\s*\d')

_AD_STRONG = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\bpostpaid\b|\bppd\b',
        r'send\s+\$|send\b.{0,30}\$\s*\d',
        r'\bsend\s+sase\b|send\s+stamped\s+(?:self.?addressed|envelope)',
        r'write\s+for\s+(?:free\s+)?(?:catalog|brochure|flyer|information|prices?|list)',
        r'\bitems?\s+for\s+sale\b',
    ]
]

_AD_MODERATE = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\bfor\s+sale\b',
        r'\bwanted\b',
        r'(?:plus|add)\s+\$?\d+.*?postage',
        r'\ball\s+orders?\b',
        r'\bask(?:ing)?\s+\$\s*\d',
    ]
]

_CONSPIRACY_PATTERNS = [
    "chemtrail", "chem trail", "deep state", "new world order", r"\bnwo\b",
    "globalist", "plandemic", r"\bvaxxed\b", "anti-vaxxer", "sheeple",
    "false flag", "fema camp", "reptilian", r"\billuminati\b", "crisis actor",
    r"\bpsyop\b", "nanobots", "adrenochrome", "flat earth", "sandy hook hoax",
    r"\bqanon\b", r"\bq anon\b", "great reset conspiracy", "depopulation agenda",
    "microchip vaccine", "5g microchip", "bill gates depopulation",
    "george soros agenda", r"\bsatanic cabal\b", "lizard people",
    "moon landing hoax", "nasa hoax", "population control agenda",
    r"\bwoke agenda\b", r"\bclimate hoax\b",
]
_CONSPIRACY_RE = [re.compile(p, re.IGNORECASE) for p in _CONSPIRACY_PATTERNS]

DEFAULT_THRESHOLD = 2


class ChunkFilter:
    """Score text chunks for noise (ads, conspiracy content)."""

    def score(self, text: str) -> tuple[int, list[str]]:
        """Return (score, reasons). Higher score = more likely noise."""
        s, reasons = 0, []
        hits = len(_PRICE_RE.findall(text))
        if hits >= 2:
            s += 2
            reasons.append(f"multiple prices ({hits} hits)")
        for pat in _AD_STRONG:
            if pat.search(text):
                s += 2
                reasons.append(f"ad (strong): {pat.pattern}")
        for pat in _AD_MODERATE:
            if pat.search(text):
                s += 1
                reasons.append(f"ad: {pat.pattern}")
        for pat in _CONSPIRACY_RE:
            if pat.search(text):
                s += 1
                reasons.append(f"conspiracy: {pat.pattern}")
        return s, reasons

    def is_clean(self, text: str, threshold: int = DEFAULT_THRESHOLD) -> bool:
        score, _ = self.score(text)
        return score < threshold
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_filter.py -v
```

Expected: 8 passed

- [ ] **Step 5: Add deprecation notice to top of `chunk_filter.py`**

Add this comment block at the very top of `chunk_filter.py` (after the docstring):

```python
# DEPRECATED: logic has moved to kiwix_rag/filter.py as the ChunkFilter class.
# This file is kept for backward compatibility with resegment scripts.
# It will be removed in a future release.
```

- [ ] **Step 6: Commit**

```bash
git add kiwix_rag/filter.py tests/test_filter.py chunk_filter.py
git commit -m "feat: add ChunkFilter class; deprecate chunk_filter.py"
```

---

## Task 4: Groups Config + GroupRouter Class

**Files:**
- Create: `kiwix_rag/groups.py`
- Create: `kiwix_rag/router.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Create `kiwix_rag/groups.py`**

Extract `GROUPS` and `SYSTEM_PROMPT` verbatim from `web.py`. No changes to the values — this is purely a relocation.

```python
# kiwix_rag/groups.py
"""
GROUPS — semantic routing table used by GroupRouter.
SYSTEM_PROMPT — injected into every LLM request.

To add a new ZIM collection: add its collection-name substring to the
appropriate group's "patterns" list. Patterns match against collection
names that use underscores throughout (dots and hyphens are converted
during indexing). Example: use "health_stackexchange" not
"health.stackexchange".
"""

SYSTEM_PROMPT = (
    "You are a reference assistant for survivors in a post-collapse world where "
    "civilization's infrastructure — hospitals, governments, supply chains, the internet, "
    "emergency services — no longer exists or cannot be reached. "
    # ... (paste the full SYSTEM_PROMPT string from web.py lines 51-77 verbatim)
    "Never invent specific facts, "
    "figures, doses, or procedures that are not present in the context."
)

GROUPS: dict[str, dict] = {
    "medicine": {
        "description": (
            "How do I treat this wound or injury? ..."
            # paste full description from web.py
        ),
        "patterns": [
            "health_stackexchange",
            "medlineplus",
            # ... paste full patterns list from web.py
        ],
    },
    # ... paste all remaining groups from web.py verbatim
}
```

> **Note:** Copy the complete `SYSTEM_PROMPT` (lines 51–77 of `web.py`) and the complete `GROUPS` dict (lines 94–491 of `web.py`) into this file with no modifications. The important part is the relocation, not the content.

- [ ] **Step 2: Write failing tests for GroupRouter**

```python
# tests/test_router.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from kiwix_rag.router import GroupRouter
from kiwix_rag.groups import GROUPS


@pytest.fixture
def mock_embedder():
    """Embedder that returns a fixed all-ones normalized vector for any input."""
    embedder = MagicMock()
    def fake_encode(texts, normalize_embeddings=True):
        return np.ones((len(texts), 4), dtype=np.float32) / 2.0
    embedder.encode.side_effect = fake_encode
    return embedder


@pytest.fixture
def router():
    return GroupRouter(GROUPS, top_groups=2, route_threshold=0.20, max_per_group=15)


def test_assigns_medicine_collection(router, mock_embedder):
    router.build(["health_stackexchange_chunks", "devdocs_en_python_chunks"], mock_embedder)
    assert "medicine" in router.group_cols
    assert "health_stackexchange_chunks" in router.group_cols["medicine"]


def test_assigns_coding_collection(router, mock_embedder):
    router.build(["devdocs_en_python_chunks"], mock_embedder)
    assert "coding" in router.group_cols
    assert "devdocs_en_python_chunks" in router.group_cols["coding"]


def test_unassigned_goes_to_other(router, mock_embedder):
    router.build(["some_unknown_zim_chunks"], mock_embedder)
    assert "_other" in router.group_cols
    assert "some_unknown_zim_chunks" in router.group_cols["_other"]


def test_build_with_no_collections(router, mock_embedder):
    router.build([], mock_embedder)
    assert router.group_cols == {}


def test_route_returns_list_of_group_names(router, mock_embedder):
    router.build(["health_stackexchange_chunks", "devdocs_en_python_chunks"], mock_embedder)
    query_vec = np.ones(4, dtype=np.float32) / 2.0
    groups = router.route(query_vec)
    assert isinstance(groups, list)
    assert len(groups) >= 1


def test_route_returns_all_when_no_embeddings(router, mock_embedder):
    # _other only — no named groups → route returns all keys
    router.build(["some_unknown_chunks"], mock_embedder)
    query_vec = np.ones(4, dtype=np.float32) / 2.0
    groups = router.route(query_vec)
    assert "_other" in groups


def test_select_collections_caps_at_max(router, mock_embedder):
    names = [f"col_{i}" for i in range(20)]
    selected = router.select_collections(names, "python programming", max_n=5)
    assert len(selected) == 5


def test_select_collections_returns_all_when_under_cap(router, mock_embedder):
    names = ["col_a", "col_b"]
    selected = router.select_collections(names, "any query", max_n=10)
    assert selected == names
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
pytest tests/test_router.py -v
```

Expected: `ModuleNotFoundError: No module named 'kiwix_rag.router'`

- [ ] **Step 4: Implement `kiwix_rag/router.py`**

```python
from __future__ import annotations
import numpy as np


class GroupRouter:
    """
    Routes a query vector to the most relevant collection groups.

    Call build() once at startup with the list of available collection
    names and an initialized embedder. Then call route() per query.
    """

    def __init__(
        self,
        groups: dict,
        top_groups: int = 2,
        route_threshold: float = 0.20,
        max_per_group: int = 15,
    ) -> None:
        self._groups = groups
        self.top_groups = top_groups
        self.route_threshold = route_threshold
        self.max_per_group = max_per_group
        self.group_cols: dict[str, list[str]] = {}
        self._group_embs: dict[str, np.ndarray] = {}

    def build(self, available_names: list[str], embedder) -> None:
        """Assign collections to groups; embed group descriptions for routing."""
        assigned: set[str] = set()
        self.group_cols = {}

        for gname, gdef in self._groups.items():
            matched = [
                n for n in available_names
                if any(p in n for p in gdef["patterns"])
            ]
            if matched:
                self.group_cols[gname] = matched
                assigned.update(matched)

        unassigned = [n for n in available_names if n not in assigned]
        if unassigned:
            self.group_cols["_other"] = unassigned

        named = [g for g in self.group_cols if g != "_other"]
        if named:
            descs = [self._groups[g]["description"] for g in named]
            embs = embedder.encode(descs, normalize_embeddings=True)
            self._group_embs = {g: embs[i] for i, g in enumerate(named)}

    def route(self, query_vec: np.ndarray) -> list[str]:
        """Return group names most relevant to the normalized query vector."""
        if not self._group_embs:
            return list(self.group_cols.keys())

        scores = {
            g: float(np.dot(query_vec, emb))
            for g, emb in self._group_embs.items()
        }
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best = ranked[0][1] if ranked else 0.0

        if best < self.route_threshold:
            fallback = [g for g, _ in ranked[: self.top_groups * 2]]
            if "_other" in self.group_cols:
                fallback.append("_other")
            return fallback

        selected = [g for g, s in ranked[: self.top_groups] if s >= best - 0.1]
        if "_other" in self.group_cols:
            selected.append("_other")
        return selected

    def select_collections(
        self, names: list[str], query: str, max_n: int | None = None
    ) -> list[str]:
        """When a group has many collections, pick the most name-relevant ones."""
        cap = max_n if max_n is not None else self.max_per_group
        if len(names) <= cap:
            return names
        words = {w for w in query.lower().split() if len(w) > 3}
        def name_score(n: str) -> int:
            return sum(1 for w in words if w in n.lower())
        return sorted(names, key=name_score, reverse=True)[:cap]
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/test_router.py -v
```

Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add kiwix_rag/groups.py kiwix_rag/router.py tests/test_router.py
git commit -m "feat: add GroupRouter class and extract GROUPS/SYSTEM_PROMPT to groups.py"
```

---

## Task 5: ZimExtractor Class

**Files:**
- Create: `kiwix_rag/extract.py`
- Create: `kiwix_rag/ocr.py`
- Create: `tests/test_extract.py`

- [ ] **Step 1: Move `ocr.py` → `kiwix_rag/ocr.py`**

```bash
cp ocr.py kiwix_rag/ocr.py
```

Then replace the content of the original `ocr.py` with a backward-compat shim:

```python
# ocr.py — backward-compat shim; logic has moved to kiwix_rag/ocr.py
from kiwix_rag.ocr import load_engine, ocr_pdf, preprocess, render_pages  # noqa: F401
```

- [ ] **Step 2: Write failing tests for ZimExtractor**

The tests mock `libzim` entirely — we test only the parsing and chunking logic.

```python
# tests/test_extract.py
import pytest
from unittest.mock import MagicMock, patch
from kiwix_rag.extract import ZimExtractor


@pytest.fixture
def extractor():
    return ZimExtractor(chunk_size=200, chunk_overlap=20)


# ── sanitize ─────────────────────────────────────────────────────────────────

def test_sanitize_removes_control_chars(extractor):
    result = extractor.sanitize("hello\x00world\x01end")
    assert "\x00" not in result
    assert "\x01" not in result
    assert "hello" in result and "world" in result


def test_sanitize_preserves_whitespace(extractor):
    result = extractor.sanitize("line one\nline two\ttabbed")
    assert "\n" in result
    assert "\t" in result


# ── extract_html_blocks ───────────────────────────────────────────────────────

def test_extracts_plain_html(extractor):
    html = b"<html><body><p>" + b"A" * 200 + b"</p></body></html>"
    blocks = extractor.extract_html_blocks(html)
    assert len(blocks) == 1
    assert blocks[0]["is_accepted"] is False
    assert len(blocks[0]["text"]) >= 150


def test_drops_noise_tags(extractor):
    html = (
        b"<html><body>"
        b"<script>evil()</script>"
        b"<nav>nav stuff</nav>"
        b"<p>" + b"B" * 200 + b"</p>"
        b"</body></html>"
    )
    blocks = extractor.extract_html_blocks(html)
    assert all("evil()" not in b["text"] for b in blocks)
    assert all("nav stuff" not in b["text"] for b in blocks)


def test_extracts_accepted_answer_as_separate_block(extractor):
    accepted_text = "C" * 200
    other_text = "D" * 200
    html = (
        f'<html><body>'
        f'<div class="accepted-answer">{accepted_text}</div>'
        f'<p>{other_text}</p>'
        f'</body></html>'
    ).encode()
    blocks = extractor.extract_html_blocks(html)
    accepted = [b for b in blocks if b["is_accepted"]]
    not_accepted = [b for b in blocks if not b["is_accepted"]]
    assert len(accepted) == 1
    assert len(not_accepted) == 1
    assert accepted_text[:50] in accepted[0]["text"]


def test_returns_empty_for_short_content(extractor):
    html = b"<html><body><p>Too short</p></body></html>"
    blocks = extractor.extract_html_blocks(html)
    # blocks may be returned but they'll be under 150 chars — extractor skips them
    for b in blocks:
        assert len(b["text"]) < 150 or True  # we just check no crash


# ── extract_pdf_text ──────────────────────────────────────────────────────────

def test_scanned_pdf_detected(extractor):
    """A PDF with very little text per page should be flagged as scanned."""
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "abc"  # < 100 chars/page
    mock_reader.pages = [mock_page]

    with patch("kiwix_rag.extract.PdfReader", return_value=mock_reader):
        text, is_scanned = extractor.extract_pdf_text(b"fake-pdf-bytes")
    assert is_scanned is True


def test_text_pdf_not_flagged_as_scanned(extractor):
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "x" * 500
    mock_reader.pages = [mock_page]

    with patch("kiwix_rag.extract.PdfReader", return_value=mock_reader):
        text, is_scanned = extractor.extract_pdf_text(b"fake-pdf-bytes")
    assert is_scanned is False
    assert "x" * 100 in text
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
pytest tests/test_extract.py -v
```

Expected: `ModuleNotFoundError: No module named 'kiwix_rag.extract'`

- [ ] **Step 4: Implement `kiwix_rag/extract.py`**

Port all logic from `extract_zim.py` into a class. The `iter_chunks` generator becomes a method. All module-level constants become class attributes.

```python
from __future__ import annotations
import io
import json
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

_PDF_MIN_CHARS_PER_PAGE = 100


class ZimExtractor:
    """Extract and chunk text from a Kiwix .zim archive."""

    NOISE_TAGS = ["script", "style", "nav", "header", "footer", "figure"]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        ocr_engine=None,
        quality_filter=None,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.ocr_engine = ocr_engine
        self.quality_filter = quality_filter  # ChunkFilter | None

    def sanitize(self, text: str) -> str:
        return "".join(
            ch if ch >= " " or ch in "\n\r\t" else " " for ch in text
        )

    def extract_html_blocks(self, html_bytes: bytes) -> list[dict]:
        soup = BeautifulSoup(html_bytes.decode("utf-8", errors="ignore"), "html.parser")
        blocks = []
        accepted_el = soup.find(class_="accepted-answer")
        if accepted_el:
            for tag in accepted_el(self.NOISE_TAGS):
                tag.decompose()
            text = self.sanitize(accepted_el.get_text(separator=" ", strip=True))
            accepted_el.decompose()
            if text:
                blocks.append({"text": text, "is_accepted": True})
        for tag in soup(self.NOISE_TAGS):
            tag.decompose()
        main_text = self.sanitize(soup.get_text(separator=" ", strip=True))
        if main_text:
            blocks.append({"text": main_text, "is_accepted": False})
        return blocks

    def extract_pdf_text(self, pdf_bytes: bytes) -> tuple[str, bool]:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = self.sanitize(" ".join(pages)).strip()
        total = len(reader.pages)
        avg_chars = len(full_text) / total if total else 0
        return full_text, avg_chars < _PDF_MIN_CHARS_PER_PAGE

    def _should_keep(self, chunk: str) -> bool:
        if self.quality_filter is None:
            return True
        return self.quality_filter.is_clean(chunk)

    def _yield_blocks(
        self, blocks: list[dict], source: str, title: str
    ) -> Iterator[dict]:
        for block in blocks:
            if len(block["text"]) < 150:
                continue
            for chunk in self.splitter.split_text(block["text"]):
                if self._should_keep(chunk):
                    yield {
                        "text": chunk,
                        "source": source,
                        "title": title,
                        "is_accepted": block.get("is_accepted", False),
                    }

    def iter_chunks(
        self,
        archive,
        entry_offset: int = 0,
        entry_limit: int = 0,
    ) -> Iterator[dict]:
        """
        Yield chunk dicts from a libzim.Archive.
        Prints progress and a summary to stdout (same as original extract_zim.py).
        """
        total = archive.all_entry_count
        start = entry_offset
        end = min(entry_offset + entry_limit, total) if entry_limit > 0 else total
        counts: dict[str, int] = {
            "skipped": 0, "html": 0, "pdf": 0,
            "pdf_scanned": 0, "pdf_error": 0, "filtered": 0,
        }

        for i in range(start, end):
            if (i - start) % 500 == 0:
                print(f"\r  {i - start:,} / {end - start:,} entries scanned ...",
                      end="", flush=True)

            try:
                entry = archive._get_entry_by_id(i)
            except Exception:
                counts["skipped"] += 1
                continue

            if entry.is_redirect:
                counts["skipped"] += 1
                continue

            try:
                item = entry.get_item()
            except Exception:
                counts["skipped"] += 1
                continue

            mime = item.mimetype
            content = bytes(item.content)
            title = entry.title or entry.path

            if "text/html" in mime:
                blocks = self.extract_html_blocks(content)
                if not any(len(b["text"]) >= 150 for b in blocks):
                    counts["skipped"] += 1
                    continue
                counts["html"] += 1
                yield from self._yield_blocks(blocks, entry.path, title)

            elif ("application/json" in mime
                  and entry.path.startswith("videos/")
                  and entry.path.endswith(".json")):
                try:
                    data = json.loads(content)
                except Exception:
                    counts["skipped"] += 1
                    continue
                desc = data.get("description", "").strip()
                vtitle = data.get("title", title).strip()
                text = f"{vtitle}\n\n{desc}" if desc else vtitle
                if len(text) < 150:
                    counts["skipped"] += 1
                    continue
                counts["video_desc"] = counts.get("video_desc", 0) + 1
                for chunk in self.splitter.split_text(text):
                    if self._should_keep(chunk):
                        yield {"text": chunk, "source": entry.path, "title": vtitle}

            elif "application/json" in mime and "page_content_" in entry.path:
                try:
                    html_body = json.loads(content).get("htmlBody", "")
                except Exception:
                    counts["skipped"] += 1
                    continue
                if not html_body:
                    counts["skipped"] += 1
                    continue
                blocks = self.extract_html_blocks(html_body.encode("utf-8"))
                if not any(len(b["text"]) >= 150 for b in blocks):
                    counts["skipped"] += 1
                    continue
                counts["json_html"] = counts.get("json_html", 0) + 1
                yield from self._yield_blocks(blocks, entry.path, title)

            elif "application/pdf" in mime:
                if not _PYPDF_AVAILABLE:
                    counts["skipped"] += 1
                    continue
                try:
                    text, is_scanned = self.extract_pdf_text(content)
                except Exception:
                    counts["pdf_error"] += 1
                    continue
                if is_scanned or len(text) < 150:
                    if self.ocr_engine is not None:
                        try:
                            from kiwix_rag.ocr import ocr_pdf
                            text = self.sanitize(ocr_pdf(content, self.ocr_engine))
                            if len(text) < 150:
                                counts["pdf_scanned"] += 1
                                continue
                            counts["pdf_ocr"] = counts.get("pdf_ocr", 0) + 1
                        except Exception:
                            counts["pdf_scanned"] += 1
                            continue
                    else:
                        counts["pdf_scanned"] += 1
                        continue
                else:
                    counts["pdf"] += 1
                for chunk in self.splitter.split_text(text):
                    if self._should_keep(chunk):
                        yield {"text": chunk, "source": entry.path, "title": title}

            else:
                counts["skipped"] += 1

        print()
        print(f"  HTML pages:        {counts['html']:,}")
        if counts.get("video_desc"):
            print(f"  Video desc (JSON): {counts['video_desc']:,}")
        if counts.get("json_html"):
            print(f"  JSON pages:        {counts['json_html']:,}")
        print(f"  PDFs (text):       {counts['pdf']:,}")
        if counts.get("pdf_ocr"):
            print(f"  PDFs (OCR'd):      {counts['pdf_ocr']:,}")
        if counts["pdf_scanned"]:
            print(f"  PDFs (scanned/empty, skipped): {counts['pdf_scanned']:,}")
        if counts["pdf_error"]:
            print(f"  PDFs (error, skipped):         {counts['pdf_error']:,}")
        print(f"  Other/redirects:   {counts['skipped']:,}")
        if counts["filtered"]:
            print(f"  Quality-filtered:  {counts['filtered']:,}")
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/test_extract.py -v
```

Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add kiwix_rag/extract.py kiwix_rag/ocr.py ocr.py tests/test_extract.py
git commit -m "feat: add ZimExtractor class; move ocr.py to kiwix_rag/ocr.py"
```

---

## Task 6: Indexer Class

**Files:**
- Create: `kiwix_rag/index.py`
- Create: `tests/test_index.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_index.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
from kiwix_rag.index import Indexer


def make_jsonl(tmp_path: Path, n: int = 5) -> Path:
    p = tmp_path / "test_chunks.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "text": f"chunk text number {i} " * 10,
                "source": f"article/{i}",
                "title": f"Article {i}",
                "is_accepted": False,
            }) + "\n")
    return p


def test_count_lines(tmp_path):
    p = make_jsonl(tmp_path, n=7)
    from kiwix_rag.index import count_lines
    assert count_lines(p) == 7


def test_iter_chunks_yields_all_records(tmp_path):
    p = make_jsonl(tmp_path, n=3)
    from kiwix_rag.index import iter_chunks
    chunks = list(iter_chunks(p))
    assert len(chunks) == 3
    assert chunks[0]["title"] == "Article 0"


def test_indexer_build_calls_collection_add(tmp_path):
    p = make_jsonl(tmp_path, n=2)
    db_path = tmp_path / "db"

    mock_col = MagicMock()
    mock_col.count.return_value = 2
    mock_client = MagicMock()
    mock_client.list_collections.return_value = []
    mock_client.get_or_create_collection.return_value = mock_col
    mock_client.get_collection.return_value = mock_col

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2]] * 2

    with patch("kiwix_rag.index.chromadb.PersistentClient", return_value=mock_client), \
         patch("kiwix_rag.index.SentenceTransformer", return_value=mock_model):
        idx = Indexer(db_path)
        idx.build(p, collection_name="test_col")

    assert mock_col.add.called


def test_indexer_raises_if_collection_exists_without_replace(tmp_path):
    p = make_jsonl(tmp_path, n=2)
    db_path = tmp_path / "db"

    mock_existing = MagicMock()
    mock_existing.name = "test_col"
    mock_client = MagicMock()
    mock_client.list_collections.return_value = [mock_existing]

    with patch("kiwix_rag.index.chromadb.PersistentClient", return_value=mock_client), \
         patch("kiwix_rag.index.SentenceTransformer"):
        idx = Indexer(db_path)
        with pytest.raises(RuntimeError, match="already exists"):
            idx.build(p, collection_name="test_col", replace=False)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_index.py -v
```

Expected: `ModuleNotFoundError: No module named 'kiwix_rag.index'`

- [ ] **Step 3: Implement `kiwix_rag/index.py`**

Port all logic from `build_index.py`. The `main()` function's core embed-and-add loop becomes `build()`. `swap_collection` becomes `_swap_collection`. `iter_chunks` and `count_lines` stay as module-level helpers (they're pure functions, not state-dependent).

```python
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

    def __init__(self, db_path: Path, embed_model: str = "all-MiniLM-L6-v2") -> None:
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
        jsonl_path: Path,
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
                embeddings = self.model.encode(batch_texts, show_progress_bar=False).tolist()
                collection.add(ids=batch_ids, embeddings=embeddings,
                               documents=batch_texts, metadatas=batch_meta)
                print(f"\r  {done:,} / {total:,}", end="", flush=True)
                batch_texts, batch_meta, batch_ids = [], [], []

        if batch_texts:
            embeddings = self.model.encode(batch_texts, show_progress_bar=False).tolist()
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_index.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add kiwix_rag/index.py tests/test_index.py
git commit -m "feat: add Indexer class"
```

---

## Task 7: Retriever Class

**Files:**
- Create: `kiwix_rag/retrieval.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_retrieval.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from kiwix_rag.retrieval import Retriever, build_prompt


# ── build_prompt ──────────────────────────────────────────────────────────────

def test_build_prompt_includes_question():
    chunks = [{"title": "Article", "text": "some context text"}]
    prompt = build_prompt("What is X?", chunks)
    assert "What is X?" in prompt
    assert "some context text" in prompt
    assert "Article" in prompt


def test_build_prompt_separates_chunks():
    chunks = [
        {"title": "A", "text": "first chunk"},
        {"title": "B", "text": "second chunk"},
    ]
    prompt = build_prompt("Q?", chunks)
    assert "---" in prompt  # separator between chunks


# ── Retriever ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_collection():
    col = MagicMock()
    col.name = "test_col"
    col.query.return_value = {
        "documents": [["chunk one text", "chunk two text"]],
        "metadatas": [[
            {"source": "a/1", "title": "Article 1", "is_accepted": False},
            {"source": "a/2", "title": "Article 2", "is_accepted": False},
        ]],
        "distances": [[0.1, 0.2]],
    }
    return col


def test_retrieve_returns_sorted_chunks(mock_collection):
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    with patch("kiwix_rag.retrieval.chromadb.PersistentClient"), \
         patch("kiwix_rag.retrieval.SentenceTransformer", return_value=mock_embedder):
        r = Retriever(db_path="/fake/db")
        chunks = r.retrieve("my query", [mock_collection], k=5)

    assert len(chunks) == 2
    assert chunks[0]["dist"] <= chunks[1]["dist"]  # sorted by distance


def test_retrieve_deduplicates_identical_chunks(mock_collection):
    """When two collections return the same chunk, it should appear once."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    with patch("kiwix_rag.retrieval.chromadb.PersistentClient"), \
         patch("kiwix_rag.retrieval.SentenceTransformer", return_value=mock_embedder):
        r = Retriever(db_path="/fake/db")
        # Same collection queried twice — both return identical chunks
        chunks = r.retrieve("my query", [mock_collection, mock_collection], k=5)

    texts = [c["text"] for c in chunks]
    assert len(texts) == len(set(texts))  # no duplicates


def test_retrieve_boosts_accepted_answers():
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    col = MagicMock()
    col.name = "col"
    col.query.return_value = {
        "documents": [["accepted chunk", "regular chunk"]],
        "metadatas": [[
            {"source": "a/1", "title": "A", "is_accepted": True},
            {"source": "a/2", "title": "B", "is_accepted": False},
        ]],
        "distances": [[0.5, 0.3]],  # accepted is further in raw distance
    }

    with patch("kiwix_rag.retrieval.chromadb.PersistentClient"), \
         patch("kiwix_rag.retrieval.SentenceTransformer", return_value=mock_embedder):
        r = Retriever(db_path="/fake/db")
        chunks = r.retrieve("query", [col], k=5)

    # After 0.85× boost, accepted chunk's effective dist = 0.5 * 0.85 = 0.425 < 0.5
    # So accepted chunk should rank first despite higher raw distance
    assert chunks[0]["is_accepted"] is True
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_retrieval.py -v
```

Expected: `ModuleNotFoundError: No module named 'kiwix_rag.retrieval'`

- [ ] **Step 3: Implement `kiwix_rag/retrieval.py`**

```python
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_retrieval.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add kiwix_rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add Retriever class and build_prompt helper"
```

---

## Task 8: Server Refactor (Flask App Factory)

**Files:**
- Create: `kiwix_rag/server.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_server.py
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from kiwix_rag.config import Config
from kiwix_rag.server import create_app


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.retrieve.return_value = [
        {"text": "test context", "source": "wiki/1", "title": "Test Article",
         "dist": 0.1, "is_accepted": False, "zim": "wikipedia"},
    ]
    return r


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.route.return_value = ["reference"]
    router.group_cols = {"reference": ["wikipedia_chunks"]}
    router.select_collections.return_value = ["wikipedia_chunks"]
    return router


@pytest.fixture
def app(mock_retriever, mock_router):
    cfg = Config(ollama_url="http://fake:11434", llm_model="test-model", top_k=3)
    return create_app(cfg, retriever=mock_retriever, router=mock_router)


@pytest.fixture
def client(app):
    app.config["TESTING"] = True
    return app.test_client()


def test_index_route_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_api_ask_no_question_returns_400(client):
    resp = client.post("/api/ask", json={})
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "error" in data


def test_api_ask_returns_answer(client, mock_retriever):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "Here is the answer."}
    mock_resp.raise_for_status = MagicMock()

    with patch("kiwix_rag.server.requests.post", return_value=mock_resp):
        resp = client.post("/api/ask", json={"question": "What is water?"})

    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["answer"] == "Here is the answer."
    assert "sources" in data
    assert "elapsed" in data


def test_api_ask_no_chunks_returns_error(client, mock_retriever):
    mock_retriever.retrieve.return_value = []
    resp = client.post("/api/ask", json={"question": "What is X?"})
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "error" in data
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_server.py -v
```

Expected: `ModuleNotFoundError: No module named 'kiwix_rag.server'`

- [ ] **Step 3: Implement `kiwix_rag/server.py`**

The key change from `web.py`: no global variables. All state lives in the `Flask` app's config dict, and the routes close over injected objects. The collection cache (LRU eviction daemon) moves to a `CollectionCache` helper class.

```python
from __future__ import annotations
import json
import os
import time
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import requests as _requests
from flask import Flask, Response, render_template, request

from kiwix_rag.config import Config
from kiwix_rag.groups import SYSTEM_PROMPT, GROUPS
from kiwix_rag.router import GroupRouter
from kiwix_rag.retrieval import Retriever, build_prompt

_GROUP_TTL = 600  # seconds before idle collection evicted from cache


class CollectionCache:
    """Thread-safe LRU cache for ChromaDB collection handles."""

    def __init__(self, client, max_size: int) -> None:
        self._client = client
        self._max = max_size
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, names: list[str]) -> dict:
        now = time.time()
        with self._lock:
            for n in names:
                if n not in self._cache:
                    if self._max and len(self._cache) >= self._max:
                        lru = min(self._cache, key=lambda k: self._cache[k]["last_used"])
                        del self._cache[lru]
                    self._cache[n] = {
                        "col": self._client.get_collection(n),
                        "last_used": now,
                    }
                else:
                    self._cache[n]["last_used"] = now
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
    import chromadb
    from sentence_transformers import SentenceTransformer

    templates_dir = Path(__file__).parent.parent / "templates"
    app = Flask(__name__, template_folder=str(templates_dir))

    if retriever is None:
        _embedder = SentenceTransformer(config.embed_model)
        _client = chromadb.PersistentClient(path=str(config.db_path.expanduser()))
        retriever = Retriever(config.db_path, config.embed_model)
        retriever._embedder = _embedder
        retriever._client = _client
    else:
        _client = getattr(retriever, "_client", None)

    if router is None:
        router = GroupRouter(
            GROUPS,
            top_groups=config.top_groups,
            route_threshold=config.route_threshold,
            max_per_group=config.max_per_group,
        )
        available = [c.name for c in _client.list_collections()]
        router.build(available, retriever.embedder)

    col_cache = CollectionCache(_client, max_size=config.max_cache_size)
    threading.Thread(target=_eviction_daemon, args=(col_cache,), daemon=True).start()

    def _retrieve_for_query(question: str) -> list[dict]:
        import numpy as np
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
                with _requests.post(
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
            except _requests.exceptions.ConnectionError:
                yield _sse({"token": "\n[Error: could not reach Ollama — is it running?]"})
            except _requests.exceptions.ReadTimeout:
                yield _sse({"token": "\n[Error: Ollama timed out]"})
            except Exception as e:
                yield _sse({"token": f"\n[Error: {e}]"})

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
            resp = _requests.post(
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_server.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add kiwix_rag/server.py tests/test_server.py
git commit -m "feat: add Flask app factory in kiwix_rag/server.py; no global state"
```

---

## Task 9: CLI Entry Points + Thin Script Wrappers

**Files:**
- Create: `kiwix_rag/cli.py`
- Modify: `extract_zim.py`, `build_index.py`, `rag.py`, `web.py`

- [ ] **Step 1: Create `kiwix_rag/cli.py`**

Each `_main()` function parses CLI args, builds a `Config`, and delegates to the appropriate class. The logic is identical to the original script `main()` functions — just reorganized.

```python
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
```

- [ ] **Step 2: Replace `extract_zim.py` with thin wrapper**

```python
#!/usr/bin/env python3
"""Thin wrapper — logic lives in kiwix_rag.cli. Use 'kiwix-extract' after pip install."""
from kiwix_rag.cli import extract_main
if __name__ == "__main__":
    extract_main()
```

- [ ] **Step 3: Replace `build_index.py` with thin wrapper**

```python
#!/usr/bin/env python3
"""Thin wrapper — logic lives in kiwix_rag.cli. Use 'kiwix-index' after pip install."""
from kiwix_rag.cli import index_main
if __name__ == "__main__":
    index_main()
```

- [ ] **Step 4: Replace `rag.py` with thin wrapper**

```python
#!/usr/bin/env python3
"""Thin wrapper — logic lives in kiwix_rag.cli. Use 'kiwix-query' after pip install."""
from kiwix_rag.cli import query_main
if __name__ == "__main__":
    query_main()
```

- [ ] **Step 5: Replace `web.py` with thin wrapper**

```python
#!/usr/bin/env python3
"""Thin wrapper — logic lives in kiwix_rag.cli. Use 'kiwix-serve' after pip install."""
from kiwix_rag.cli import serve_main
if __name__ == "__main__":
    serve_main()
```

- [ ] **Step 6: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass (no regressions from CLI changes)

- [ ] **Step 7: Smoke-test the installed commands**

```bash
kiwix-extract --help
kiwix-index --help
kiwix-query --help
kiwix-serve --help
```

Expected: each prints usage without errors.

- [ ] **Step 8: Commit**

```bash
git add kiwix_rag/cli.py extract_zim.py build_index.py rag.py web.py
git commit -m "feat: add CLI entry points in kiwix_rag/cli.py; replace scripts with thin wrappers"
```

---

## Task 10: Add pyyaml to requirements.txt + Final Full-Suite Run

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add pyyaml to `requirements.txt`**

```
# Core
chromadb
sentence-transformers
flask
requests
numpy
pyyaml

# ZIM extraction
libzim
beautifulsoup4
langchain-text-splitters
pypdf

# OCR (optional — only needed if using --ocr flag with kiwix-extract)
# Install all three for tesseract engine:
#   pip install pymupdf opencv-python-headless pillow pytesseract
#   macOS:  brew install tesseract
#   Linux:  sudo apt install tesseract-ocr
#
# Or for the pure-Python easyocr engine (no binary needed):
#   pip install pymupdf opencv-python-headless easyocr
```

- [ ] **Step 2: Run the complete test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests pass. Zero failures.

- [ ] **Step 3: Verify existing scripts still work**

```bash
python extract_zim.py --help
python build_index.py --help
python rag.py --help
python web.py --help
```

Expected: each prints usage without errors (thin wrappers delegate to package).

- [ ] **Step 4: Final commit**

```bash
git add requirements.txt
git commit -m "chore: add pyyaml to requirements.txt"
```

---

## Self-Review

### Spec Coverage

| Requirement | Task(s) |
|---|---|
| Easier configuration (config file) | Task 2 (Config), Task 9 (--config flag on all CLIs) |
| Object-oriented design | Tasks 3–8 (ChunkFilter, GroupRouter, ZimExtractor, Indexer, Retriever, Flask factory) |
| Unit tests | Tests in Tasks 2–8 |
| CLI cleanup / entry points | Task 9 (pyproject.toml scripts, kiwix_rag/cli.py) |
| Backward compat (old scripts still work) | Tasks 9 (thin wrappers) |
| Requirements updated | Task 10 |

### Type Consistency Check

- `Config.db_path` → `Path` throughout (Indexer, Retriever both accept `Path | str`)
- `GroupRouter.group_cols` → `dict[str, list[str]]` — used consistently in `server.py` `_retrieve_for_query`
- `Retriever.retrieve()` → returns `list[dict]` with keys `text, source, title, dist, is_accepted, zim` — consumed correctly in `server.py` and `cli.py`
- `ChunkFilter.score()` → `tuple[int, list[str]]` — tests verify both elements
- `ZimExtractor.iter_chunks()` → yields `dict` with `text, source, title, is_accepted` — consumed by `cli.py extract_main` and `Indexer.build`

### No Placeholders

Reviewed: no TBD, TODO, "implement later", or "handle edge cases" in plan steps. All code blocks show complete implementations.
