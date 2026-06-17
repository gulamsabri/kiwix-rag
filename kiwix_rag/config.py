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
    max_cache_bytes: int = 11_000_000_000
    # When process RSS exceeds this, the server resets the ChromaDB client to free
    # accumulated HNSW segments (ChromaDB has no working segment eviction). Keep it
    # below MemoryMax minus one query's working set. 0 disables the reset.
    memory_high_water_bytes: int = 6_000_000_000
    max_per_group: int = 15
    # Pin the server to a subset of collections (None = serve every collection
    # in the DB). Used for relevance and to avoid loading giant indexes while
    # testing/debugging.
    collections: list[str] | None = None
    # Exclude collections with more than this many vectors from the searchable
    # set (and empty collections). 0 disables the limit.
    max_collection_size: int = 0
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
        values: dict[str, Any] = {}

        # YAML layer
        yaml_path = path if path is not None else Path("config.yaml")
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                try:
                    loaded = yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML in {yaml_path}: {e}") from e
            values.update(loaded)

        # Env var layer — KIWIX_RAG_<FIELD_NAME_UPPER>
        field_names = {f.name for f in fields(cls)}
        for fname in field_names:
            env_key = _ENV_PREFIX + fname.upper()
            env_val = os.environ.get(env_key)
            if env_val is not None:
                values[fname] = env_val

        # CLI/kwargs layer
        values.update(overrides)

        # Type coercion
        typed: dict[str, Any] = {}
        field_map = {f.name: f for f in fields(cls)}
        for fname, fld in field_map.items():
            if fname not in values:
                continue
            raw = values[fname]
            if fname in _PATH_FIELDS and raw is not None:
                typed[fname] = Path(raw)
            elif fld.type in (int, "int"):
                try:
                    typed[fname] = int(raw)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Config field '{fname}' must be an integer, got {raw!r}"
                    ) from None
            elif fld.type in (float, "float"):
                try:
                    typed[fname] = float(raw)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Config field '{fname}' must be a float, got {raw!r}"
                    ) from None
            else:
                typed[fname] = raw

        return cls(**typed)
