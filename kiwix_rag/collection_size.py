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
            try:
                rows = con.execute(
                    "SELECT c.name, s.id FROM segments s "
                    "JOIN collections c ON s.collection = c.id"
                ).fetchall()
            finally:
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
