#!/usr/bin/env python3
"""Postgres + pgvector client replacing chromadb.PersistentClient.

Exposes the same operations web.py / build_index.py / rag.py call on the
ChromaDB client: list_collections, get_collection, count, query, upsert,
delete_collection, create_collection. Schema is one partitioned `chunks`
table — one partition per collection, one HNSW index per partition.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

EMBED_DIM = 384

_PARTITION_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _partition_name(collection: str) -> str:
    """Map a collection name to a safe Postgres partition table name."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", collection)
    if not _PARTITION_NAME_RE.fullmatch(safe):
        raise ValueError(f"invalid collection name: {collection!r}")
    return f"col_{safe}"


class PgClient:
    """Drop-in replacement for chromadb.PersistentClient."""

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 5):
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            open=True,
            configure=self._configure_conn,
        )
        self.ensure_schema()

    @staticmethod
    def _configure_conn(conn: psycopg.connection.Connection) -> None:
        register_vector(conn)

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> "PgClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def ensure_schema(self) -> None:
        """Create the partitioned chunks table + registry if absent. Idempotent."""
        with self._pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          text NOT NULL,
                    collection  text NOT NULL,
                    embedding   vector({EMBED_DIM}) NOT NULL,
                    document    text NOT NULL,
                    source      text NOT NULL,
                    title       text NOT NULL,
                    is_accepted boolean DEFAULT false,
                    PRIMARY KEY (collection, id)
                ) PARTITION BY LIST (collection)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collections_registry (
                    collection    text PRIMARY KEY,
                    vector_count  int,
                    imported_at   timestamptz,
                    hnsw_built    boolean DEFAULT false
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migration_errors (
                    collection  text NOT NULL,
                    chunk_id    text NOT NULL,
                    error       text NOT NULL,
                    at          timestamptz DEFAULT now(),
                    PRIMARY KEY (collection, chunk_id)
                )
            """)
