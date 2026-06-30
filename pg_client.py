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
import numpy as np
from psycopg import sql
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

    def list_collections(self) -> list[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT collection FROM collections_registry ORDER BY collection"
            ).fetchall()
        return [r[0] for r in rows]

    def count(self, name: str) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT vector_count FROM collections_registry WHERE collection = %s", (name,)
            ).fetchone()
        return int(row[0]) if row else 0

    def create_collection(self, name: str) -> "CollectionHandle":
        pname = _partition_name(name)
        with self._pool.connection() as conn:
            conn.execute("BEGIN")
            exists = conn.execute(
                "SELECT 1 FROM collections_registry WHERE collection = %s", (name,)
            ).fetchone()
            if exists:
                conn.execute("ROLLBACK")
                return self.get_collection(name)
            conn.execute(
                sql.SQL("CREATE TABLE {} PARTITION OF chunks FOR VALUES IN ({})").format(
                    sql.Identifier(pname), sql.Literal(name)
                )
            )
            conn.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops)").format(
                    sql.Identifier(f"idx_{pname}_hnsw"), sql.Identifier(pname)
                )
            )
            conn.execute(
                "INSERT INTO collections_registry (collection, vector_count, imported_at, hnsw_built) "
                "VALUES (%s, 0, NULL, false)", (name,)
            )
            conn.execute("COMMIT")
        return self.get_collection(name)

    def delete_collection(self, name: str) -> None:
        pname = _partition_name(name)
        with self._pool.connection() as conn:
            conn.execute("BEGIN")
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(pname)))
            conn.execute("DELETE FROM collections_registry WHERE collection = %s", (name,))
            conn.execute("DELETE FROM migration_errors WHERE collection = %s", (name,))
            conn.execute("COMMIT")

    def get_collection(self, name: str) -> "CollectionHandle":
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM collections_registry WHERE collection = %s", (name,)
            ).fetchone()
        if not row:
            raise KeyError(f"collection not found: {name!r}")
        return CollectionHandle(self._pool, name)


class CollectionHandle:
    """Wraps a collection name + pooled psycopg connection.

    Exposes .name, .count(), .query(), .upsert() — matching the shape of the
    ChromaDB collection objects web.py:retrieve() calls.
    """

    def __init__(self, pool: ConnectionPool, name: str):
        self._pool = pool
        self.name = name

    def count(self) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE collection = %s", (self.name,)
            ).fetchone()
        return int(row[0])

    def upsert(self, ids: list[str], embeddings: list[list[float]],
               documents: list[str], metadatas: list[dict]) -> None:
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("ids, embeddings, documents, metadatas must be equal length")
        if not ids:
            return
        rows = [
            (
                str(ids[i]),
                self.name,
                np.asarray(embeddings[i], dtype=np.float32),
                str(documents[i]),
                str(metadatas[i].get("source", "")),
                str(metadatas[i].get("title", "")),
                bool(metadatas[i].get("is_accepted", False)),
            )
            for i in range(len(ids))
        ]
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (id, collection, embedding, document, source, title, is_accepted)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (collection, id) DO UPDATE SET
                        embedding   = EXCLUDED.embedding,
                        document    = EXCLUDED.document,
                        source      = EXCLUDED.source,
                        title       = EXCLUDED.title,
                        is_accepted = EXCLUDED.is_accepted
                    """,
                    rows,
                )
            conn.execute(
                "UPDATE collections_registry SET vector_count = "
                "(SELECT COUNT(*) FROM chunks WHERE collection = %s) "
                "WHERE collection = %s",
                (self.name, self.name),
            )

    def query(self, embedding: list[float], k: int = 3) -> list[dict]:
        vec = np.asarray(embedding, dtype=np.float32)
        with self._pool.connection() as conn:
            register_vector(conn)
            rows = conn.execute(
                """
                SELECT document, source, title, is_accepted,
                       embedding <=> %s AS dist
                FROM chunks
                WHERE collection = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vec, self.name, vec, k),
            ).fetchall()
        return [
            {
                "document": r[0],
                "source": r[1],
                "title": r[2],
                "is_accepted": r[3],
                "dist": float(r[4]),
            }
            for r in rows
        ]
