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
        "distances": [[0.5, 0.45]],  # accepted is further in raw distance
    }

    with patch("kiwix_rag.retrieval.chromadb.PersistentClient"), \
         patch("kiwix_rag.retrieval.SentenceTransformer", return_value=mock_embedder):
        r = Retriever(db_path="/fake/db")
        chunks = r.retrieve("query", [col], k=5)

    # After 0.85x boost: accepted effective dist = 0.5 * 0.85 = 0.425 < 0.5
    # So accepted chunk should rank first despite higher raw distance
    assert chunks[0]["is_accepted"] is True


# ── ChromaDB client reset (the real memory bound) ───────────────────────────────

def test_reset_client_clears_cache_and_rebuilds():
    """reset_client must clear ChromaDB's system cache (frees loaded segments)
    and return a fresh client — the only thing that actually bounds memory in
    chromadb 1.5.x."""
    mock_embedder = MagicMock()
    with patch("kiwix_rag.retrieval.chromadb.PersistentClient") as mock_client, \
         patch("kiwix_rag.retrieval.SharedSystemClient") as mock_shared, \
         patch("kiwix_rag.retrieval.SentenceTransformer", return_value=mock_embedder):
        r = Retriever(db_path="/fake/db")
        first = r.client
        new = r.reset_client()

    mock_shared.clear_system_cache.assert_called_once()
    assert new is r.client
    # a brand new client object was constructed for the rebuild
    assert mock_client.call_count == 2


def test_process_rss_bytes_returns_int():
    """RSS probe returns a non-negative int (0 where /proc is unavailable)."""
    from kiwix_rag.retrieval import process_rss_bytes
    val = process_rss_bytes()
    assert isinstance(val, int)
    assert val >= 0
