# tests/test_index.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from kiwix_rag.index import Indexer, count_lines, iter_chunks


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
    assert count_lines(p) == 7


def test_iter_chunks_yields_all_records(tmp_path):
    p = make_jsonl(tmp_path, n=3)
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
