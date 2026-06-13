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
