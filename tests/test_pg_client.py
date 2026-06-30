import numpy as np
import pytest
from pg_client import PgClient


@pytest.fixture
def client(pg_dsn):
    c = PgClient(pg_dsn)
    yield c
    # clean up any collections we created
    for name in c.list_collections():
        c.delete_collection(name)
    c.close()


def test_create_list_count_delete(client: PgClient):
    client.create_collection("alpha")
    client.create_collection("beta")
    assert set(client.list_collections()) == {"alpha", "beta"}

    handle = client.get_collection("alpha")
    assert handle.name == "alpha"
    assert handle.count() == 0

    client.delete_collection("alpha")
    assert "alpha" not in client.list_collections()
    assert client.count("beta") == 0


def _vec(seed: float, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(int(seed))
    v = rng.random(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).tolist()


def test_upsert_idempotent(client: PgClient):
    client.create_collection("docs")
    h = client.get_collection("docs")
    metas = [{"source": "a.md", "title": "A", "is_accepted": False},
             {"source": "b.md", "title": "B", "is_accepted": True}]
    h.upsert(["1", "2"], [_vec(1.0), _vec(2.0)], ["doc one", "doc two"], metas)
    assert h.count() == 2
    # Re-upsert same ids — count must be unchanged (ON CONFLICT update)
    h.upsert(["1", "2"], [_vec(1.0), _vec(2.0)], ["doc one v2", "doc two v2"], metas)
    assert h.count() == 2
    # Document was updated
    results = h.query(_vec(1.0), k=1)
    assert results[0]["document"] == "doc one v2"


def test_query_returns_expected_fields(client: PgClient):
    client.create_collection("q")
    h = client.get_collection("q")
    h.upsert(["1"], [_vec(5.0)], ["the doc"], [{"source": "s.md", "title": "T", "is_accepted": True}])
    results = h.query(_vec(5.0), k=1)
    assert len(results) == 1
    r = results[0]
    assert set(r.keys()) == {"document", "source", "title", "is_accepted", "dist"}
    assert r["document"] == "the doc"
    assert r["source"] == "s.md"
    assert r["title"] == "T"
    assert r["is_accepted"] is True
    # Distance to self (normalized vectors, cosine) should be ~0
    assert r["dist"] < 0.01
