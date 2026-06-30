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
