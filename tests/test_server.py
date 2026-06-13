# tests/test_server.py
import json
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


def test_ask_sse_streams_tokens(client):
    def fake_post(*args, **kwargs):
        class FakeResp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def raise_for_status(self):
                pass
            def iter_lines(self):
                import json
                yield json.dumps({"response": "Hello", "done": False}).encode()
                yield json.dumps({"response": " world", "done": True}).encode()
        return FakeResp()

    with patch("kiwix_rag.server.requests.post", side_effect=fake_post):
        resp = client.get("/ask?q=test+question")

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "Hello" in body
    assert "world" in body
