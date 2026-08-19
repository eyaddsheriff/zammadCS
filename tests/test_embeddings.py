"""Embedding client guards.

Every test here covers a failure that produces *wrong* results rather than
an error: vectors paired with the wrong chunk, or silently mixed models.
Retrieval would just get mysteriously worse.
"""

import pytest

from zammad_agent.config import EmbeddingConfig
from zammad_agent.llm.embeddings import EmbeddingClient, EmbeddingError


class FakeResponse:
    def __init__(self, payload, ok=True, status_code=200):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Stands in for requests.Session, returning a scripted payload."""

    def __init__(self, response):
        self.response = response
        self.headers = {}
        self.posted = []

    def post(self, url, json=None, timeout=None):
        self.posted.append(json)
        return self.response

    def close(self):
        pass


def _client(response, batch_size=32):
    config = EmbeddingConfig(base_url="http://localhost:11434/v1", api_key="k", model="bge-m3")
    client = EmbeddingClient(config, batch_size=batch_size)
    client._session = FakeSession(response)
    return client


def _data(vectors, indices=None):
    indices = indices if indices is not None else range(len(vectors))
    return {"data": [{"index": i, "embedding": v} for i, v in zip(indices, vectors)]}


def test_empty_input_makes_no_request():
    client = _client(FakeResponse(_data([])))
    assert client.embed([]) == []
    assert client._session.posted == []


def test_returns_vectors_in_input_order():
    client = _client(FakeResponse(_data([[1.0], [2.0]])))
    assert client.embed(["a", "b"]) == [[1.0], [2.0]]


def test_reorders_by_response_index():
    # Order is not guaranteed by the API, which is why it sends an index.
    # Trusting array order would pair every chunk with someone else's vector.
    client = _client(FakeResponse(_data([[2.0], [1.0]], indices=[1, 0])))
    assert client.embed(["a", "b"]) == [[1.0], [2.0]]


def test_rejects_missing_index_field():
    payload = {"data": [{"embedding": [1.0]}]}
    client = _client(FakeResponse(payload))
    with pytest.raises(EmbeddingError, match="index"):
        client.embed(["a"])


def test_rejects_count_mismatch():
    # A short response would shift every later vector against the wrong chunk.
    client = _client(FakeResponse(_data([[1.0]])))
    with pytest.raises(EmbeddingError, match="asked for 2"):
        client.embed(["a", "b"])


def test_rejects_inconsistent_dimensions():
    # Means the provider switched models mid-run; storing these together
    # produces an index that only fails later, at query time.
    client = _client(FakeResponse(_data([[1.0], [1.0, 2.0]])))
    with pytest.raises(EmbeddingError, match="inconsistent"):
        client.embed(["a", "b"])


def test_batches_large_inputs():
    client = _client(FakeResponse(_data([[1.0], [1.0]])), batch_size=2)
    client.embed(["a", "b", "c", "d"])
    assert len(client._session.posted) == 2


def test_model_is_exposed_for_index_tagging():
    # The index records this so it cannot later be queried with another model.
    client = _client(FakeResponse(_data([])))
    assert client.model == "bge-m3"
